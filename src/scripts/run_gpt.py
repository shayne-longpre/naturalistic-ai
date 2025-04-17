import sys
import asyncio
import argparse
import pandas as pd
from helpers import gpt
from utils import *

# Preliminaries
sys.path.append("./")


def make_prompt(args):
    TASK_DESCRIPTION = read_json("src/scripts/instructions.json")
    OPTIONS = read_json("src/scripts/taxonomy_options.json")

    PREAMBLE = f"""You are a high-quality annotation assistant. Your task is to annotate conversation logs between users and AI chatbots. You will be given a specific task description, all possible label options for the task, and a part of the conversation, including the user prompt and model response from both previous and current turns. These might be pulled from any part of a multi-turn conversation. As a high-quality annotator you will diligently provide annotations on the current turn that are:
1. Comprehensive: You will list all relevant annotations as the tasks can be multi-class (only one label is true) or multi-label (multiple categories can be true at once). Pay special attention to subtle, or implied properties of the input conversation. 
2. Precise: You must answer as a JSON list of dictionaries of exact labels name(s) and confidence (a score between 0 and 1) without any additional explanation, reasoning, or text.
3. Calibrated: Reflect appropriate confidence in your annotation. If the input is ambiguous or open to multiple interpretations, acknowledge that explicitly.
The conversation log is enclosed between <START_CONVERSATION> and <END_CONVERSATION> tags. The previous conversation turn is enclosed as JSON between <START_PREVIOUS_TURN> and <END_PREVIOUS_TURN> tags, while the current conversation turn is enclosed as JSON between <START_CURRENT_TURN> and <END_CURRENT_TURN> tags.
The previous conversation turn is provided for context; however, generate an annotation only for the current {args.level_id}. 
Only use information present or inferable from the input. Avoid hallucinations or unjustified assumptions."""

    JSON_INSTRUCTION = f"Return the answer as a JSON list of dictionaries, each with the fields 'labels' (exact label name(s) before ':' and after '- ') and 'confidence' (a score between 0 and 1). Do not include any explanation, reasoning, or additional text."

    task_description = TASK_DESCRIPTION[args.level_id][args.prompt_id]
    options = OPTIONS[args.level_id][args.prompt_id]

    prompt = f"""{PREAMBLE}

Task description: {task_description}
Options: {options}

<START_CONVERSATION>

<START_PREVIOUS_TURN>
{{prev_text}}
<END_PREVIOUS_TURN>

<START_CURRENT_TURN>
{{curr_text}}
<END_CURRENT_TURN>

<END_CONVERSATION>

{JSON_INSTRUCTION}
Response: """
    return prompt


def format_conversation_turns_free(conversation):
    pairs = []
    for i in range(0, len(conversation) - 1, 2):
        user_turn = conversation[i]
        assistant_turn = conversation[i + 1] if i + 1 < len(conversation) else None
        if user_turn["role"] == "user" or user_turn["role"] == "human" and assistant_turn and assistant_turn["role"] == "assistant" or assistant_turn["role"] == "gpt":
            pairs.append((user_turn["text"], assistant_turn["text"]))

    formatted_turns = []
    for i in range(len(pairs)):
        # Previous turn
        if i == 0:
            prev_user = "None"
            prev_assistant = "None"
        else:
            prev_user, prev_assistant = pairs[i - 1]

        # Current turn
        curr_user, curr_assistant = pairs[i]

        prev_text = f"Previous user prompt: {prev_user}\nPrevious model response: {prev_assistant}"
        curr_text = f"Current user prompt: {curr_user}\nCurrent model response: {curr_assistant}"

        formatted_turns.append((prev_text, curr_text))
    return formatted_turns


def format_conversation_turns_json(conversation):
    pairs = []
    for i in range(0, len(conversation) - 1, 2):
        user_turn = conversation[i]
        assistant_turn = conversation[i + 1] if i + 1 < len(conversation) else None
        if user_turn["role"] == "user" or user_turn["role"] == "human" and assistant_turn and assistant_turn["role"] == "assistant" or assistant_turn["role"] == "gpt":
            pairs.append((user_turn["text"], assistant_turn["text"]))

    formatted_turns = []
    for i in range(len(pairs)):
        # Previous turn
        if i == 0:
            prev_user = "None"
            prev_assistant = "None"
        else:
            prev_user, prev_assistant = pairs[i - 1]

        # Current turn
        curr_user, curr_assistant = pairs[i]

        prev_text = json.dumps({
            "Previous user prompt": prev_user,
            "Previous model response": prev_assistant
        }, ensure_ascii=False, indent=2)

        curr_text = json.dumps({
            "Current user prompt": curr_user,
            "Current model response": curr_assistant
        }, ensure_ascii=False, indent=2)

        formatted_turns.append((prev_text, curr_text))
    return formatted_turns



def extract_samples_and_metadata(args, dataframe, existing_pairs):
    sample, metadata, turn_ids = [], [], []

    for _, row in dataframe.iterrows():
        conversation = row["conversation"]
        if args.input_format.lower() == "json":
            formatted_pairs = format_conversation_turns_json(conversation)
        elif args.input_format.lower() == "free":
            formatted_pairs = format_conversation_turns_free(conversation)
        else:
            raise ValueError("input_format needs to be either 'json' or 'free'.")

        for i, (prev_text, curr_text) in enumerate(formatted_pairs):
            if (row["ex_id"], i) in existing_pairs:
                continue

            prompt = make_prompt(args)
            prompt = prompt.replace("{prev_text}", prev_text).replace("{curr_text}", curr_text)

            sample.append(prompt)
            metadata.append({
                "ex_id": row["ex_id"],
                "turn": i,
                "dataset_id": row["dataset_id"],
                "model": row["model"]
            })
            turn_ids.append(i)
    return sample, metadata, turn_ids


async def run_gpt(args, batch_size=1):
    existing_ex_ids = load_existing_exid_turn_pairs(args.save)   
    dataframe = pd.read_json(args.input, orient="records")
    formatted_prompts, metadata, turn_ids = extract_samples_and_metadata(args, dataframe, existing_ex_ids)
    print(f"Formatted prompts: {len(formatted_prompts)}, Metadata: {len(metadata)}")
    
    if not formatted_prompts:
        print("All examples already exist in the save file.")
        return

    gpt_instance = gpt.GPT(model=args.model_id, prompt=args.prompt_id)

    for batch, meta_batch, turn_id_batch in zip(
        batch_generator(formatted_prompts, batch_size),
        batch_generator(metadata, batch_size),
        batch_generator(turn_ids, batch_size)
    ):
        try:
            batch_responses = await gpt_instance.process_prompts_in_batches_async(batch)
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue
        
        batch_output = []
        print("Prompt: ", batch[0])
        for response, meta, turn_id in zip(batch_responses, meta_batch, turn_id_batch):
            print("Response: ", response)

            if response is None:
                print(f"[Error] Skipping saving due to failed response for turn {turn_id}, ex_id: {meta['ex_id']}")
                with open(args.save.replace('-failed.jsonl', '-failed2.jsonl'), 'a') as f:
                    f.write(json.dumps({
                        **meta,
                        "model_id": args.model_id,
                        "level_id": args.level_id,
                        "prompt_id": args.prompt_id,
                        "turn": turn_id,
                        "input": batch[0]
                    }, ensure_ascii=False) + '\n')
                continue
            else:
                response_entry = {
                    **meta,
                    "model_id": args.model_id,
                    "level_id": args.level_id,
                    "prompt_id": args.prompt_id,
                    "turn": turn_id,
                    "input": batch[0],
                    "response": response
                }
                batch_output.append(response_entry)

            if batch_output:
                append_jsonl(batch_output, args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input json file.")
    parser.add_argument("--input_format", type=str, required=True, default=None)
    parser.add_argument("--level_id", type=str, required=True, default=None)
    parser.add_argument("--prompt_id", type=str, required=True, default=None)
    parser.add_argument("--model_id", type=str, required=True, default=None, choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o3-mini"])
    parser.add_argument("--save", type=str, required=True, help="Save path.")
    args = parser.parse_args()

    asyncio.run(run_gpt(args))