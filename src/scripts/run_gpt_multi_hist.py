import os
import sys
import asyncio
import argparse
import pandas as pd

# Preliminaries
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from src.helpers import gpt, io


def make_prompt(args, include_prev_turn=True):
    TASK_DESCRIPTION = io.read_json("src/scripts/instructions.json")
    OPTIONS = io.read_json("src/scripts/taxonomy_options.json")

    PREAMBLE = f"""You are a high-quality annotation assistant. Your task is to annotate conversation logs between users and AI chatbots. You will be given a specific task description, all possible label options for the task, and a part of the conversation, including the user prompt and model response from both previous and current turns. These might be pulled from any part of a multi-turn conversation. As a high-quality annotator you will diligently provide annotations on the current turn that are:
1. Comprehensive: You will list all relevant annotations as the tasks can be multi-class (only one label is true) or multi-label (multiple categories can be true at once). Pay special attention to subtle, or implied properties of the input conversation. 
2. Precise: You must answer as a JSON list of dictionaries of exact labels name(s) and confidence (a score between 0 and 1) without any additional explanation, reasoning, or text.
3. Calibrated: Reflect appropriate confidence in your annotation. If the input is ambiguous or open to multiple interpretations, acknowledge that explicitly.
The conversation log is enclosed between <START_CONVERSATION> and <END_CONVERSATION> tags. When available, one or more previous conversation turns are included as JSON between <START_PREVIOUS_TURN> and <END_PREVIOUS_TURN> to provide context, unless the current turn is the first in the conversation. Multiple previous conversation turns may be included if their combined length is not too long. The current turn is always enclosed between <START_CURRENT_TURN> and <END_CURRENT_TURN>.
While the previous conversation turn may be provided for context, generate an annotation only for the current {args.level_id}. 
Only use information present or inferable from the input. Avoid hallucinations or unjustified assumptions."""

    JSON_INSTRUCTION = f"Return the answer as a JSON list of dictionaries, each with the fields 'labels' (exact label name(s) before ':' and after '- ') and 'confidence' (a score between 0 and 1). Do not include any explanation, reasoning, or additional text."

    task_description = TASK_DESCRIPTION[args.level_id][args.prompt_id]
    options = OPTIONS[args.level_id][args.prompt_id]

    previous_turn_block = (
    "<START_PREVIOUS_TURN>\n{prev_text}\n<END_PREVIOUS_TURN>\n\n"
        if include_prev_turn else ""
    )

    prompt = f"""{PREAMBLE}

Task description: {task_description}
Options: {options}

<START_CONVERSATION>
{previous_turn_block}<START_CURRENT_TURN>
{{curr_text}}
<END_CURRENT_TURN>

<END_CONVERSATION>

{JSON_INSTRUCTION}
Response: """
    return prompt


def format_conversation_turns_free(conversation, args, max_prev_chars=300):
    pairs = []
    turn_ids = []
    for i in range(0, len(conversation) - 1, 2):
        user_turn = conversation[i]
        assistant_turn = conversation[i + 1] if i + 1 < len(conversation) else None
        if user_turn["role"] in ["user", "human"] and assistant_turn and assistant_turn["role"] in ["assistant", "gpt"]:
            pairs.append((user_turn["content"], assistant_turn["content"]))
            turn_ids.append(user_turn["turn"])

    formatted_turns = []
    for i in range(len(pairs)):
        prev_text = "None"

        # Collect previous 2 turns if available
        if i >= 1:
            prev_turns = []
            for offset in [2, 1]:  # check 2-turn-back, then 1-turn-back
                if i - offset >= 0:
                    u, a = pairs[i - offset]
                    prev_turns.append((u, a))

            # Always include most recent previous turn
            most_recent_u, most_recent_a = pairs[i - 1]
            full_prev_text = f"Previous user prompt: {most_recent_u}\nPrevious model response: {most_recent_a}"

            if len(prev_turns) == 2:
                older_u, older_a = prev_turns[0]
                older_text = f"Previous user prompt: {older_u}\nPrevious model response: {older_a}"
                combined = older_text + "\n\n" + full_prev_text
                if len(combined) <= max_prev_chars:
                    prev_text = combined
                else:
                    prev_text = full_prev_text
            else:
                prev_text = full_prev_text

        # Current turn
        curr_user, curr_assistant = pairs[i]
        if args.level_id == "prompt":
            curr_text = f"Current user prompt: {curr_user}"
        else:
            curr_text = f"Current user prompt: {curr_user}\nCurrent model response: {curr_assistant}"

        formatted_turns.append((prev_text, curr_text, turn_ids[i]))
    return formatted_turns



def format_conversation_turns_json(conversation, args, max_prev_chars=300):
    import json
    pairs = []
    turn_ids = []
    for i in range(0, len(conversation) - 1, 2):
        user_turn = conversation[i]
        assistant_turn = conversation[i + 1] if i + 1 < len(conversation) else None
        if user_turn["role"] in ["user", "human"] and assistant_turn and assistant_turn["role"] in ["assistant", "gpt"]:
            pairs.append((user_turn["content"], assistant_turn["content"]))
            turn_ids.append(user_turn["turn"])

    formatted_turns = []
    for i in range(len(pairs)):
        prev_text = "None"

        if i >= 1:
            prev_turns = []
            for offset in [2, 1]:
                if i - offset >= 0:
                    u, a = pairs[i - offset]
                    prev_turns.append((u, a))

            most_recent_u, most_recent_a = pairs[i - 1]
            full_prev_obj = {
                "Previous user prompt": most_recent_u,
                "Previous model response": most_recent_a
            }

            if len(prev_turns) == 2:
                older_u, older_a = prev_turns[0]
                older_obj = {
                    "Previous user prompt 1": older_u,
                    "Previous model response 1": older_a,
                    "Previous user prompt 2": most_recent_u,
                    "Previous model response 2": most_recent_a
                }
                json_str = json.dumps(older_obj, ensure_ascii=False)
                if len(json_str) <= max_prev_chars:
                    prev_text = json.dumps(older_obj, ensure_ascii=False, indent=2)
                else:
                    prev_text = json.dumps(full_prev_obj, ensure_ascii=False, indent=2)
            else:
                prev_text = json.dumps(full_prev_obj, ensure_ascii=False, indent=2)

        # Current turn
        curr_user, curr_assistant = pairs[i]
        if args.level_id == "prompt":
            curr_text = json.dumps({
                "Current user prompt": curr_user
            }, ensure_ascii=False, indent=2)
        else:
            curr_text = json.dumps({
                "Current user prompt": curr_user,
                "Current model response": curr_assistant
            }, ensure_ascii=False, indent=2)

        formatted_turns.append((prev_text, curr_text, turn_ids[i]))

    return formatted_turns



def extract_samples_and_metadata(args, dataframe, existing_pairs):
    sample, metadata, order_ids, turn_ids = [], [], [], []
    ex_id_suffix_counter = defaultdict(int)
    seen_ex_ids = set()

    for _, row in dataframe.iterrows():
        conversation = row["conversation"]
        ex_id_base = row["ex_id"]

        if args.input_format.lower() == "json":
            formatted_pairs = format_conversation_turns_json(conversation, args)
        elif args.input_format.lower() == "free":
            formatted_pairs = format_conversation_turns_free(conversation, args)
        else:
            raise ValueError("input_format needs to be either 'json' or 'free'.")

        # Assign suffix if already annotated before
        if (ex_id_base, 0) in existing_pairs or ex_id_base in seen_ex_ids:
            # Add suffix
            suffix_idx = ex_id_suffix_counter[ex_id_base]
            suffix = string.ascii_lowercase[suffix_idx]
            ex_id = f"{ex_id_base}_{suffix}"
            ex_id_suffix_counter[ex_id_base] += 1
        else:
            ex_id = ex_id_base

        seen_ex_ids.add(ex_id)

        for i, (prev_text, curr_text, turn_id) in enumerate(formatted_pairs):
            if (ex_id, i) in existing_pairs:
                continue

            include_prev = not ("None" in prev_text or i == 0)
            prompt = make_prompt(args, include_prev_turn=include_prev)

            if include_prev:
                prompt = prompt.replace("{prev_text}", prev_text)
            prompt = prompt.replace("{curr_text}", curr_text)

            sample.append(prompt)
            metadata.append({
                "ex_id": ex_id,  # suffixed version
                "order": i,
                "turn": turn_id,
                "dataset_id": row["dataset_id"],
                "model": row["model"]
            })
            order_ids.append(i)
            turn_ids.append(turn_id)

    return sample, metadata, order_ids, turn_ids



async def run_gpt(args, batch_size=1):
    existing_ex_ids = load_existing_exid_turn_pairs(args.save)   
    dataframe = pd.read_json(args.input, orient="records")
    formatted_prompts, metadata, order_ids, turn_ids = extract_samples_and_metadata(args, dataframe, existing_ex_ids)
    print(f"Formatted prompts: {len(formatted_prompts)}, Metadata: {len(metadata)}")
    
    if not formatted_prompts:
        print("All examples already exist in the save file.")
        return

    gpt_instance = gpt.GPT(model=args.model_id, prompt=args.prompt_id)

    for batch, meta_batch, order_id_batch, turn_id_batch in zip(
        batch_generator(formatted_prompts, batch_size),
        batch_generator(metadata, batch_size),
        batch_generator(order_ids, batch_size),
        batch_generator(turn_ids, batch_size)
    ):
        try:
            batch_responses = await gpt_instance.process_prompts_in_batches_async(batch)
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue
        
        batch_output = []
        print("Prompt: ", batch[0])
        for response, meta, order_id, turn_id in zip(batch_responses, meta_batch, order_id_batch, turn_id_batch):
            print("Response: ", response)

            if response is None:
                print(f"[Error] Skipping saving due to failed response for order {order_id}, turn {turn_id}, ex_id: {meta['ex_id']}")
                with open(args.save.replace('.jsonl', '-failed.jsonl'), 'a') as f:
                    f.write(json.dumps({
                        **meta,
                        "model_id": args.model_id,
                        "level_id": args.level_id,
                        "prompt_id": args.prompt_id,
                        "order": order_id,
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
                    "order": order_id,
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