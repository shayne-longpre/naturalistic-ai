import sys
import argparse
import pandas as pd
sys.path.append("./")
sys.path.append("./src")
from src.scripts.utils import batch_generator, append_jsonl, load_existing_exid_turn_pairs
import json 
from langdetect import detect
# Preliminaries
sys.path.append("./")

def format_conversation_turns_free(conversation):
    pairs = []
    for i in range(0, len(conversation) - 1, 2):
        user_turn = conversation[i]
        assistant_turn = conversation[i + 1] if i + 1 < len(conversation) else None
        if user_turn["role"] == "user" or user_turn["role"] == "human" and assistant_turn and assistant_turn["role"] == "assistant" or assistant_turn["role"] == "gpt":
            pairs.append((user_turn["content"], assistant_turn["content"]))

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

    if len(conversation) == 1:
        print("Processing a single-turn conversation.", flush=True)
        single_turn = conversation[0]
        print(single_turn, flush=True)
        pairs.append((single_turn["text"], ""))
    for i in range(0, len(conversation) - 1, 2):
        print(f"Processing conversation turn {i} and {i + 1}.", flush = True)
        user_turn = conversation[i]
        assistant_turn = conversation[i + 1] if i + 1 < len(conversation) else None
        print(f"User turn: {user_turn}, Assistant turn: {assistant_turn}", flush = True)
        if user_turn["role"] == "user" or user_turn["role"] == "human" and assistant_turn and assistant_turn["role"] == "assistant" or assistant_turn["role"] == "gpt":
            pairs.append((user_turn["content"], assistant_turn["content"]))

    print(f"Extracted {len(pairs)} conversation pairs.", flush = True)
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
    print(f"Formatted {len(formatted_turns)} conversation turns.", flush = True)
    return formatted_turns


def extract_samples_and_metadata(args, dataframe, existing_pairs):
    sample, metadata, turn_ids = [], [], []

    for _, row in dataframe.iterrows():
        conversation = row["conversation"]
        print(f"Conversation: {conversation}", flush = True)

        if args.input_format.lower() == "json":
            formatted_pairs = format_conversation_turns_json(conversation)
        elif args.input_format.lower() == "free":
            formatted_pairs = format_conversation_turns_free(conversation)
        else:
            raise ValueError("input_format needs to be either 'json' or 'free'.")

        for i, (prev_text, curr_text) in enumerate(formatted_pairs):
            if (row["conversation_id"], i) in existing_pairs:
                continue
            
            prompt = "" + prev_text + "\n" + curr_text
            sample.append(prompt)
            metadata.append({
                "conversation_id": row["conversation_id"],
                "turn": i,
                "dataset_id": row["dataset_id"],
                "model": row["model"]
            })
            turn_ids.append(i)
    return sample, metadata, turn_ids

####### New / Basic Functions for Annoations ########

def get_string_length(x): 
    return len(x)

def get_word_count(x): 
    return len(x.split())

def get_char_count(x):
    return len(x.replace(" ", ""))

def get_language(x):
    try:
        lang = detect(x)
    except:
        lang = "unknown"
    return lang


def run_simple_annotations(args, batch_size=1, verbose = True):
    existing_ex_ids = load_existing_exid_turn_pairs(args.save)
    if verbose:
        print(f"Loaded {len(existing_ex_ids)} existing example-turn pairs from {args.save}.")
        
    dataframe = pd.read_json(args.input, orient="records")

    if verbose:
        print(f"Loaded {dataframe.shape[0]} records from {args.input}.", flush = True)
        
    formatted_prompts, metadata, turn_ids = extract_samples_and_metadata(args, dataframe, existing_ex_ids)
    if verbose:
        print(f"Formatted prompts: {len(formatted_prompts)}, Metadata: {len(metadata)}")
    
    if not formatted_prompts:
        print("All examples already exist in the save file.")
        return

    for batch, meta_batch, turn_id_batch in zip(
        batch_generator(formatted_prompts, batch_size),
        batch_generator(metadata, batch_size),
        batch_generator(turn_ids, batch_size)
    ):
        if verbose:
            print("Processing batch with first prompt:")
            print(batch[0])
        
        
        ### TODO: fix this to match 'label' and 'confidence' formatting in the annotations. 
        batch_responses = [[
            {
            "String Length": get_string_length(prompt), 
            "Word Count": get_word_count(prompt),
            "Character Count": get_char_count(prompt),
            "Language": get_language(prompt)       
            }]
            for prompt in batch
        ]
    
        batch_output = []
        for response, meta, turn_id in zip(batch_responses, meta_batch, turn_id_batch):
            if verbose:
                print("Response:", response)

            response_entry = {
                **meta,
                "model_id": "None", # TODO: check if this breaks annotation processing. 
                "level_id": args.level_id,
                "prompt_id": args.prompt_id,
                "turn": turn_id,
                "input": batch[0],
                "response": json.dumps(response, ensure_ascii=False, indent=2) #batch_responses
            }
            batch_output.append(response_entry)

        if batch_output:
            append_jsonl(batch_output, args.save)
            if verbose:
                print(f"Appended {len(batch_output)} entries to {args.save}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input json file.")
    parser.add_argument("--input_format", type=str, required=True, default=None)
    parser.add_argument("--level_id", type=str, required=True, default=None)
    parser.add_argument("--prompt_id", type=str, required=True, default=None)
    parser.add_argument("--save", type=str, required=True, help="Save path.")
    args = parser.parse_args()

    run_simple_annotations(args)
    