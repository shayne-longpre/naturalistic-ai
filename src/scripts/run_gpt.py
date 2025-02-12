import sys
import asyncio
import json
import pandas as pd
import argparse
from tqdm import tqdm

sys.path.append("./")
sys.path.append("src/")

from helpers import constants, io, gpt

def batch_generator(data_list, batch_size):
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]


def append_jsonl(data, file_path):
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")


def load_existing_ex_ids(save_fpath):
    try:
        existing_data = io.read_jsonl(save_fpath)
        return {entry["ex_id"] for entry in existing_data}
    except FileNotFoundError:
        return set()


def valid_turn(text):
    return pd.notna(text) and text.strip() != ""


def extract_samples_and_metadata(dataframe, system_level_id, system_prompt, system_prompt_template, existing_ex_ids):
    sample, metadata = [], []
    
    level_map = {
        "conversation": lambda row: [
            " ".join(
                f"[{turn['role']}] {turn['text']}" 
                for turn in row["conversation"] if valid_turn(turn["text"])
            )
        ],
        "prompt": lambda row: [
            turn["text"]
            for turn in row["conversation"] if turn["role"] == "user" or turn["role"] == "human" and valid_turn(turn["text"])
        ],
        "response": lambda row: [
            turn["text"]
            for turn in row["conversation"] if turn["role"] == "assistant" or turn["role"] == "gpt" and valid_turn(turn["text"])
        ],
        "turn": lambda row: [
            f"[{row['conversation'][i]['role']}] {row['conversation'][i]['text']} "
            f"[{row['conversation'][i+1]['role']}] {row['conversation'][i+1]['text']}"
            for i in range(len(row['conversation']) - 1)
            if valid_turn(row['conversation'][i]['text']) and valid_turn(row['conversation'][i+1]['text'])
        ],
    }

    for _, row in dataframe.iterrows():
        if row["ex_id"] in existing_ex_ids:
            continue
        
        data_list = []
        
        if system_prompt == "multi_turn_relationship":
            user_turns = [turn["text"] for turn in row["conversation"] if turn["role"] == "user" or turn["role"] == "human" and valid_turn(turn["text"])]
            prev_prompts = []
            
            for current_text in user_turns:
                prev_prompts_str = " ".join(prev_prompts) if prev_prompts else "[NONE]"
                placeholders = {"{prev_prompts}": prev_prompts_str, "{text}": current_text}
                formatted_prompt = system_prompt_template
                for placeholder, value in placeholders.items():
                    formatted_prompt = formatted_prompt.replace(placeholder, value)
                
                sample.append(formatted_prompt)
                metadata.append({
                    "ex_id": row["ex_id"],
                    "dataset_id": row["dataset_id"],
                    "model": row["model"]
                })
                prev_prompts.append(current_text)
        else:
            if system_level_id not in level_map:
                raise ValueError("Invalid system_level_id. Must be one of: conversation, prompt, response, turn.")
            
            extractor = level_map[system_level_id]
            data_list = extractor(row)

            for data in data_list:
                placeholders = {"{text}": data}
                formatted_prompt = system_prompt_template
                for placeholder, value in placeholders.items():
                    formatted_prompt = formatted_prompt.replace(placeholder, value)
                sample.append(formatted_prompt)
                metadata.append({
                    "ex_id": row["ex_id"],
                    "dataset_id": row["dataset_id"],
                    "model": row["model"]
                })
    return sample, metadata


async def run_gpt(system_prompt_id, model_id, input_fpath, save_fpath, batch_size=1):
    try:
        system_level = system_prompt_id.split("_")[-1]
        system_prompt = "_".join(system_prompt_id.split("_")[:-1])
        system_prompt_template = io.read_json(constants.SYSTEM_PROMPTS_FPATH)[system_level][system_prompt]
    except KeyError:
        print("Please choose an existing level-prompt pair from data/system_prompts.json.")
        return
    
    existing_ex_ids = load_existing_ex_ids(save_fpath)    
    dataframe = pd.read_json(input_fpath, orient="records")
    formatted_prompts, metadata = extract_samples_and_metadata(dataframe, system_level, system_prompt, system_prompt_template, existing_ex_ids)
    print(f"Formatted prompts: {len(formatted_prompts)}, Metadata: {len(metadata)}")
    
    if not formatted_prompts:
        print("All examples already exist in the save file.")
        return

    gpt_instance = gpt.GPT(model=model_id, prompt=system_prompt_template)

    for batch, meta_batch in zip(batch_generator(formatted_prompts, batch_size), batch_generator(metadata, batch_size)):
        try:
            batch_responses = await gpt_instance.process_prompts_in_batches_async(batch)
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue
        
        batch_output = []
        for response, meta in zip(batch_responses, meta_batch):
            response_entry = {
                **meta,
                "model_id": model_id,
                "system_level_id": system_level,
                "system_prompt_id": system_prompt,
                "input": batch[0],
                "response": response
            }
            batch_output.append(response_entry)

            if batch_output:
                append_jsonl(batch_output, save_fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT on a sample.")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the sample json file."
    )
    parser.add_argument(
        "--system_prompt_id", 
        type=str, 
        required=True, 
        help="System prompt ID (e.g., self_disclosure_conversation)."
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"], 
        required=True, 
        help="GPT model to use."
    )
    parser.add_argument(
        "--save", 
        type=str, 
        required=True, 
        help="Save path."
    )
    
    args = parser.parse_args()
    asyncio.run(run_gpt(args.system_prompt_id, args.model_id, args.input, args.save))
