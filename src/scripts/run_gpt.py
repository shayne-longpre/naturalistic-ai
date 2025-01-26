import sys
import asyncio
import json
import re
import pickle
import pandas as pd
from tqdm import tqdm
import csv
import argparse

sys.path.append("./")
sys.path.append("src/")

from helpers import constants
from helpers import io
from helpers import gpt


def batch_generator(data_list, batch_size):
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]

async def process_batch(gpt_instance, batch):
    """Processes a single batch of prompts asynchronously."""
    responses = await gpt_instance.process_prompts_in_batches_async(batch)
    return [
        {
            "input": prompt, 
            "response": response
        }
        for prompt, response in zip(batch, responses)
    ]

def valid_turn(text):
    return pd.notna(text) and text.strip() != ""

def extract_samples_and_metadata(dataframe, system_level_id, system_prompt_template):
    sample, metadata = [], []

    turn_columns = [col for col in dataframe.columns if col.startswith("Turn ")]
    max_turns = len(turn_columns)

    level_map = {
        "conversation": lambda row: [" ".join(row[f"Turn {i}"] for i in range(max_turns) if valid_turn(row.get(f"Turn {i}", "")))],
        "prompt": lambda row: [row[f"Turn {i}"] for i in range(0, max_turns, 2) if valid_turn(row.get(f"Turn {i}", ""))],
        "response": lambda row: [row[f"Turn {i}"] for i in range(1, max_turns, 2) if valid_turn(row.get(f"Turn {i}", ""))],
        "turn": lambda row: [
            f"{row[f'Turn {i}']} {row[f'Turn {i+1}']}" for i in range(max_turns - 1)
            if valid_turn(row.get(f"Turn {i}", "")) and valid_turn(row.get(f"Turn {i+1}", ""))
        ]
    }

    if system_level_id not in level_map:
        raise ValueError("Invalid system_level_id. Must be one of: conversation, prompt, response, turn.")

    extractor = level_map[system_level_id]

    for _, row in dataframe.iterrows():
        data_list = extractor(row)
        for data in data_list:
            # Add more placeholders if needed (e.g., "{outputformat}":"JSON")
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

async def run_gpt(
    system_prompt_id,
    model_id,
    input_fpath,
    save_fpath,
    batch_size=20,
):
    try:
        system_level = system_prompt_id.split("_")[-1]
        system_prompt = "_".join(system_prompt_id.split("_")[:-1])
        system_prompt_template = io.read_json(constants.SYSTEM_PROMPTS_FPATH)[system_level][system_prompt]
        # system_prompt = io.read_json(constants.DETAILED_SYSTEM_PROMPTS_FPATH)[system_level][system_prompt]
    except KeyError:
        print("Please choose an existing level-prompt pair from data/system_prompts.json.")
        return
    
    dataframe = pd.read_json(input_fpath, orient="records")
    formatted_prompts, metadata = extract_samples_and_metadata(dataframe, system_level, system_prompt_template)

    gpt_instance = gpt.GPT(model=model_id, prompt=system_prompt_template)
    print(gpt_instance)
    all_responses = []
    
    for batch, meta_batch in zip(batch_generator(formatted_prompts, batch_size), batch_generator(metadata, batch_size)):
        print(f"Processing batch: {batch}")
        batch_responses = await process_batch(gpt_instance, batch)
        print(f"Batch Responses: {batch_responses}")
        for response, meta in zip(batch_responses, meta_batch):
            print(f"Response: {response}, Metadata: {meta}")
            all_responses.append({
                **meta,
                "system_level_id": system_level,
                "system_prompt_id": system_prompt,
                "input": response["input"],
                "response": response["response"]
            })
    io.write_jsonl(all_responses, save_fpath)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT on a sample.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the sample json file."
    )
    parser.add_argument(
        "--system_prompt_id",
        type=str,
        default=None,
        help='Choose from system prompt keys in data/system_prompts.json and a level (conversation, prompt, response, turn) (i.e. self_disclosure_conversation).',
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        help='Specify the GPT model to use from the following options: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo.',
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help='Save path.',
    )


    args = parser.parse_args()
    print(args.model_id)

    asyncio.run(run_gpt(
        args.system_prompt_id,
        args.model_id,
        args.input,
        args.save,
    ))
