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
            "response": response.replace("```", "").replace("json", "").replace("\n", "")
        }
        for prompt, response in zip(batch, responses)
    ]

def extract_samples_and_metadata(dataframe, system_level_id):
    sample, metadata = [], []
    level_map = {
        "conversation": lambda row: " ".join(
            row[f"Turn {i}"] for i in range(6) if pd.notna(row[f"Turn {i}"])
        ),
        "prompt": lambda row: [row[f"Turn {i}"] for i in [0, 2, 4] if pd.notna(row[f"Turn {i}"])],
        "response": lambda row: [row[f"Turn {i}"] for i in [1, 3, 5] if pd.notna(row[f"Turn {i}"])],
        "turn": lambda row: [
            f"{row[f'Turn {i}']} {row[f'Turn {i+1}']}" for i in range(5)
            if pd.notna(row[f"Turn {i}"]) and pd.notna(row[f"Turn {i+1}"])
        ]
    }

    if system_level_id not in level_map:
        raise ValueError("Invalid system_level_id. Must be one of: conversation, prompt, response, turn.")

    extractor = level_map[system_level_id]
    for _, row in dataframe.iterrows():
        data = extractor(row)
        if isinstance(data, list):
            for item in data:
                sample.append(item)
                metadata.append({
                    "ex_id": row["ex_id"],
                    "dataset_id": row["dataset_id"],
                    "model": row["model"]
                })
        elif data:
            sample.append(data)
            metadata.append({
                "ex_id": row["ex_id"],
                "dataset_id": row["dataset_id"],
                "model": row["model"]
            })
    return sample, metadata

async def run_gpt(
    system_level_id,
    system_prompt_id,
    input_fpath,
    save_fpath,
    batch_size=20,
):
    try:
        system_prompt = io.read_json(constants.SYSTEM_PROMPTS_FPATH)[system_level_id][system_prompt_id]
        # system_prompt = io.read_json(constants.DETAILED_SYSTEM_PROMPTS_FPATH)[system_level_id][system_prompt_id]
    except KeyError:
        print("Please choose an existing level-prompt pair from data/system_prompts.json.")
        return
    
    dataframe = pd.read_csv(input_fpath)
    sample, metadata = extract_samples_and_metadata(dataframe, system_level_id)

    gpt_instance = gpt.GPT(model="gpt-4o", prompt=system_prompt)
    all_responses = []
    
    for batch, meta_batch in zip(batch_generator(sample, batch_size), batch_generator(metadata, batch_size)):
        formatted_prompts = [
            system_prompt.replace("{text}", item) for item in batch
        ]
        batch_responses = await process_batch(gpt_instance, formatted_prompts)
        for response, meta in zip(batch_responses, meta_batch):
            all_responses.append({
                **meta,
                "system_level_id": system_level_id,
                "system_prompt_id": system_prompt_id,
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
        help="Path to the sample csv file."
    )
    parser.add_argument(
        "--system_level_id",
        type=str,
        default=None,
        help='Choose a level from: conversation, prompt, response, turn.',
    )
    parser.add_argument(
        "--system_prompt_id",
        type=str,
        default=None,
        help='Choose from system prompt keys in data/system_prompts.json.',
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help='Save path.',
    )


    args = parser.parse_args()

    asyncio.run(run_gpt(
        args.system_level_id,
        args.system_prompt_id,
        args.input,
        args.save,
    ))
