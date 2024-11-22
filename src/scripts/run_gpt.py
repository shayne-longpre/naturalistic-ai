import sys
import asyncio
import json
import re
import pickle
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
        {"input": prompt, "response": response}
        for prompt, response in zip(batch, responses)
    ]
    
async def run_gpt(
    system_prompt_id,
    input_fpath,
    save_fpath,
    batch_size=20,
):
    system_prompt = io.read_json(constants.SYSTEM_PROMPTS_FPATH)[system_prompt_id]
    sample = io.read_json(input_fpath)
    gpt_instance = gpt.GPT(model="gpt-4o", prompt=system_prompt)

    all_responses = []
    for batch in batch_generator(sample, batch_size):
        batch_responses = await process_batch(gpt_instance, batch)
        all_responses.extend(batch_responses)

    io.write_jsonl(all_responses, save_fpath)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT on a sample.")
    parser.add_argument(
        "--input", type=str, default="", help="Path to the sample file."
    )
    parser.add_argument(
        "--system_prompt_id",
        type=str,
        default=None,
        help='Choose from system prompt keys in data/system_prompts.json',
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help='Save path.',
    )


    args = parser.parse_args()

    asyncio.run(run_gpt(
        args.system_prompt_id,
        args.input,
        args.save,
    ))
