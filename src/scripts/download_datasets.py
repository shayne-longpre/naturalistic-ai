import sys
import os
import argparse
import numpy as np
import pandas as pd
import random
from datetime import datetime
from huggingface_hub import hf_hub_download
from datasets import load_dataset

sys.path.append("./")
sys.path.append("src/")
from dataset_utils import Conversation

from helpers import constants
from helpers import io
import uuid

"""
This file is used to download and format datasets in a common format (list of Conversation objects). To use the datasets, use the 
"""

def download_lmsys_1m():
    print("Starting Download for lmsys-chat-1m...")
    # https://huggingface.co/datasets/lmsys/lmsys-chat-1m
    dset = io.huggingface_download('lmsys/lmsys-chat-1m', split='train')

    def process_data(datum):
        conversation = [
            {
                "role": msg.get("role"),
                "turn": idx,
                "text": msg.get("content", "")
            }
            for idx, msg in enumerate(datum.get("conversation", []))
        ]

        return Conversation(
            ex_id="lmsys1m_" + datum.get('conversation_id'),
            dataset_id="lmsys1m_",
            user_id=None,
            time=None,
            model=datum.get('model'),
            conversation=conversation,
            geography=None,
            languages=datum.get('language', None),
        )

    return [process_data(datum) for datum in dset]

# Download WildChat
def download_wildchat_v1():
    dset = io.huggingface_download("allenai/WildChat-1M", split="train")

    def process_data(datum):
        state = datum.get('state')
        country = f"{datum.get('country', 'Unknown')}"
        timestamp = datum.get('timestamp')
        
        conversation = [
            {
                "role": msg.get("role"),
                "turn": idx,
                "text": msg.get("content", "")
            }
            for idx, msg in enumerate(datum.get("conversation", []))
        ]

        return Conversation(
            ex_id="wildchat_" + datum.get('conversation_hash'),
            dataset_id="wildchat_1m",
            user_id=datum.get('hashed_ip'),
            time=timestamp.isoformat() if isinstance(timestamp, datetime) else None,
            model=datum.get('model'),
            conversation=conversation,
            geography=country if state is None else f"{country}; {state}",
            languages=None,
        )
    
    return [process_data(datum) for datum in dset]

# Download ShareGPT
def download_sharegpt_v1():
    # unfiltered: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
    sharegpt_dir = "anon8231489123/ShareGPT_Vicuna_unfiltered"
    sv_dset_p1 = hf_hub_download(
        repo_id=sharegpt_dir,
        filename="sg_90k_part1.json",
        subfolder="HTML_cleaned_raw_dataset",
        repo_type="dataset",
    )
    sv_dset_p2 = hf_hub_download(
        repo_id=sharegpt_dir,
        filename="sg_90k_part2.json",
        subfolder="HTML_cleaned_raw_dataset",
        repo_type="dataset",
    )
    full_dset = pd.concat([pd.read_json(sv_dset_p1), pd.read_json(sv_dset_p2)]).to_dict(
        "records"
    )

    sharegpt_systems = ["system", "human", "user", "gpt", "chatgpt", "bing", "bard", "assistant"]
    def process_data(datum):
        conversation = []
        
        for idx, msg in enumerate(datum.get("conversations", [])):
            assert msg.get("from", "") in sharegpt_systems, "Error: " + msg["from"]
            conversation.append({
                "role": msg.get("from"),
                "turn": idx,
                "text": msg.get("value", "")
            })

        return Conversation(
            ex_id="sharegpt_" + datum.get('id'),
            dataset_id="sharegpt",
            user_id=None,
            time=None,  # TODO: fill in rough period
            model=None,  # TODO: fill in OpenAI models at that time
            conversation=conversation,
            geography=None,
            languages=None,
        )

    return [process_data(datum) for datum in full_dset] 


# Download ChatBotArena 
def download_chatbot_arena():
    dset = io.huggingface_download("lmsys/chatbot_arena_conversations", split="train")

    from datetime import datetime

    def add_turn_and_rename_keys(conv:list[object]):
        turn_count = 0
        conv_with_turn = []
        for statement in conv:
            statement["turn"] = turn_count
            statement["text"] = statement.pop("content")
            if statement["role"] == "assistant":
                turn_count = turn_count+1
            conv_with_turn.append(statement)
        return conv_with_turn

    def process_data(datum):
        state = datum.get('state')
        country = f"{datum.get('country', 'Unknown')}"
        timestamp = datum.get('tstamp')

        # Chatbot Arena contains pairs of conversations
        conv_a = datum.get("conversation_a")
        conv_b = datum.get("conversation_b")

        conv_a_reformatted = add_turn_and_rename_keys(conv_a)
        conv_b_reformatted = add_turn_and_rename_keys(conv_b)
        
        conversation_a = Conversation(
            ex_id="chatbot_arena_" + datum.get('question_id') + "_a",
            dataset_id="chatbot_arena",
            user_id=datum.get('judge'),
            time=timestamp.isoformat() if isinstance(timestamp, datetime) else None,
            model=datum.get('model_a'),
            conversation=conv_a_reformatted,
            geography=country if state is None else f"{country}; {state}",
            languages=datum.get("language")
        )
        
        conversation_b = Conversation(
            ex_id="chatbot_arena_" + datum.get('question_id') + "_b",
            dataset_id="chatbot_arena",
            user_id=datum.get('judge'),
            time=timestamp.isoformat() if isinstance(timestamp, datetime) else None,
            model=datum.get('model_b'),
            conversation=conv_b_reformatted,
            geography=country if state is None else f"{country}; {state}",
            languages=datum.get("language")
        )

        return conversation_a, conversation_b 
    
    conversations_to_return = []
    
    for datum in dset: 
        conv_a, conv_b = process_data(datum)
        conversations_to_return.append(conv_a)
        conversations_to_return.append(conv_b)
    
    return conversations_to_return


def download_alpaca_eval():
    dset = load_dataset("tatsu-lab/alpaca_eval", split = "eval", trust_remote_code=True, token = True) #TODO integrate this with io helpers

    def process_data(datum):
        conv = [{
                "role": "user",
                "turn": 0, 
                "text": datum.get("instruction")
                }, {
                "role": "assistant",
                "turn": 0, 
                "text": datum.get("output")
                }]

        
        return Conversation(
            ex_id="alpaca_eval_" + str(uuid.uuid4()),
            dataset_id="alpaca_eval",
            user_id=str(uuid.uuid4()),
            time=None,
            model=datum.get('generator'),
            conversation=conv,
            geography="Unknown",
            languages="English"
        )
        

    return [process_data(datum) for datum in dset] 



DOWNLOAD_FUNCTIONS = {
    "wildchat_v1": download_wildchat_v1,
    "lmsys_1m": download_lmsys_1m,
    "sharegpt_v1": download_sharegpt_v1,
    "chatbot_arena": download_chatbot_arena,
    "alpaca_eval": download_alpaca_eval
}


def main(dataset_id:str, sample: int, dataset_folder:str, save_path_overwrite: str, dataset_file_type:str = "jsonl"):
    assert dataset_id in DOWNLOAD_FUNCTIONS, f"{dataset_id} not in {DOWNLOAD_FUNCTIONS.keys()}"
    assert dataset_file_type in ["json", "jsonl", "csv"], f"{dataset_file_type} is not one of [json, jsonl, csv]."

    dset_loader = DOWNLOAD_FUNCTIONS[dataset_id]
    dset = dset_loader()

    if sample:
        dset = random.sample(dset, int(sample))

    if save_path_overwrite: 
        save_path = save_path_overwrite
    else: 
        os.makedirs(f"{dataset_folder}", exist_ok=True)
        os.makedirs(f"{dataset_folder}/{dataset_id}", exist_ok=True)
        save_path = f"{dataset_folder}/{dataset_id}/dataset.{dataset_file_type}"
    
    if save_path.endswith(".jsonl"):
        dset = [x.to_dict() for x in dset]
        io.write_jsonl(dset, save_path)
    elif save_path.endswith(".csv"):
        dset_df = pd.DataFrame([x.to_dict(unpack_conversation=True) for x in dset])
        dset_df.to_csv(save_path, index=False)
    else:
        raise ValueError(f"Don't recognize this save path extension for the constructed save_path: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_id",
        required=True,
        default=None,
        help=f"Dataset ID from {DOWNLOAD_FUNCTIONS.keys()}"
    )
    parser.add_argument(
        "--sample",
        required=False,
        default=False,
        help=f"An integer for how many to sample from the dataset."
    )
    parser.add_argument(
        "--dataset_folder",
        required=True,
        default="../../datasets",
        help="General 'datasets' folder where you plan to store datasets in. Datasets are saved in {dataset_folder}/{dataset_name}/<actual data files> for consistency."
    )
    parser.add_argument(
        "--save_path_overwrite",
        required=False,
        default="",
        help="By default, Datasets are saved in {dataset_folder}/{dataset_name}/<actual data files> for consistency. To define a specific save path instead, provide the full path here."
    )
    args = parser.parse_args()
    main(args.dataset_id, args.sample, args.dataset_folder, args.save_path_overwrite)