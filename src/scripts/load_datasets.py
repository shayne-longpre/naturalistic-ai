import sys
import os
import argparse
import numpy as np
import pandas as pd
import random
from datetime import datetime
from huggingface_hub import hf_hub_download

sys.path.append("./")
sys.path.append("src/")

from helpers import constants
from helpers import io


class Conversation(object):
    """A conversation object, with all metadata."""

    def __init__(
        self,
        ex_id,
        dataset_id,
        user_id,
        time,
        model,
        conversation,
        geography=None,
        languages=None,
    ):
        self.ex_id = ex_id
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.time = time
        self.model = model
        self.conversation = conversation
        self.geography = geography
        self.languages = languages

    def to_dict(self, unpack_conversation=False):
        obj = {
            "ex_id": self.ex_id,
            "dataset_id": self.dataset_id,
            "user_id": self.user_id,
            "time": self.time,
            "model": self.model,
            "geography": self.geography,
            "languages": self.languages,
        }
        if unpack_conversation:
            for i in range(6):
                obj[f"Turn {i}"] = self.conversation[i]["text"] if i < len(self.conversation) else ""
        else:
            obj["conversation"] = self.conversation
        return obj
    


def download_lmsys_1m():
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


DATASETS = {
    "wildchat_v1": download_wildchat_v1,
    "lmsys_1m": download_lmsys_1m,
    "sharegpt_v1": download_sharegpt_v1,
}

def main(dataset_id, sample, save_fpath):
    assert dataset_id in DATASETS, f"{dataset_id} not in {DATASETS.keys()}"
    dset_loader = DATASETS[dataset_id]
    dset = dset_loader()
    if sample:
        dset = random.sample(dset, int(sample))

    if save_fpath.endswith(".jsonl"):
        dset = [x.to_dict() for x in dset]
        io.write_jsonl(dset, save_fpath)
    elif save_fpath.endswith(".csv"):
        dset_df = pd.DataFrame([x.to_dict(unpack_conversation=True) for x in dset])
        dset_df.to_csv(save_fpath, index=False)
    else:
        raise ValueError("Don't recognize this save path extension.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        default=None,
        help=f"Dataset ID from {DATASETS.keys()}"
    )
    parser.add_argument(
        "--sample",
        required=False,
        default=False,
        help=f"An integer for how many to sample from the dataset."
    )
    parser.add_argument(
        "--save",
        required=True,
        default=None,
        help="Save filepath for dataset."
    )
    args = parser.parse_args()
    main(args.dataset, args.sample, args.save)