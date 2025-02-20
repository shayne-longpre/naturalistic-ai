import random
from typing import List
from torch.utils.data import DataLoader
import os
import json
from datasets import load_dataset
import pandas as pd
import json
import jsonlines
"""
This dataset_utils.py file is used to define the Dataset and Conversation objects. A Dataset is used to define the collection of conversations or samples. 
"""

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
    

DATASET_CATEGORIES = {
    ### User Datasets ###
    "wildchat_v1": "usage",
    "lmsys_1m": "usage",
    "sharegpt_v1": "usage",
    ### Benchmarks ###
    "mmlu": "benchmark",
}

DATASET_TYPES = {
    ### User Datasets ###
    "wildchat_v1": "conversation",
    "lmsys_1m": "conversation",
    "sharegpt_v1": "conversation",
    ### Benchmarks ###
    "mmlu": "qa",
}

def process_json_into_conversations(path):
    conversations = []
    with open(path, 'r') as file:
        data = json.load(file)
        for item in data:
            conversation = [
                {"text": turn["text"]} for turn in item["conversation"]
            ]
            
            conv_obj = Conversation(
                ex_id=item["ex_id"],
                dataset_id=item["dataset_id"],
                user_id=item["user_id"],
                time=item["time"],
                model=item["model"],
                conversation=conversation,
                geography=item["geography"],
                languages=item["languages"]
            )

            conversations.append(conv_obj)
    return conversations

def process_csv_into_conversations(path):
    conversations = []
    data = pd.read_csv(path)
    for _, row in data.iterrows():
        conv_unpacked = [
            {"text": row[f"Turn {i}"]} for i in range(6) if pd.notna(row[f"Turn {i}"])
        ]
        conv_obj = Conversation(
            ex_id=row["ex_id"],
            dataset_id=row["dataset_id"],
            user_id=row["user_id"],
            time=row["time"],
            model=row["model"],
            conversation=conv_unpacked,
            geography=row["geography"],
            languages=row["languages"]
        )
        conversations.append(conv_obj)
        
    return conversations

def process_jsonl_into_conversations(path):
    conversations = []
  
    with jsonlines.open(path) as reader:
        for item in reader:
            conversation = [
                {"text": turn["text"]} for turn in item["conversation"]
            ]
            
            conv_obj = Conversation(
                ex_id=item["ex_id"],
                dataset_id=item["dataset_id"],
                user_id=item["user_id"],
                time=item["time"],
                model=item["model"],
                conversation=conversation,
                geography=item["geography"],
                languages=item["languages"]
            )

            conversations.append(conv_obj)
    
    return conversations

def read_data_from_files(dataset_id, local_path=None):
    """
    Read data from downloaded file. This is used by the Dataset object to get the data for a specific dataset (specified by the dataset_id). 
    If no local_path is provided, the function will look for a valid file at '../../dataset_downloads/{dataset_id}/dataset.{json,csv,jsonl}.

    Args:
        dataset_id (str): The id of the dataset to load.
        local_path (str, optional): The local path to the dataset. Defaults to None. If not provided, the dataset is loaded via HF API. 

    Returns:
        List[Conversation]: The loaded dataset as a list of Conversation objects.
    """
    conversations = []
    # print(local_path)
    # print(os.path.exists(local_path))
  
    default_paths_to_check = [f"../../dataset_downloads/{dataset_id}/dataset.json", f"../../dataset_downloads/{dataset_id}/dataset.csv"]
    
    # Determine if the provided path is valid, or if the dataset exists at the default save location. 
    path_to_use = ""
    if local_path and os.path.exists(local_path):
        path_to_use = local_path
    elif os.path.exists(default_paths_to_check[0]): 
        path_to_use = default_paths_to_check[0]
    elif os.path.exists(default_paths_to_check[1]): 
        path_to_use = default_paths_to_check[1]
    else: 
        raise ValueError(f"No valid path for {dataset_id} provided, and no valid file found at the default download location of ../../dataset_downloads/{dataset_id}.csv or ../../dataset_downloads/{dataset_id}.json")
    
     # If a valid path exists, load it as a list of conversation objects and return. 
    print(f"Reading File {path_to_use}", flush=True)
    if path_to_use.endswith('.json'):
        conversations = process_json_into_conversations(path_to_use)
    
    elif path_to_use.endswith('.jsonl'):
        conversations = process_jsonl_into_conversations(path_to_use)
    elif path_to_use.endswith('.csv'):
        conversations = process_csv_into_conversations(path_to_use)
    else:
        raise ValueError("Unsupported file format. Only .json and .csv are supported.")
  
    return conversations


class Dataset(): 
    """Dataset class used to define the common features / functions of any evaluation or usage dataset."""

    def __init__(self, dataset_id: str, local_path: str = None):
        assert dataset_id in DATASET_CATEGORIES, f"{dataset_id} not in {DATASET_CATEGORIES.keys()}. Please add it to the DATASET_TYPES Dictionary."
        self.dataset_id:str = dataset_id
        self.category:str = DATASET_CATEGORIES[dataset_id]
        self.data:List[Conversation] = read_data_from_files(dataset_id=dataset_id, local_path = local_path)
        
    def __len__(self):
        return len(self.data)

    def sample(self, n):
        """Sample n conversations from the dataset."""
        return random.sample(self.data, n)

    def slice(self, start, end):
        """Get a slice of the dataset from start to end."""
        return self.data[start:end]

    def get_dataloader(self, batch_size:int = 32, shuffle:bool = False ):
        dl = DataLoader(self.data, batch_size=batch_size, shuffle=shuffle)        
