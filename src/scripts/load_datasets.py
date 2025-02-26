import pandas as pd 
import jsonlines 
import json
import argparse
import os
from typing import Literal
import numpy as np
import sys
sys.path.append("./")
sys.path.append("src/")
from src.scripts.dataset_utils import Dataset, Conversation

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

def read_data_from_files(dataset_id, path_to_dataset_downloads=""):
    """
    Read data from downloaded file. This is used by the Dataset object to get the data for a specific dataset (specified by the dataset_id). 
    If no path_to_dataset_downloads is provided, the function will look for a valid file at '../../dataset_downloads/{dataset_id}/dataset.{json,csv,jsonl}.

    Args:
        dataset_id (str): The id of the dataset to load.
        path_to_dataset_downloads (str, optional): The local path to the dataset. Defaults to None. If not provided, the dataset is loaded via HF API. 

    Returns:
        List[Conversation]: The loaded dataset as a list of Conversation objects.
    """
    conversations = []
  
    # Check all extensions at the provided folder, as well as the default download location.
    paths_to_check = [
        f"{path_to_dataset_downloads}/{dataset_id}/dataset.json",
        f"{path_to_dataset_downloads}/{dataset_id}/dataset.jsonl",
        f"{path_to_dataset_downloads}/{dataset_id}/dataset.csv",
        f"dataset_downloads/{dataset_id}/dataset.json",
        f"dataset_downloads/{dataset_id}/dataset.jsonl",
        f"dataset_downloads/{dataset_id}/dataset.csv"
    ]

    path_to_use = next((path for path in paths_to_check if os.path.exists(path)), None)

    if not path_to_use:
        raise ValueError(f"No valid file found for {dataset_id} in provided directory {path_to_dataset_downloads}/{dataset_id}, and no valid file found at the default download location of dataset_downloads/{dataset_id}.csv / .jsonl / .json")
    
     # If a valid path exists, load it as a list of conversation objects and return. 
    print(f"Reading File {path_to_use}", flush=True)
    if path_to_use.endswith('.json'):
        conversations = process_json_into_conversations(path_to_use)
    elif path_to_use.endswith('.jsonl'):
        conversations = process_jsonl_into_conversations(path_to_use)
    elif path_to_use.endswith('.csv'):
        conversations = process_csv_into_conversations(path_to_use)
    else:
        raise ValueError("Unsupported file format. Only .json, .jsonl, and .csv are supported.")
  
    return conversations


DATASET_LOCATIONS = {
    ### User Datasets ###
    "wildchat_v1": "dataset_downloads/wildchat_v1",
    "lmsys_1m": "dataset_downloads/lmsys_1m",
    "sharegpt_v1": "dataset_downloads/sharegpt_v1",

    ### Benchmarks ###
    "chatbot_arena": "dataset_downloads/chatbot_arena",
    "alpaca_eval": "dataset_downloads/alpaca_eval",
    "mmlu": "dataset_downloads/mmlu"
}
DATASET_CATEGORIES = {
    ### User Datasets ###
    "wildchat_v1": "conversation",
    "lmsys_1m": "conversation",
    "sharegpt_v1": "conversation",
    
    ### Benchmarks ###
    "chatbot_arena": "benchmark",
    "alpaca_eval": "benchmark",
    "mmlu": "benchmark"
}
DATASET_SIZES = {
    ### User Datasets ###
    "wildchat_v1": 1000000,
    "lmsys_1m": 1000000,
    "sharegpt_v1": 90665,
    
    ### Benchmarks ###
    "chatbot_arena": 66000,
    "alpaca_eval":  805,
    "mmlu": 14042
}

def filter_by_id(ids: list[str]):
    valid_ids = []
    for id in ids: 
        if id in DATASET_LOCATIONS.keys():
            valid_ids.append(id)
        else: 
            print(f"\n\n**** WARNING: {id} was not found as a valid dataset id. Available datasets include: {DATASET_LOCATIONS.keys()}")
    return valid_ids

def filter_by_categories(categories: list[str]):
    valid_ids = []
    for id, category in DATASET_CATEGORIES.items():
        if category in categories:
            valid_ids.append(id)
    if len(valid_ids) == 0: 
        print(f"\n\n**** WARNING: no datasets were found with the provided categories. Available categories include: {list(set(DATASET_LOCATIONS.values()))}")
    return valid_ids

def filter_by_size(size_range:list[int]):
    valid_ids = []
    for id, size in DATASET_SIZES.items(): 
        if size >= size_range[1] and size <= size_range[0]: 
            valid_ids.append(id)
    if len(valid_ids) == 0: 
        print(f"\n\n**** WARNING: no datasets were found within the provided size range of {size_range}. The sizes of available datasets are: {', '.join([f'\n{key}: {value}' for key, value in DATASET_SIZES.items()])}")
    
    return []

def load_datasets(by: Literal["id", "category", "size"], ids=[], categories=[], size_range =[], path_to_dataset_downloads =""):
    # Find the dataset ids that match the specifications
    if by == "id": 
        if len(ids) == 0: 
            raise ValueError(f"You've specified loading datasets by ID, but have not provided any dataset ids. Try something like: load_datasets(by='id', ids=['mmlu'])")
        matching_dataset_ids = filter_by_id(ids)

    elif by == "category": 
        if len(categories) == 0: 
            raise ValueError(f"You've specified loading datasets by categories, but have not provided any categories. Try something like: load_datasets(by='category', categories=['benchmark'])")
        matching_dataset_ids = filter_by_categories(categories)

    elif by == "size":
        if len(size_range) < 2: 
            raise ValueError(f"You've specified loading datasets by size, but have not provided a valid size_range. Try something like: load_datasets(by='size', size_range=[0,10000])")
        matching_dataset_ids = filter_by_size(size_range)

    # Confirm at least one match is found
    if len(matching_dataset_ids) == 0: 
        raise ValueError(f"No datasets meet the specifications given. The available datasets include {DATASET_LOCATIONS.keys()}")
   
    print(f"Found {len(matching_dataset_ids)} datasets that meet the desired specifications: {matching_dataset_ids}. Loading them from data files...")
   
    # Load the dataset objects and return 
    datasets = []
    for dataset_id in matching_dataset_ids: 
        dataset=dataset = read_data_from_files(dataset_id=dataset_id, path_to_dataset_downloads = path_to_dataset_downloads)
        datasets.append(dataset)
    
    return datasets

