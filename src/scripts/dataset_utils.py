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
This dataset_utils.py file is used to define the Dataset and Conversation objects. 
A Dataset is used to define the collection of conversations or samples. 
To download a dataset, use the download_datasets.py file. 
To load a dataset, use the load_datasets.py file. 
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
    

class Dataset(): 
    """Dataset class used to define the common features / functions of any evaluation or usage dataset."""

    def __init__(self, dataset_id: str, data:List[Conversation]):
        self.dataset_id:str = dataset_id
        self.category:str = DATASET_CATEGORIES[dataset_id]
        self.data:List[Conversation] = data
        
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
        return dl         
