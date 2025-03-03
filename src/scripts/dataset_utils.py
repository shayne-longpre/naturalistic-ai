import random
from typing import List
from torch.utils.data import DataLoader
import pandas as pd 
import os 
import io
import jsonlines
import json
import warnings

"""
This dataset_utils.py file is used to define the Dataset and Conversation objects, and corresponding functions. 
A Dataset is used to define the collection of conversations or samples. 
To download a dataset, use the download_datasets.py file. 
To load a dataset for use, use the load_datasets.py file. 
"""

###################### Processing Functions ###################### 

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


###################### Conversation and Dataset Objects ###################### 

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
    """
    Dataset class used to define the common features / functions of any evaluation or usage dataset. 
    There are 2 ways to provide data to the Dataset Object: 
    - If you already have a list of dataset objects, you can pass it in the constructor: e.g. mmlu = Dataset(data = [conv1, conv2]) 
    - If you don't already have the data loaded, you can read it from file: e.g. mmlu=Dataset(id). mmlu.load_data_from_file(args)
    """

    def __init__(self, dataset_id: str, data:List[Conversation] = []):
        self.dataset_id:str = dataset_id
        self.data:List[Conversation] = data
        if len(data) ==0: 
            raise ValueError("No Data Provided to the Dataset class.")
        if len(self.data) > 0:
            self.features = list(data[0].to_dict().keys())
        else: 
            self.features = []
        
    def __len__(self):
        return len(self.data)

    ### Operations ###
    def sample(self, n):
        """Sample n conversations from the dataset."""
        if len(self.data)>0:
            return random.sample(self.data, n)
        else: 
            warnings.warn("Cannot sample data, as no data has been loaded. Load data by calling load_data_from_file()")
            return self.data

    def slice(self, start, end):
        """Get a slice of the dataset from start to end."""
        if len(self.data)>0:
            return self.data[start:end]
        else: 
            warnings.warn("Cannot slice data, as no data has been loaded. Load data by calling load_data_from_file()")
            return self.data


    def to_pandas(self):
        """
        This function converts the list of conversation objects to a pandas dataframe. It does *NOT* change the underlying self.data state. 
        """
        if len(self.data)>0:
            warnings.warn("Cannot convert to pandas, as no data has been loaded. Load data by calling load_data_from_file().")
            return 
        
        dicts = []
        for conv in self.data: 
            dicts.append(conv.to_dict())

        df = pd.DataFrame.from_dict(dicts)
        return df
    
    ### Getters ###
    def get_feature(self, feature, as_pandas = False):
        """
        This function by default returns the data as 2 lists, one of the example ids and one of the requested feature. 
        If as_pandas is set to True, the function returns a pandas Dataframe with two columns, ex_id and the requested feature.
        """
        if len(self.data)>0:
            warnings.warn("Cannot get feature, as no data has been loaded. Load data by calling load_data_from_file().")
            return 
        
        if feature in self.features: 
            ids = []
            features = []
            for conv in self.data: 
                conv_dict = conv.to_dict()
                ids.append(conv_dict["ex_id"])
                features.append(conv_dict[feature])
            
            if as_pandas: 
                return pd.DataFrame.from_dict({"ex_id": ids, f"{feature}": features})
            else: 
                return ids, features
        else: 
            print(f"Requested Feature not Found in this Dataset. Available Features Include: {self.features}")
        return 

    def get_metadata_only(self, as_pandas = False): 
        """
        This function by default returns the data as a list of conversation objects, just without the conversation field. 
        If as_pandas is set to True, the function returns a pandas Dataframe of the metadata.
        """
        if len(self.data)>0:
            warnings.warn("Cannot get metadata, as no data has been loaded. Load data by calling load_data_from_file().")
            return 
        return_conversations = []
        
        for conv in self.data: 
            conv_dict = conv.to_dict()
            conv_dict.pop("conversation", None)
            if as_pandas: 
                return_conversations.append(conv_dict)
            else:
                return_conversations.append(Conversation(**conv_dict))
        
        if as_pandas: 
            return pd.DataFrame.from_dict(return_conversations)
        else: 
            return return_conversations
    
    def get_dataloader(self, batch_size:int = 32, shuffle:bool = False ):
        if len(self.data)>0:
            warnings.warn("Cannot get dataloader, as no data has been loaded. Load data by calling load_data_from_file().")
            return 
        dl = DataLoader(self.data, batch_size=batch_size, shuffle=shuffle)
        return dl 
    
    ### File Operations ###
    def load_data_from_file(self, path_to_dataset_downloads = ""):
        """
        Read data from downloaded file. This is used to load the data for a specific dataset (specified by the dataset_id). 
        If no path_to_dataset_downloads is provided, the function will look for a valid file at '../../dataset_downloads/{dataset_id}/dataset.{json,csv,jsonl}.

        Args:
            path_to_dataset_downloads (str, optional): The local path to the dataset. Defaults to None. If not provided, the dataset is loaded via HF API. 

        Returns:
            List[Conversation]: The loaded dataset as a list of Conversation objects. It is saved 
        """
        conversations = []
  
        # Check all extensions at the provided folder, as well as the default download location.
        paths_to_check = [
            f"{path_to_dataset_downloads}/{self.dataset_id}/dataset.json",
            f"{path_to_dataset_downloads}/{self.dataset_id}/dataset.jsonl",
            f"{path_to_dataset_downloads}/{self.dataset_id}/dataset.csv",
            f"dataset_downloads/{self.dataset_id}/dataset.json",
            f"dataset_downloads/{self.dataset_id}/dataset.jsonl",
            f"dataset_downloads/{self.dataset_id}/dataset.csv"
        ]

        path_to_use = next((path for path in paths_to_check if os.path.exists(path)), None)

        if not path_to_use:
            raise ValueError(f"No valid file found for {self.dataset_id} in provided directory {path_to_dataset_downloads}/{self.dataset_id}, and no valid file found at the default download location of dataset_downloads/{self.dataset_id}.csv / .jsonl / .json")
        
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
    
        self.data = conversations
        print(f"Data loaded! Access using .data, .get_dataloader, or .to_pandas")
        return 

    def write_to_file(self, dataset_folder:str, save_path_overwrite: str, dataset_file_type:str = "jsonl"):
        """
        This function writes the loaded data to a file. 
        Args: 
            dataset_folder: General 'datasets' folder where you plan to store datasets in. Datasets are saved in {dataset_folder}/{dataset_name}/<actual data files> for consistency.
            save_path_overwrite: By default, Datasets are saved in {dataset_folder}/{dataset_name}/<actual data files> for consistency. To define a specific save path instead, provide the full path here
             
        """
        if len(self.data)>0:
            warnings.warn("Cannot write data to file, as no data has been loaded. Load data by calling load_data_from_file().")
            return 
        
        assert dataset_file_type in ["json", "jsonl", "csv"], f"{dataset_file_type} is not one of [json, jsonl, csv]."
        if save_path_overwrite: 
            save_path = save_path_overwrite
        else: 
            os.makedirs(f"{dataset_folder}", exist_ok=True)
            os.makedirs(f"{dataset_folder}/{self.dataset_id}", exist_ok=True)
            save_path = f"{dataset_folder}/{self.dataset_id}/dataset.{dataset_file_type}"
        
        if save_path.endswith(".jsonl"):
            dset = [x.to_dict() for x in self.data]
            io.write_jsonl(dset, save_path)
        elif save_path.endswith(".csv"):
            dset_df = pd.DataFrame([x.to_dict(unpack_conversation=True) for x in dset])
            dset_df.to_csv(save_path, index=False)
        else:
            raise ValueError(f"Don't recognize this save path extension for the constructed save_path: {save_path}")
        
        print(f"{self.dataset_id} dataset saved successfully to {save_path}")
        return save_path
            