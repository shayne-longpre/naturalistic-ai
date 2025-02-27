import random
from typing import List
from torch.utils.data import DataLoader
import pandas as pd 

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
        self.data:List[Conversation] = data
        if len(data) ==0: 
            raise ValueError("No Data Provided to the Dataset class.")
        self.features = list(data[0].to_dict().keys())
        
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
    
    def to_pandas(self):
        """
        This function converts the list of conversation objects to a pandas dataframe. It does *NOT* change the underlying self.data state. 
        """
        dicts = []
        for conv in self.data: 
            dicts.append(conv.to_dict())

        df = pd.DataFrame.from_dict(dicts)
        return df
    
    def get_feature(self, feature, as_pandas = False):
        """
        This function by default returns the data as 2 lists, one of the example ids and one of the requested feature. 
        If as_pandas is set to True, the function returns a pandas Dataframe with two columns, ex_id and the requested feature.
        """
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
            


        
    
        
