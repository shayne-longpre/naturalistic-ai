import os
import sys
import gzip
import shlex
import subprocess
import yaml
import json
import jsonlines
import multiprocessing
import random
from ast import literal_eval
import pandas as pd
import typing
from collections import defaultdict
# from semanticscholar import SemanticScholar
import base64
from PIL import Image
from io import BytesIO
import numpy as np

import requests
from datasets import Dataset, load_dataset

# import src.helpers.constants as constants
# from . import constants

#############################################################################
############### Image Reading and Conversion
#############################################################################

def convert_base64_to_PIL_image(base64_string):
    
    # Remove the metadata part of the Base64 string
    base64_string = base64_string.split(",")[1]

    # Decode the Base64 string
    image_data = base64.b64decode(base64_string)

    # Make a PIL Image 
    image = Image.open(BytesIO(image_data))
    return image

def convert_base64_to_np_array(base64_string):
    
    # Remove the metadata part of the Base64 string
    base64_string = base64_string.split(",")[1]

    # Decode the Base64 string
    image_data = base64.b64decode(base64_string)

    # Convert the image data to a PIL Image
    image = Image.open(BytesIO(image_data))

    # Convert the PIL Image to a numpy array
    np_array = np.array(image)

    return np_array

#############################################################################
############### Local File IO
#############################################################################

def listdir_nohidden(path: str) -> typing.List[str]:
    """Returns all non-hidden files within a directory."""
    assert os.path.exists(path) and os.path.isdir(path)
    return [os.path.join(path, f) for f in os.listdir(path) if not f.startswith(".")]

def read_txt(path: str) ->  typing.List[typing.Any]:
    with open(path, "r", encoding="utf8") as f:
        return [l.strip() for l in f.readlines()]

def write_txt(path: str, data: str):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf8") as outf:
        outf.write(data)

def write_json(data, outpath, compress: bool=False):
    dirname = os.path.dirname(outpath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    if compress:
        with gzip.open(outpath, 'wt', encoding='UTF-8') as zipfile:
            zipfile.write(json.dumps(data, ensure_ascii=False, indent=4))
    else:
        with open(outpath, 'w', encoding='utf-8') as outf:
            json.dump(data, outf, ensure_ascii=False, indent=4)


def read_json(inpath: str, verbose=False):
    if verbose:
        print(f"Reading {inpath}...")
    if inpath[-2:] in ["gz", "gzip"]:
        with gzip.open(inpath, 'rb') as fp:
            return json.load(fp)

    with open(inpath, 'rt', encoding='UTF-8') as inf:
        return json.load(inf)

def write_jsonl(
    data: typing.Union[pd.DataFrame, typing.List[typing.Dict]],
    outpath: str,
    compress: bool=False,
    dumps=None,
):
    dirname = os.path.dirname(outpath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    if isinstance(data, list):
        if compress:
            with gzip.open(outpath, 'wb') as fp:
                json_writer = jsonlines.Writer(fp)#, dumps=dumps)
                json_writer.write_all(data)
        else:
            with open(outpath, "wb") as fp:
                json_writer = jsonlines.Writer(fp) #, dumps=dumps)
                json_writer.write_all(data)
    else: # Must be dataframe:
        data.to_json(outpath, orient="records", lines=True, compression="gzip" if compress else "infer")

def read_jsonl(inpath: str) -> typing.List[typing.Dict]:
    if inpath[-2:] in ["gz", "gzip"]:
        with gzip.open(inpath, 'rb') as fp:
            j_reader = jsonlines.Reader(fp)
            return [l for l in j_reader]
    else:
        with open(inpath, "rb") as fp:
            j_reader = jsonlines.Reader(fp)
            return [l for l in j_reader]

def read_yaml(inpath: str):
    with open(inpath, 'r') as inf:
        return yaml.safe_load(inf)



#############################################################################
############### HuggingFace Downloaders
#############################################################################


def huggingface_download(
    data_address, name=None, data_dir=None, data_files=None, split=None, sample=None,
):
    """Download a dataset from the Hugging Face Hub.

    It supports various options for specifying the dataset to download,
    such as providing a name, a data directory, data files, or a split.

    Args:
        data_address (str): The address or identifier of the dataset
        name (str, optional): Name of the dataset to download. Defaults to None.
        data_dir (str, optional): Path to the directory containing the dataset files. Defaults to None.
        data_files (str or list, optional): Path(s) to specific dataset files. Defaults to None.
        split (str, optional): Name of the split to take (usually "train"). Defaults to None.
        sample (None or int): How many rows to sample. Defaults to None, to take all data.

    Returns:
        list or Dataset: The downloaded dataset as a list of items,
            or Hugging Face Dataset object (if failed converted to list).
    """
    assert not (data_dir and data_files)

    # num_proc = max(multiprocessing.cpu_count() // 2, 1)
    if data_files:
        dset = load_dataset(data_address, data_files=data_files, use_auth_token=True)
    elif data_dir:
        dset = load_dataset(data_address, data_dir=data_dir, use_auth_token=True)
    elif name:
        dset = load_dataset(data_address, name, use_auth_token=True)
    else:
        dset = load_dataset(data_address) #, use_auth_token=True)

    if split:
        dset = dset[split]

    try:
        dset = dset.to_list()
    except:
        print("Trouble converting Hugging Face dataset to list...")
        pass

    if sample is not None:
        dset = random.sample(dset, sample)
    return dset
