import sys
import os
import argparse
import numpy as np
import pandas as pd
import random
from datetime import datetime
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from datasets import load_dataset

sys.path.append("./")
sys.path.append("src/")
from dataset_utils import Conversation, Dataset

from helpers import io
import uuid
import requests
import json
import shutil

from bs4 import BeautifulSoup

"""
This file is used to download and format datasets in a common format (list of Conversation objects). 
To download a script, run something like: "python src/scripts/download_datasets.py --dataset_id=mmlu". 
See arg description for specifics. 
"""

def download_lmsys_1m():
    print("\nDownloading lmsys-chat-1m...")
    # https://huggingface.co/datasets/lmsys/lmsys-chat-1m
    dset = io.huggingface_download('lmsys/lmsys-chat-1m', split='train')

    def process_data(datum):
        conversation = [
            {
                "role": msg.get("role"),
                "turn": idx +1, # start at 1 for consistency
                "text": msg.get("content", ""),
                "image": ''
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

    print("Processing lmsys-chat-1m into conversation format...")
    return [process_data(datum) for datum in tqdm(dset, desc="Processing lymsys-chat-1m")]

# Download WildChat
def download_wildchat_v1():
    print("Starting Download for WildChat-1M...")
    dset = io.huggingface_download("allenai/WildChat-1M", split="train")

    def process_data(datum):
        state = datum.get('state')
        country = f"{datum.get('country', 'Unknown')}"
        timestamp = datum.get('timestamp')
        
        conversation = [
            {
                "role": msg.get("role"),
                "turn": idx +1, # start turn count at 1
                "text": msg.get("content", ""),
                "image": ''
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
    
    return [process_data(datum) for datum in tqdm(dset, desc="Processing WildChat")]

# Download ShareGPT
def download_sharegpt_v1():
    print("Starting Download for ShareGPT...")
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
                "turn": idx + 1, # start turn count at 1
                "text": msg.get("value", ""), 
                "image": msg.get("image", "") if msg.get("image") else ''
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

    return [process_data(datum) for datum in tqdm(full_dset, desc="Processing ShareGPT")] 

# Download ChatBotArena 
def download_chatbot_arena():
    print("Starting Download for ChatBotArena...")
    dset = io.huggingface_download("lmsys/chatbot_arena_conversations", split="train")

    def add_turn_and_rename_keys(conv:list[object]):
        turn_count = 1
        conv_with_turn = []
        for statement in conv:
            statement["turn"] = turn_count
            statement["text"] = statement.pop("content")
            statement["image"] = ''
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
    
    for datum in tqdm(dset, desc="Processing ChatBotArena"): 
        conv_a, conv_b = process_data(datum)
        conversations_to_return.append(conv_a)
        conversations_to_return.append(conv_b)
    
    return conversations_to_return

# Download Alpaca Eval
def download_alpaca_eval():
    print("Starting Download for AlpacaEval..")
    dset = load_dataset("tatsu-lab/alpaca_eval", split = "eval", trust_remote_code=True, token = True) #TODO integrate this with io helpers

    def process_data(datum):
        conv = [{
                "role": "user",
                "turn": 1, 
                "text": datum.get("instruction"),
                "image": ''
                }, {
                "role": "assistant",
                "turn": 1, 
                "text": datum.get("output"),
                "image": ''
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
        

    return [process_data(datum) for datum in tqdm(dset, desc="Processing AlpacaEval")]

# Download MMLU
def download_mmlu():
    print("Starting Download for MMLU...")
    # ['question', 'choices', 'answer'],
    categories = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

    choice_indiciators = ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)", "i)", "j)", "k)", "l)", "m)", "n)", "o)", "p)", "q)", "r)", "s)", "t)", "u)", "v)", "w)", "x)", "y)", "z)"]

    def process_data(datum):
            conv = [{
                    "role": "user",
                    "turn": 1, 
                    "text": datum.get("question") + " " + " ".join(f"{choice_indiciators[i]} {datum.get("choices")[i]}" for i in range(len(datum.get("choices")))),
                    "image": ''
                    }
                    ]
            print(conv)

            return Conversation(
                ex_id="mmlu_" + str(uuid.uuid4()),
                dataset_id="mmlu",
                user_id=str(uuid.uuid4()),
                time=None,
                model=None,
                conversation=conv,
                geography="Unknown",
                languages="English"
            )
    
    conversations_to_return = []
    for category in tqdm(categories, desc="Processing MMLU Categories"):
        dset = load_dataset("tasksource/mmlu", category, token=True)["test"]
        for datum in dset:
            conversations_to_return.append(process_data(datum))
    
    return conversations_to_return

# Download HLE
def download_hle():
    print("Starting Download for HLE (Humanity's Last Exam)...")
    dset = io.huggingface_download('cais/hle', split='test')

    def process_data(datum):
        conv = [{
            "role": "user",
            "turn": 1, 
            "text": datum.get("question"),
            "image": datum.get("image") if datum.get("image") else ''
            }]
        
        return Conversation(
            ex_id="hle_" + datum.get('id'),
            dataset_id="hle",
            user_id=str(datum.get('author_name')),
            time="02/11/2025", # huggingface release date
            model=None,
            conversation=conv,
            geography="Unknown",
            languages="English"
        )
    
    
    return [process_data(datum) for datum in tqdm(dset, desc="Processing Humanity's Last Exam")]

# Download GPQA
def download_gpqa():
    print("Starting Download for GPQA...")
    dset = io.huggingface_download('Idavidrein/gpqa', 'gpqa_extended', split="train")

    def process_data(datum):
        choice_strings = ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
        random.shuffle(choice_strings)

        choices = [
            f"a) {datum.get(choice_strings[0])}",
            f"b) {datum.get(choice_strings[1])}",
            f"c) {datum.get(choice_strings[2])}",
            f"d) {datum.get(choice_strings[3])}"
        ]
        
        conv = [{
                "role": "user",
                "turn": 1, 
                "text": f"{datum.get('Question')}\nChoices:\n" + "\n".join(choices),
                "image": ''
                }]
        

        return Conversation(
            ex_id="gpqa_" + datum.get('Record ID'),
            dataset_id="gpqa",
            user_id=str(datum.get('Question Writer')),
            time="11/29/2023",
            model=None,
            conversation=conv,
            geography="Unknown",
            languages="English"
        )
    
    return [process_data(datum) for datum in tqdm(dset, desc="Processing GPQA")]


# Download SWE-Bench
def download_swebench():
    print("Starting Download for SWE Bench...")
    dset = io.huggingface_download('princeton-nlp/SWE-bench', split="test")

    def process_data(datum):
        
        conv = [{
                "role": "user",
                "turn": 1, 
                "text": f"Problem Statement: {datum.get('problem_statement')} \nRepo: {datum.get('repo')} \nBase_commit: {datum.get('base_commit')} \n",
                "image": ''
                }]
        

        return Conversation(
            ex_id="swebench_" + datum.get('instance_id'),
            dataset_id="swebench",
            user_id=str(datum.get('repo')), # Is the repo a good indicator of user_id?
            time=datum.get('created_at').isoformat() if isinstance(datum.get('timestamp'), datetime) else None,
            model=None,
            conversation=conv,
            geography="Unknown",
            languages="Unknown"
        )
    
    return [process_data(datum) for datum in tqdm(dset, desc="Processing SWE Bench")]

# Download Math500
def download_math500():
    print("Starting Download for Math500...")
    dset = io.huggingface_download('ankner/math-500', split="test")

    def process_data(datum):
        
        conv = [{
                "role": "user",
                "turn": 1, 
                "text": f"{datum.get('problem')}", 
                "image": ''
                }]
        
        id = str(uuid.uuid4())
        return Conversation(
            ex_id="math500_" + id,
            dataset_id="math500",
            user_id=id,
            time="12/20/2024",
            model=None,
            conversation=conv,
            geography="Unknown",
            languages="English"
        )
    
    return [process_data(datum) for datum in tqdm(dset, desc="Processing Math500")]

def download_arc_agi_v2(): 
    # Because the data is hosted on github, we had to make a custom download function. Because this was completely specific to this particular dataset, I didn't want to add it as a general helper. 
    def download_arc_agi_from_github():
        # URL of the GitHub directory
        github_url = "https://github.com/arcprize/ARC-AGI-2/tree/main/data/evaluation"

        # Folder to save the downloaded JSON files
        output_folder = "downloaded_json_files"
        os.makedirs(output_folder, exist_ok=True)

        # Get the HTML content of the GitHub directory
        response = requests.get(github_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all JSON file links
        file_links = soup.find_all('a', href=True)
        json_links = [
            f"https://raw.githubusercontent.com{link['href'].replace('/blob', '')}"
            for link in file_links if link['href'].endswith('.json')
        ]

        # Download each JSON file
        
        for json_url in tqdm(json_links, desc="Downloading JSON files"):
            file_name = os.path.basename(json_url)
            file_path = os.path.join(output_folder, file_name)
            response = requests.get(json_url)
            with open(file_path, 'wb') as file:
                file.write(response.content)

        # Folder containing the downloaded JSON files
        input_folder = "downloaded_json_files"
        output_file = "combined_data.jsonl"

        # Open the output JSONL file for writing
        with open(output_file, "w") as jsonl_file:
            # Iterate through all JSON files in the folder
            for filename in os.listdir(input_folder):
                if filename.endswith(".json"):
                    file_path = os.path.join(input_folder, filename)
                    # Read the JSON file
                    with open(file_path, "r") as json_file:
                        data = json.load(json_file)
                        # Write each JSON object as a line in the JSONL file
                        if isinstance(data, list):
                            for item in data:
                                jsonl_file.write(json.dumps(item) + "\n")
                        else:
                            jsonl_file.write(json.dumps(data) + "\n")

        print(f"All downloaded JSON files have been combined into {output_file} and now are being deleted.")

        # Path to the folder to be deleted
        folder_path = "downloaded_json_files"

        # Delete the folder and its contents
        shutil.rmtree(folder_path, ignore_errors=True)

        print(f"Folder '{folder_path}' has been deleted.")
        
    def process_data(datum): 
        """Process the data from the JSONL file and return a Conversation object."""
        conv = [{
                "role": "user",
                "turn": 1, 
                "text": f"Demonstration: {datum.get('train')} \nTest: {datum.get('test')}", 
                "image": ''
                }]
        
        id = str(uuid.uuid4())
        return Conversation(
            ex_id="arc_agi_v2_" + id,
            dataset_id="arc_agi_v2",
            user_id=id,
            time="03/20/2025",
            model=None,
            conversation=conv,
            geography="Unknown",
            languages="Unknown"
        )

    # First, download the JSON files from the github repo and combine them into 1 JSONL file
    download_arc_agi_from_github()

    # Now, process the combined JSONL file as expected into a list of Conversation objects
    
    # Path to the combined JSONL file
    file_path = "combined_data.jsonl"

    # Initialize an empty list to store the modified objects
    modified_objects = []

    # Read the JSONL file and apply the function
    with open(file_path, "r") as jsonl_file:
        for line in jsonl_file:
            # Parse the JSON object
            data = json.loads(line.strip())
            # Apply the modification function
            modified_data = process_data(data)
            # Append the modified object to the list
            modified_objects.append(modified_data)

    # Delete the combined JSONL file
    os.remove(file_path)
    print("Combined_data.jsonl file has been deleted now that the processed version has saved in the correct location.")
    return modified_objects

    

DOWNLOAD_FUNCTIONS = {
    "wildchat_v1": download_wildchat_v1,
    "lmsys_1m": download_lmsys_1m,
    "sharegpt_v1": download_sharegpt_v1,
    "chatbot_arena": download_chatbot_arena,
    "alpaca_eval": download_alpaca_eval,
    "mmlu": download_mmlu,
    "hle": download_hle, 
    "gpqa": download_gpqa, 
    "swebench": download_swebench, 
    "math500": download_math500, 
    "arc_agi_v2": download_arc_agi_v2,
}


def main(dataset_id:str, sample: int, dataset_folder:str, save_path_overwrite: str, dataset_file_type:str = "jsonl"):
    # Check args 
    assert dataset_id in DOWNLOAD_FUNCTIONS, f"{dataset_id} not in {DOWNLOAD_FUNCTIONS.keys()}"
    assert dataset_file_type in ["json", "jsonl", "csv"], f"{dataset_file_type} is not one of [json, jsonl, csv]."

    # Download data and optionally sample
    data_download_fn = DOWNLOAD_FUNCTIONS[dataset_id]
    data = data_download_fn()
    if sample:
        data = random.sample(data, int(sample))

    # Write to file 
    dset = Dataset(dataset_id=dataset_id, data = data)
    
    dset.write_to_file(dataset_folder=dataset_folder, save_path_overwrite = save_path_overwrite, dataset_file_type = dataset_file_type)
    

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
        required=False,
        default="dataset_downloads",
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