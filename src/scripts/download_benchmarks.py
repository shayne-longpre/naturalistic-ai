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
# from src.classes.message import Message
from src.classes.conversation import Conversation
from src.classes.dataset import Dataset

from src.helpers import io
import uuid
import requests, json

"""
This file is used to download and format datasets in a common format (list of Conversation objects). 
To download a script, run something like: "python src/scripts/download_datasets.py --dataset_id=mmlu". 
See arg description for specifics. 
"""

def download_lmsys_1m():
    """
    Huggingface: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
    Lymsys Chat 1M is a dataset of 1 million conversations between humans and various AI models collected from the LMSYS platform.
    """
    print("\nDownloading lmsys-chat-1m...")
    # https://huggingface.co/datasets/lmsys/lmsys-chat-1m
    dset = io.huggingface_download('lmsys/lmsys-chat-1m', split='train')

    def process_data(datum):
        conversation = [
            {
                "role": msg.get("role"),
                "turn": idx,
                "content": msg.get("content", ""),
                "image": "",
            }
            for idx, msg in enumerate(datum.get("conversation", []))
        ]

        return Conversation(
            conversation_id="lmsys1m_" + datum.get('conversation_id'),
            dataset_id="lmsys1m_",
            user_id=None,
            time=None,
            model=datum.get('model'),
            conversation=conversation,
            geography=None
        )

    print("Processing lmsys-chat-1m into conversation format...")
    return [process_data(datum) for datum in tqdm(dset, desc="Processing lymsys-chat-1m")]

# Download WildChat
def download_wildchat_v1():
    """
    Huggingface: https://huggingface.co/datasets/allenai/WildChat-1M
    WildChat is a dataset of conversations between humans and various AI models collected from the WildChat platform. This is the public set. 
    """
    print("Starting Download for WildChat-1M...")
    dset = io.huggingface_download("allenai/WildChat-1M", split="train")

    def process_data(datum):
        state = datum.get('state')
        country = f"{datum.get('country', 'Unknown')}"
        timestamp = datum.get('timestamp')
        
        conversation = [
            {
                "role": msg.get("role"),
                "turn": idx,
                "content": msg.get("content", ""),
                "image": "",
            }
            for idx, msg in enumerate(datum.get("conversation", []))
        ]

        return Conversation(
            conversation_id="wildchat_" + datum.get('conversation_hash'),
            dataset_id="wildchat_1m",
            user_id=datum.get('hashed_ip'),
            time=timestamp.isoformat() if isinstance(timestamp, datetime) else None,
            model=datum.get('model'),
            conversation=conversation,
            geography=country if state is None else f"{country}; {state}"
        )
    
    return [process_data(datum) for datum in tqdm(dset, desc="Processing WildChat")]

# Download WildChat (private repo)
def download_wildchat_private(sample=None):
    """
    Huggingface: https://huggingface.co/datasets/yuntian-deng/WildChat-1M-Full-with-parameters-internal
    WildChat is a dataset of conversations between humans and various AI models collected from the WildChat platform. This is the private set. 
    """
    print("Starting Download for Yuntian's WildChat-1M-Full...")
    dset = io.huggingface_download("yuntian-deng/WildChat-1M-Full-with-parameters-internal", split="train", sample=sample)

    def process_data(datum):
        state = datum.get('state')
        country = f"{datum.get('country', 'Unknown')}"
        timestamp = datum.get('timestamp')

        conversation = [
            {
                "role": msg.get("role"),
                "turn": idx,
                "content": msg.get("content", ""),
                "image": "",
            }
            for idx, msg in enumerate(datum.get("conversation", []))
        ]

        return Conversation(
            conversation_id="wildchat_" + datum.get('conversation_hash'),
            dataset_id="wildchat_1m_full",
            user_id=datum.get('hashed_ip'),
            time=timestamp.isoformat() if isinstance(timestamp, datetime) else None,
            model=datum.get('model'),
            conversation=conversation,
            geography=country if state is None else f"{country}; {state}",
        )

    processed_data = [process_data(datum) for datum in tqdm(dset, desc="Processing WildChat Full")]
    dataset = Dataset(dataset_id="wildchat_1m_full", data=processed_data)
    return dataset


# Download ShareGPT
def download_sharegpt_v1():
    """
    Huggingface: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
    ShareGPT is a dataset of conversations between humans and GPT-based models collected from the ShareGPT platform.
    """
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
                "turn": idx,
                "content": msg.get("value", ""),
                "image": msg.get("image", ""),
            })

        return Conversation(
            conversation_id="sharegpt_" + datum.get('id'),
            dataset_id="sharegpt",
            user_id=None,
            time=None,  # TODO: fill in rough period
            model=None,  # TODO: fill in OpenAI models at that time
            conversation=conversation,
            geography=None
        )

    return [process_data(datum) for datum in tqdm(full_dset, desc="Processing ShareGPT")] 

# Download ChatBotArena 
def download_chatbot_arena():
    """
    Huggingface: https://huggingface.co/datasets/lmsys/chatbot_arena_conversations
    Chatbot Arena is a dataset of conversations between different chatbots and human judges.
    """
    print("Starting Download for ChatBotArena...")
    dset = io.huggingface_download("lmsys/chatbot_arena_conversations", split="train")

    def add_turn_and_rename_keys(conv:list[object]):
        conv_with_turn = []
        for idx, statement in enumerate(conv):
            new_statement = {
                "turn": idx,
                "content": statement.get("content"),
                "image": statement.get("image"),
            }
            conv_with_turn.append(new_statement)
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
            conversation_id="chatbot_arena_" + datum.get('question_id') + "_a",
            dataset_id="chatbot_arena",
            user_id=datum.get('judge'),
            time=timestamp.isoformat() if isinstance(timestamp, datetime) else None,
            model=datum.get('model_a'),
            conversation=conv_a_reformatted,
            geography=country if state is None else f"{country}; {state}"
        )
        
        conversation_b = Conversation(
            conversation_id="chatbot_arena_" + datum.get('question_id') + "_b",
            dataset_id="chatbot_arena",
            user_id=datum.get('judge'),
            time=timestamp.isoformat() if isinstance(timestamp, datetime) else None,
            model=datum.get('model_b'),
            conversation=conv_b_reformatted,
            geography=country if state is None else f"{country}; {state}"
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
    """
    Huggingface: https://huggingface.co/datasets/tatsu-lab/alpaca_eval
    Aplaca Eval is a benchmark for evaluating instruction-following models. It contains instructions as prompts.
    """
    print("Starting Download for AlpacaEval..")
    dset = load_dataset("tatsu-lab/alpaca_eval", split = "eval", trust_remote_code=True, token = True) #TODO integrate this with io helpers

    def process_data(datum):
        conv = [
            {
                "role": "user",
                "turn": 0, 
                "content": datum.get("instruction"),
                "image": "",
            }, 
            {
                "role": "assistant",
                "turn": 1, 
                "content": datum.get("output"),
                "image": "",
            }
        ]

        
        return Conversation(
            conversation_id="alpaca_eval" + str(uuid.uuid4()).replace("-", ""),
            dataset_id="alpaca_eval",
            user_id=str(uuid.uuid4()),
            time=None,
            model=datum.get('generator'),
            conversation=conv,
            geography="Unknown"
        )
        

    return [process_data(datum) for datum in tqdm(dset, desc="Processing AlpacaEval")]

# Download MMLU
def download_mmlu():
    """
    Huggingface: https://huggingface.co/datasets/tasksource/mmlu
    MMLU is a multiple-choice benchmark dataset for measuring model performance across a diverse set of subjects.
    """
    print("Starting Download for MMLU...")
    # ['question', 'choices', 'answer'],
    categories = [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
            "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
            "college_medicine", "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
            "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
            "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history",
            "high_school_world_history", "human_aging", "human_sexuality", "international_law", "jurisprudence",
            "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics",
            "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory",
            "professional_accounting", "professional_law", "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
        ]
    choice_indicators = ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)", "i)", "j)", "k)", "l)", "m)", "n)", "o)", "p)", "q)", "r)", "s)", "t)", "u)", "v)", "w)", "x)", "y)", "z)"]
    def process_data(datum):
            choices = datum.get("choices")
           
        
            conv = [
                {
                    "role": "user",
                    "turn": 0, 
                    "content": datum.get("question") + " " + " ".join(f"{choice_indicators[i]} {datum.get("choices")[i]}" for i in range(len(datum.get("choices")))),
                    "image": "",
                }, 
                {
                    "role": "assistant",
                    "turn": 1, 
                    "content": choice_indicators[datum.get("answer")] +" " + choices[datum.get("answer")],  # a) <answer>
                    "image": "",
                }
            ]

            return Conversation(
                conversation_id="mmlu_" + str(uuid.uuid4()).replace("-", ""),
                dataset_id="mmlu",
                user_id=str(uuid.uuid4()).replace("-", ""),
                time=None,
                model=None,
                conversation=conv,
                geography="Unknown"
            )
    
    conversations_to_return = []
    for category in tqdm(categories, desc="Processing MMLU Categories"):
        dset = load_dataset("tasksource/mmlu", category, token=True)["test"]
        for datum in dset:
            conversations_to_return.append(process_data(datum))
    
    return conversations_to_return

# Download HLE
# TODO: should we include the rationale? 
def download_hle():
    """
    Huggingface: https://huggingface.co/datasets/cais/hle
    Humanity's Last Exam is a dataset of very challenging problems designed to be "the final closed-ended academic benchmark of its kind with broad subject coverage" (from huggingface). 
    """
    print("Starting Download for HLE (Humanity's Last Exam)...")
    dset = io.huggingface_download('cais/hle', split='test')

    def process_data(datum):
        conv = [
            {
                "role": "user",
                "turn": 0, 
                "content": datum.get("question"),
                "image": datum.get("image") if datum.get("image") else ''
            }, 
            {
                "role": "assistant",
                "turn": 1, 
                "content": datum.get("answer"),
                "image": "",
            }
        ]
        
        return Conversation(
            conversation_id="hle_" + datum.get('id'),
            dataset_id="hle",
            user_id=str(datum.get('author_name')),
            time="02/11/2025", # huggingface release date
            model=None,
            conversation=conv,
            geography="Unknown"
        )
    
    
    return [process_data(datum) for datum in tqdm(dset, desc="Processing Humanity's Last Exam")]

# Download GPQA
def download_gpqa():
    """
    Huggingface: https://huggingface.co/datasets/Idavidrein/gpqa
    The dataset contains graduate-level multiple choice questions with 4 answer options. We shuffle the answer order. 
    """
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
        
        conv = [
            {
                "role": "user",
                "turn": 0, 
                "content": f"{datum.get('Question')}\nChoices:\n" + "\n".join(choices),
                "image": '',
            }, 
            {
                "role": "assistant",
                "turn": 1, 
                "content": datum.get("Correct Answer"),
                "image": "",
            }
        ]
        

        return Conversation(
            conversation_id="gpqa_" + datum.get('Record ID'),
            dataset_id="gpqa",
            user_id=str(datum.get('Question Writer')),
            time="11/29/2023",
            model=None,
            conversation=conv,
            geography="Unknown"
        )
    
    return [process_data(datum) for datum in tqdm(dset, desc="Processing GPQA")]


# Download SWE-Bench
def download_swebench():
    """
        Huggingface: https://huggingface.co/datasets/princeton-nlp/SWE-bench
        The dataset contains github repository issues and their solution. The model is given the problem, the repo, and the base commit. 
    """
    print("Starting Download for SWE Bench...")
    dset = io.huggingface_download('princeton-nlp/SWE-bench', split="test")

    def process_data(datum):
        
        conv = [
            {
                "role": "user",
                "turn": 0, 
                "content": f"Problem Statement: {datum.get('problem_statement')} \nRepo: {datum.get('repo')} \nBase_commit: {datum.get('base_commit')} \n",
                "image": '',
            }, 
            {
                "role": "assistant",
                "turn": 1, 
                "content": datum.get("solution"),
                "image": "",
            }
        ]
        

        return Conversation(
            conversation_id="swebench_" + datum.get('instance_id'),
            dataset_id="swebench",
            user_id=str(datum.get('repo')), # Is the repo a good indicator of user_id?
            time=datum.get('created_at').isoformat() if isinstance(datum.get('timestamp'), datetime) else None,
            model=None,
            conversation=conv,
            geography="Unknown"
        )
    
    return [process_data(datum) for datum in tqdm(dset, desc="Processing SweBench")]

# Download MBPP 
def download_mbpp():
    """
    Huggingface: https://huggingface.co/datasets/Muennighoff/mbpp
    This dataset contains 1000 Python programming problems that are crowd-sourced and designed to be solvable by entry-level programmers.
    """
    print("Starting Download for MBPP...")
  
    dset = io.huggingface_download("mbpp", split="test")
    
    def process_data(datum):
        conversation = [
            {
                "role": "user",
                "turn": 0,
                "content": datum.get("text", "") + datum.get("code", ""),
                "image": "",
            }, 
            {
                "role": "assistant",
                "turn": 1,
                "content": datum.get("code", ""),
                "image": "",
            }
        ]

        conversation_id = "mbpp_" + str(uuid.uuid4()).replace("-", "")
        
        return Conversation(
            conversation_id=conversation_id,
            dataset_id="mbpp",
            user_id=None,
            time=None,
            model=None,
            conversation=conversation,
            geography="Unknown"
        )
        
    return [process_data(datum) for datum in tqdm(dset, desc="Processing MBPP")]

def download_humaneval():
    """
     Huggingface: https://huggingface.co/datasets/openai/openai_humaneval
     The dataset contains coding problems. We use the test set here.
    """
    print("Starting Download for HumanEval...")
    dset = io.huggingface_download("openai/openai_humaneval", split="test")

    def process_data(datum):
        conv = [
            {
                "role": "user",
                "turn": 0,
                "content": datum.get("prompt", ""),
                "image": "",
            }, 
            {
                "role": "assistant",
                "turn": 1,
                "content": datum.get("canonical_solution", ""),
                "image": "",
            }
        ]
        return Conversation(
            conversation_id="humaneval_" + str(uuid.uuid4()).replace("-", ""),
            dataset_id="humaneval",
            user_id=None,
            time=None,
            model=None,
            conversation=conv,
            geography="Unknown"
        )

    return [process_data(datum) for datum in tqdm(dset, desc="Processing HumanEval")]

def download_gsm8k():
    """
    Huggingface: https://huggingface.co/datasets/openai/gsm8k
    The dataset contains grade school math problems. We use the test set here.
    """
    print("Starting Download for GSM8K...")
    dset = load_dataset("openai/gsm8k", "main", split = "test")

    def process_data(datum):
        conv = [
            {
                "role": "user",
                "turn": 0,
                "content": datum.get("question", ""),
                "image": "",
            }
        ]
        return Conversation(
            conversation_id="gsm8k_" + str(uuid.uuid4()).replace("-", ""),
            dataset_id="gsm8k",
            user_id=None,
            time=None,
            model=None,
            conversation=conv,
            geography="Unknown"
        )

    return [process_data(datum) for datum in tqdm(dset, desc="Processing GSM8K")]

def download_bbh():
    """
     Huggingface: https://huggingface.co/datasets/lukaemon/bbh
     The dataset contains big bench hard, which is split between problems in boolean expressions, causal_judgement, and date_understanding. 
     We combine these and label with the category in the conversation id.  
    """
    print("Starting Download for BBH (lukaemon/bbh)...")
    categories = ["boolean_expressions", "causal_judgement", "date_understanding"]
    conversations_to_return = []
    for category in categories:
        dset = io.huggingface_download("lukaemon/bbh", category, split ="test")
        for datum in tqdm(dset, desc=f"Processing BBH {category}"):
            conversation = [
                {
                    "role": "user",
                    "turn": 0,
                    "content": datum.get("input", ""),
                    "image": "",
                }, 
                {
                    "role": "assistant",
                    "turn": 1,
                    "content": datum.get("target", ""),
                    "image": "",
                }
            ]
            conv = Conversation(
                conversation_id=f"bbh_{category}_" + str(uuid.uuid4()).replace("-", ""),
                dataset_id="bbh",
                user_id=None,
                time=None,
                model=None,
                conversation=conversation,
                geography="Unknown"
            )
            conversations_to_return.append(conv)
    return conversations_to_return

def download_mmlupro():
    """
     Huggingface: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
     The dataset contains more difficult questions than the original MMLU. We use the test set here.
    """
    print("Starting Download for MMLU-Pro...")
    choice_indicators = ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)", "i)", "j)", "k)", "l)", "m)", "n)", "o)", "p)", "q)", "r)", "s)", "t)", "u)", "v)", "w)", "x)", "y)", "z)"]
    
    def process_data(datum):
        # Construct the question string with choices similar to the MMLU script.
        question_text = datum.get("question", "")
        choices = datum.get("options", [])
        print(len(choices))
        choices_text = " ".join(f"{choice_indicators[i]} {choices[i]}" for i in range(len(choices)))
        print(datum.get("answer_index"))
        conv = [
            {
                "role": "user",
                "turn": 0,
                "content": question_text + " " + choices_text,
                "image": "",
            }, 
            {
                "role": "assistant",
                "turn": 1,
                "content": choice_indicators[datum.get("answer_index")] +" " + choices[datum.get("answer_index")],  # a) <answer>
                "image": "",
            }
        ]
        return Conversation(
            conversation_id="mmlupro_" + str(uuid.uuid4()).replace("-", ""),
            dataset_id="mmlupro",
            user_id=str(uuid.uuid4()).replace("-", ""),
            time=None,
            model=None,
            conversation=conv,
            geography="Unknown"
        )
    
    conversations = []
    dset = load_dataset("TIGER-Lab/MMLU-Pro", split = "test", token=True)
    for datum in tqdm(dset, desc="Processing MMLU-Pro"):
        conversations.append(process_data(datum))
    return conversations

def download_google_ifeval():
    """
     Huggingface: https://huggingface.co/datasets/google/IFEval
     This dataset contains around 500 "verifiable instructions" such as "write in more than 400 words" and "mention the keyword of AI at least 3 times" which can be verified by heuristics. 
    """
    print("Starting Download for google/IFEval...")
    dset = io.huggingface_download("google/IFEval", split="train")

    def process_data(datum):
        conv = [
            {
                "role": "user",
                "turn": 0,
                "content": datum.get("prompt", ""),
                "image": "",
            }, 
        ]
        return Conversation(
            conversation_id="ifeval_" + str(uuid.uuid4()).replace("-", ""),
            dataset_id="ifeval",
            user_id=None,
            time=None,
            model=None,
            conversation=conv,
            geography="Unknown"
        )

    return [process_data(datum) for datum in tqdm(dset, desc="Processing IFEval")]

def download_aime2025():
    """
     Huggingface: https://huggingface.co/datasets/opencompass/AIME2025
     The dataset contains math problems from the AIME 2025 competition. We use the test set here.
    """
    print("Starting Download for AIME2025...")
    dset = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")

    def process_data(datum):
        conv = [
            {
                "role": "user",
                "turn": 0,
                "content": datum.get("question", ""),
                "image": "",
            }, 
            {
                "role": "assistant",
                "turn":  1,
                "content": datum.get("answer", ""),
                "image": "",
            }
        ]
        return Conversation(
            conversation_id="aime2025_" + str(uuid.uuid4()).replace("-", ""),
            dataset_id="aime2025",
            user_id=None,
            time=None,
            model=None,
            conversation=conv,
            geography="Unknown"
        )
    
    return [process_data(datum) for datum in tqdm(dset, desc="Processing AIME2025")]

# TODO: The answers here are the private test cases. Should we include them?
def download_code_generation_lite():
    """
     Huggingface: https://huggingface.co/datasets/livecodebench/code_generation_lite
     The dataset contains coding problems with starter code and public test cases. We use the test set here.
    """
    print("Starting Download for Code Generation Lite...")
    dset = load_dataset("livecodebench/code_generation_lite", version_tag="release_v5", split = "test", trust_remote_code=True)
    #Features: ['question_title', 'question_content', 'platform', 'question_id', 'contest_id', 'contest_date', 'starter_code', 'difficulty', 'public_test_cases', 'private_test_cases', 'metadata'],
    
    def process_data(datum):
        conv = [
            {
                "role": "user",
                "turn": 0,
                "content": datum.get("question_content", "") + "\n\n" + datum.get("question_content", "") + "\n\n" + datum.get("starter_code", "") + "\n\n" + "Public Test Cases:\n" + datum.get("public_test_cases", ""),
                "image": "",
            }, 
            {
                "role": "assistant",
                "turn": 1,
                "content": datum.get("private_test_cases", ""),
                "image": "",
            }
        ]
        return Conversation(
            conversation_id="codegen_lite_" + str(uuid.uuid4()).replace("-", ""),
            dataset_id="code_generation_lite",
            user_id=None,
            time=None,
            model=None,
            conversation=conv,
            geography="Unknown"
        )
    
    return [process_data(datum) for datum in tqdm(dset, desc="Processing Code Generation Lite")]

def download_drop():
    """
     Huggingface: https://huggingface.co/datasets/ucinlp/drop
     The dataset contains reading comprehension questions. We use the validation set here.
     """
    print("Starting Download for DROP (validation set)...")
    dset = load_dataset("ucinlp/drop", split="validation")

    def process_data(datum):
        content = f"Passage: {datum.get('passage', '')}\nQuestion: {datum.get('question', '')}"
        answer = datum.get("answer_spans", "")["spans"][0]
   

        conv = [{
            "role": "user",
            "turn": 0,
            "content": content,
            "image": ""
        }, 
        {
            "role": "assistant",
            "turn": 1,
            "content": answer,
            "image": ""
        }
        
        ]
        return Conversation(
            conversation_id="drop_validation_" + str(uuid.uuid4()).replace("-", ""),
            dataset_id="drop",
            user_id=None,
            time=None,
            model=None,
            conversation=conv,
            geography="Unknown"
        )

    return [process_data(datum) for datum in tqdm(dset, desc="Processing DROP Validation")]

def download_mgsm():
    """Note: this dataset has the same test set in 11 languages. We combine them all here, and label the language in the conversation_id.Total is 2750 samples.
    Huggingface: https://huggingface.co/datasets/juletxara/mgsm.
    """
    print("Starting Download for MGSM...")
    languages = ['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh']
    conversations = []
    for lang in languages:
        print(f"Processing language: {lang}")
        ds = load_dataset("juletxara/mgsm", lang, split="test")

        for datum in tqdm(ds, desc=f"Processing MGSM {lang}"):

            conv = [
                {
                    "role": "user",
                    "turn": 0,
                    "content": datum.get("question", ""),
                    "image": "",
                }, 
                {
                    "role": "assistant",
                    "turn": 1,
                    "content": datum.get("answer_number", ""),
                    "image": "",
                }
            ]
            conversations.append(Conversation(
                conversation_id=f"mgsm_{lang}_" + str(uuid.uuid4()).replace("-", ""),
                dataset_id="mgsm",
                user_id=None,
                time=None,
                model=None,
                conversation=conv,
                geography="Unknown"
            ))
    return conversations

def download_multilingual_mmlu():
    """
     Huggingface: https://huggingface.co/datasets/openai/MMMLU
     The dataset contains the same questions in multiple languages. We combine them all here. 
    """
    print("Starting Download for Multilingual MMLU...")
    dset = load_dataset("openai/MMMLU", "default", token=True)["test"] # default contains all languages
    choice_indicators = ["A", "B", "C", "D"]    

    def process_data(datum):
        choices = {
            "A": datum.get("A"), 
            "B": datum.get("B"), 
            "C": datum.get("C"), 
            "D": datum.get("D")
        }

        choices_text = " ".join(
            f'{c}) {choices[c]}' for c in choice_indicators
        )
        # print(choices_text)
        # print(datum.get("Answer"))
        # print(choices[datum.get("Answer")])
        conv = [
            {
                "role": "user",
                "turn": 0,
                "content": datum.get("Question", "") + " " + choices_text,
                "image": "",
            }, 
            {
                "role": "assistant",
                "turn": 1,
                "content": datum.get("Answer", "") +") " + choices[datum.get("Answer", "")],
                "image": "",
            }
        ]
        return Conversation(
            conversation_id="multilingual_mmlu_" + str(uuid.uuid4()).replace("-", ""),
            dataset_id="multilingual_mmlu",
            user_id=str(uuid.uuid4()).replace("-", ""),
            time=None,
            model=None,
            conversation=conv,
            geography="Unknown"
        )

    return [process_data(datum) for datum in tqdm(dset, desc="Processing Multilingual MMLU")]

def download_lmarena_hard():
    """The huggingface repo for this dataset is https://huggingface.co/datasets/lmarena-ai/arena-hard-auto. There are two problems: 1) one of their JSON files is incorrectly formatted, breaking HF's loading API. 2) The dataset seems to only contain the model answers, not the questions. 
    To solve this, I found the questions file in their github repo and downloaded it directly in a website request. The repo is here: https://github.com/lmarena/arena-hard-auto/blob/main/data/arena-hard-v2.0/question.jsonl. 
    """
    print("Starting Download for lmarena_hard...")
    url = "https://raw.githubusercontent.com/lmarena/arena-hard-auto/196f6b826783b3da7310e361a805fa36f0be83f3/data/arena-hard-v2.0/question.jsonl"
    response = requests.get(url)
    lines = response.text.strip().split("\n")
    data = [json.loads(line) for line in lines]

    def process_data(datum):
        conv = [{
            "role": "user",
            "turn": 0,
            "content": datum.get("prompt", ""),
            "image": ""
        } 
        ]
        return Conversation(
            conversation_id="lmarena_hard_" + str(uuid.uuid4()).replace("-", ""),
            dataset_id="lmarena_hard",
            user_id=None,
            time=None,
            model=None,
            conversation=conv,
            geography="Unknown"
        )

    return [process_data(d) for d in tqdm(data, desc="Processing lmarena_hard")]


def download_codeforces_verifiable_prompts():
    """This dataset is a bit complicated on huggingface, containing a lot of metadata about the competition, and the official tests of code. 
    They release a format of their dataset structured for RL prompts, which we use here for ease. It combines the system prompt with the specific problem. 
    Huggingface: https://huggingface.co/datasets/open-r1/codeforces
    """
    print("Starting Download for Codeforces Verifiable Prompts...")
    dset = load_dataset("open-r1/codeforces", "verifiable-prompts", split="test")

    def process_data(datum):
        conv = [
            {
                "role": "user",
                "turn": 0,
                "content": datum.get("prompt", ""),
                "image": "",
            }, 
            {
                "role": "assistant",
                "turn": 1,
                "content": datum.get("official_tests", "") + datum.get("editorial" , ""),
                "image": "",
            }
        ]
        return Conversation(
            conversation_id="codeforces_" + str(uuid.uuid4()).replace("-", ""),
            dataset_id="codeforces",
            user_id=None,
            time=None,
            model=None,
            conversation=conv,
            geography="Unknown"
        )

    return [process_data(datum) for datum in tqdm(dset, desc="Processing Codeforces Verifiable Prompts")]

def download_math():
    """
    Huggingface: https://huggingface.co/datasets/qwedsacf/competition_math
    Loads competition_math which contains keys "problem" and "solution".
    """
    print("Starting Download for Competition Math...")
    dset = io.huggingface_download("qwedsacf/competition_math", split="train")

    def process_data(datum):
        conv = [
            {
                "role": "user",
                "turn": 0,
                "content": datum.get("problem", ""),
                "image": "",
            },
            {
                "role": "assistant",
                "turn": 1,
                "content": datum.get("solution", ""),
                "image": "",
            },
        ]

        return Conversation(
            conversation_id="math_" + str(uuid.uuid4()).replace("-", ""),
            dataset_id="math",
            user_id=None,
            time=None,
            model=None,
            conversation=conv,
            geography="Unknown",
        )

    return [process_data(datum) for datum in tqdm(dset, desc="Processing Competition Math (MATH)")]
    
def download_math500():
    """
    Huggingface: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
    MATH-500 contains math problems with keys "problem" and "solution".
    """
    print("Starting Download for MATH-500...")
    dset = io.huggingface_download("HuggingFaceH4/MATH-500", split="test")

    def process_data(datum):
        conv = [
            {
                "role": "user",
                "turn": 0,
                "content": datum.get("problem", ""),
                "image": "",
            },
            {
                "role": "assistant",
                "turn": 1,
                "content": datum.get("solution", ""),
                "image": "",
            },
        ]
        return Conversation(
            conversation_id="math500_" + str(uuid.uuid4()).replace("-", ""),
            dataset_id="math500",
            user_id=None,
            time=None,
            model=None,
            conversation=conv,
            geography="Unknown",
        )

    return [process_data(d) for d in tqdm(dset, desc="Processing MATH-500")]

DOWNLOAD_FUNCTIONS = {
    # Usage / Conversation Datasets
    "wildchat_v1": download_wildchat_v1,
    "wildchat_private": download_wildchat_private,
    "lmsys_1m": download_lmsys_1m,
    "sharegpt_v1": download_sharegpt_v1,
    "chatbot_arena": download_chatbot_arena,

    # Benchmark Datasets Used in At least 5 Releases in 2024 and 2025. 
    "mmlu": download_mmlu,
    "mbbpp": download_mbpp,
    "humaneval": download_humaneval, 
    "gsm8k": download_gsm8k,
    "math": download_math, 
    "bbh": download_bbh, 
    "gpqa": download_gpqa, 
    "mmlupro": download_mmlupro,    
    "ifeval": download_google_ifeval,
    "math-500": download_math500,
    "aime2025": download_aime2025,
    "code_gen_lite": download_code_generation_lite,
    "drop": download_drop, 
    "mgsm": download_mgsm,
    "multi_mmlu": download_multilingual_mmlu,
    "lmarena_hard": download_lmarena_hard,
    "codeforces": download_codeforces_verifiable_prompts,
   
    #Other Common Benchmarks
    "swebench": download_swebench,
    "hle": download_hle, 
    "alpaca_eval": download_alpaca_eval,
}



def download_dataset(
    dataset_id:str, 
    sample: int = None,
    save_path_overwrite: str = None, 
):
    
    assert dataset_id in DOWNLOAD_FUNCTIONS, f"{dataset_id} not in {DOWNLOAD_FUNCTIONS.keys()}"

    # Download data and optionally sample
    data_download_fn = DOWNLOAD_FUNCTIONS[dataset_id]
    
    data = data_download_fn()

    if sample is not None:
        data = random.sample(data, int(sample))

    # Write to file 

    if save_path_overwrite is not None and save_path_overwrite != "": 
        save_path = save_path_overwrite  
    else: 
        save_path = f"dataset_downloads/{dataset_id}/full.json"
    
    dset = Dataset(dataset_id=dataset_id, data=data)
    print(f"Saving {len(data)} conversations to {save_path}...")
    dset.save_to_json(json_path = save_path)

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
        default=None,
        help=f"An integer for how many to sample from the dataset."
    )
    parser.add_argument(
        "--save_path_overwrite",
        required=False,
        default=None,
        help="By default, Datasets are saved in data/<dataset_id>/full.json for consistency. To define a specific save path instead, provide the full path here."
    )
    args = parser.parse_args()
    download_dataset(args.dataset_id, args.sample, args.save_path_overwrite)