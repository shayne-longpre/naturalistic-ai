"""Utils for Clio scripts."""

import json
import os
import re
from collections import defaultdict

import pandas as pd

from src.clio_scripts.clio_constants import CLUSTER_DATA, SKILLS


def get_hierarchical_clusters(file_path=CLUSTER_DATA):
    """
    Reads the CLUSTER_DATA TSV file and constructs a nested dictionary representing hierarchical clusters.
    Example structure:
    {
        "Explain technical concepts and provide implementation guidance": {
            "Explain programming concepts and provide technical guidance": [
                "Explain AI model versions and their capabilities"
            ],
            "Help me implement programming algorithms and data structures": [
                "Implement and analyze tree data structures and algorithms"
                ]
        },
    }
    Args:
        df (pd.DataFrame): DataFrame containing cluster data.
    Returns:
        dict: Dictionary with hierarchical clusters.
    """
    # Read the TSV file into a DataFrame
    df = pd.read_csv(file_path, sep='\t')

    # Initialize the nested dictionary
    nested_dict = defaultdict(lambda: defaultdict(list))

    # Iterate through each row and build the nested dictionary
    for _, row in df.iterrows():
        level_2 = row['cluster_name_2']
        level_1 = row['cluster_name_1']
        level_0 = row['cluster_name_0']

        nested_dict[level_2][level_1].append(level_0)

    # Optionally convert to regular dict for cleaner printing or saving
    nested_dict = {k: dict(v) for k, v in nested_dict.items()}
    return nested_dict


def get_task_classification_prompt(options_str):
    """
    Constructs a prompt for task classification based on the provided options.
    Prompt is based on the original paper: https://arxiv.org/abs/2503.04761
    """
    # commenting out the random sampling of options_str to increase the use of cache
    # options_str = "\n".join(random.sample(options_str, len(options_str)))
    options_str = "\n".join(options_str)
    return f"""Consider the following list of classification options:

<options>
{options_str}
</options>

Your job is to identify which option best describes the previous conversation. In this case, the provided options are occupational tasks. Your job is to identify which task is performed by the assistant in the previous human-AI assistant conversation.

What is the answer? You MUST provide an option exactly as written above. If multiple options apply, choose the single-most pertinent one. First, start off by considering various aspects of the conversation in <scratchpad> tags in the most four sentences, and then provide the final answer in <answer> tags with no other commentary.
"""


def get_occupational_check_prompt():
    return """Your job is to answer this question about the preceding conversation:

<question>
Does the conversation possibly involve an occupational task?
</question>

What is the answer? You MUST answer either only "Yes" or "No". Provide the answer in <answer> tags with no other commentary.
"""


def get_skill_classification_prompt(options_str):
    """
    Constructs a prompt for skill classification based on the predefined options.
    Prompt and the skill options are based on the original paper: https://arxiv.org/abs/2503.04761
    """
    # shuffle SKILLS and set as options_str
    # options_str = "\n".join(random.sample(options_str, len(options_str)))
    # commenting out the random sampling of options_str to increase the use of cache
    options_str = "\n".join(options_str)
    return f"""Please identify which categories best describe the conversation. Consider the provided list of occupational skills. Your job is to identify ALL skills exhibited by the assitant in the following human-AI assitant conversation. You can select multiple skills if appropriate. Select all that apply. Please comma-separate your selections (e.g., 'social perceptiveness', 'science', 'writing', etc.) and provide no additonal commentary. If no skills are exhibited by the assitant return 'none'.

<options>
{options_str}
</options>
"""


def get_clio_message(conversation, content):
    """
    Returns a dictionary representing a message with the given content.
    Message is structured based on the original paper: https://arxiv.org/abs/2503.04761

    Args:
        content (str): The content of the message.

    Returns:
        dict: A dictionary with the message content.
    """
    message = [
        {"role": "user", "content": f"The following is a conversation between an AI assistant and a user:\n{conversation}"},
        {"role": "assistant", "content": "I understand"},
        {"role": "user", "content": content}
    ]
    return message


def load_existing_annotations(filepath):
    if not os.path.exists(filepath):
        return dict()

    existing_annotations = dict()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                existing_annotations[data.get('conversation_id')]  = data.get('clio_annotation')
    return existing_annotations


def init_clio_annotation_dict():
    """
    Initializes a dictionary to store CLIO annotations.

    Returns:
        dict: An empty dictionary to store CLIO annotations.
    """
    return {
        "is_occupational_task": "",
        "cluster_top": "",
        "cluster_medium": "",
        "cluster_bottom": "",
        "occupational_skills": "",
    }


def extract_answer_from_response(response, clio_annotation):
    """
    Extracts the answer from a GPT response.

    Args:
        response (str): The GPT response containing an <answer> tag.

    Returns:
        str: The extracted answer, or 'Unknown' if not found.
    """
    if clio_annotation in ["is_occupational_task", "cluster_top", "cluster_medium", "cluster_bottom"]:
        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return "Parsing error: <answer> format is not correct or missing."
    elif clio_annotation == "occupational_skills":
        return response.strip()


def assign_answer_to_annotation(annotation, response, clio_annotation):
    """
    Assigns the extracted answer to the appropriate CLIO annotation field.

    Args:
        annotation (dict): The CLIO annotation dictionary.
        response (str): The GPT response containing an <answer> tag.
        clio_annotation (str): The specific annotation field to update.

    Returns:
        dict: Updated CLIO annotation dictionary.
    """
    answer = extract_answer_from_response(response, clio_annotation)
    annotation[clio_annotation] = answer

    return annotation, response


def get_task_list(annotation):
    ordered_keys = []
    clusters_by_order = ["cluster_top", "cluster_medium", "cluster_bottom"]
    for idx, cluster in enumerate(clusters_by_order):
        if annotation[cluster] != "":
            ordered_keys.append(annotation[cluster])
    return ordered_keys


def get_next_keys(annotation):
    clio_tasks_dict = get_hierarchical_clusters()
    ordered_keys = get_task_list(annotation)

    if len(ordered_keys) == 3:
        return []  # no more tasks to classify
    else:
        for idx, key in enumerate(ordered_keys):
            try:
                clio_tasks_dict = clio_tasks_dict[key]
            except KeyError:
                print(f"KeyError: {key} not found in clio_tasks_dict. Returning empty list.")
                return []
        return clio_tasks_dict if isinstance(clio_tasks_dict, list) else list(clio_tasks_dict.keys())


def format_samples_per_clio_annotation(samples, metadatas, conversation_ids, clio_annotations, annotation_key):
    """
    Formats samples based on the specified annotation key.
    metadatas, conversation_ids, and clio_annotations are preserved to keep track the order since any sample is skipped if len(next_keys) == 0
    Args:
        samples (list): List of sample objects.
        metadatas (list): List of metadata dictionaries.
        conversation_ids (list): List of conversation IDs.
        clio_annotations (list): List of CLIO annotations.
        annotation_key (str): The key to determine the formatting logic. Options are "is_occupational_task", "cluster_top", "cluster_medium", "cluster_bottom", "occupational_skills".
    Returns:
        tuple: A tuple containing formatted samples, metadatas, conversation_ids, and clio_annotations.
    """
    # Initialize lists to hold formatted samples, metadatas, conversation_ids, and clio_annotations
    formatted_samples, formatted_metadatas, formatted_conversation_ids, formatted_clio_annotations = [], [], [], []
    for sample, metadata, conversation_id, annotations in zip(samples, metadatas, conversation_ids, clio_annotations):
        if annotation_key == "is_occupational_task":
            formatted_samples.append(get_clio_message(sample.to_string(), get_occupational_check_prompt()))
            formatted_metadatas.append(metadata)
            formatted_conversation_ids.append(conversation_id)
            formatted_clio_annotations.append(annotations)
        elif annotation_key.startswith("cluster_"):
            next_keys = get_next_keys(annotations)
            if len(next_keys) == 0:
                continue
            else:
                formatted_samples.append(get_clio_message(sample.to_string(), get_task_classification_prompt(next_keys)))
                formatted_metadatas.append(metadata)
                formatted_conversation_ids.append(conversation_id)
                formatted_clio_annotations.append(annotations)
        elif annotation_key == "occupational_skills":
            formatted_samples.append(get_clio_message(sample.to_string(), get_skill_classification_prompt(SKILLS)))
            formatted_metadatas.append(metadata)
            formatted_conversation_ids.append(conversation_id)
            formatted_clio_annotations.append(annotations)
    # Return the formatted samples, metadatas, conversation_ids, and clio_annotations
    return formatted_samples, formatted_metadatas, formatted_conversation_ids, formatted_clio_annotations
