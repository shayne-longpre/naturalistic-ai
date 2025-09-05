import sys
import json
import re
import copy
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
from src.helpers import io

sys.path.append("../")

from src.helpers import constants


# Helper Functions
def clean_annotation_label(txt):
    for punct in ".,!?':;/()-_&":
        txt = txt.replace(punct, " ")
    return " ".join(txt.lower().strip().split())


def extract_options(options_dict: Dict[str, Dict[str, str]], level_key: str, prompt_key: str) -> List[str]:
    """Extract labels from a predefined options dictionary."""
    return constants.ANNOTATION_TAXONOMY_REVERSE_REMAPPER[f"{level_key}_{prompt_key}"].keys()

    # print(options_dict.keys())
    # if level_key not in options_dict or prompt_key not in options_dict[level_key]:
    #     return []
    #     # raise ValueError(f"Invalid keys: {level_key}:{prompt_key} not found.")

    # raw = options_dict[level_key][prompt_key]
    # return [line.strip()[2:].split(':', 1)[0].strip()
    #         for line in raw.strip().splitlines() if line.strip().startswith('- ')]


def extract_json(response_str: str) -> Optional[List[Dict[str, Any]]]:
    """Extract JSON list from a response string."""
    try:
        match = re.search(r'```json\s*(.*?)\s*```', response_str, re.DOTALL)
        json_str = match.group(1) if match else response_str
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def label_in_options_partial(label: str, options: List[str]) -> bool:
    """Check if a label is partially matched based on provided options."""
    norm_label = clean_annotation_label(label)
    matches = [
        norm_label == clean_annotation_label(opt) or norm_label in clean_annotation_label(opt)
        for opt in options
    ]
    # if not any(matches):
    #     print(label)
    return any(matches)

# OLD VERSION:
# def label_in_options_exact(label: str, options: List[str]) -> bool:
#     """Check if a label is valid based on provided options."""
#     def normalize_label(label: str) -> str:
#         """Normalize labels by removing extra spaces and hyphens."""
#         return label.replace('- ', '-').strip()

#     norm_label = normalize_label(label)
#     return any(norm_label == normalize_label(opt) for opt in options)

# OLD VERSION:
# def label_in_options_partial(label: str, options: List[str]) -> bool:
#     """Check if a label is partially matched based on provided options."""
#     def normalize_label(label: str) -> str:
#         return label.replace('- ', '-').strip()

#     norm_label = normalize_label(label)
#     return any(
#         norm_label == normalize_label(opt) or norm_label in normalize_label(opt)
#         for opt in options
#     )

# Core Validation Function

def validate_entry(
    entry: Dict[str, Any], 
    options: List[str],
    default_to_none: bool = False,
) -> Tuple[bool, str, Optional[List[Dict[str, Any]]]]:
    """Validate an entry against options and extract JSON data if valid.
    
    Summary: Right now this function just checks dictionary is well formed, and that labels, when normalized, 
    match something from "../src/scripts/taxonomy_options.json".

    TODO: After, we should remap all labels so they look like ones from Cedric.
    """

    response_data = extract_json(entry.get('response', ''))
    if not isinstance(response_data, list):
        return 'Invalid JSON list', None

    if response_data == []:
        if default_to_none:
            return "Valid", [{"labels": "None", "confidence": 1.0}]
        else:
            return "Returned empty response", []

    for item in response_data:
        if not isinstance(item, dict):
            return 'Item not a dictionary', response_data
        if 'labels' not in item or 'confidence' not in item:
            return 'Missing keys', response_data
        if not isinstance(item['confidence'], (int, float)) or not (0 <= item['confidence'] <= 1):
            return 'Confidence out of range', response_data

        labels = item['labels']
        # if isinstance(labels, list) and labels[0] == None:
        #     print('zero')
        labels = labels if isinstance(labels, list) else [labels]

        if not all(label_in_options_partial(label, options) for label in labels):
            return 'Invalid option', response_data

    return 'Valid', response_data


def parse_automatic_annotations(
    fpath, 
    conf_threshold: int=0.3,
    verbose: bool=False
):
    # print(fpath)
    raw_entries = io.read_jsonl(fpath)
    if not raw_entries:
        return []

    level_id = raw_entries[0]["level_id"]
    prompt_id = raw_entries[0]["prompt_id"]
    # Resolve taxonomy options path relative to this file so it works regardless of CWD
    _classes_dir = os.path.dirname(os.path.abspath(__file__))
    _src_dir = os.path.dirname(_classes_dir)
    _taxonomy_options_path = os.path.join(_src_dir, "scripts", "taxonomy_options.json")
    OPTIONS = io.read_json(_taxonomy_options_path)
    task_options = extract_options(OPTIONS, level_id, prompt_id)
    if not task_options:
        return []

    default_to_none = prompt_id in ("sensitive_use_flags", "interaction_features", "topic")
    # print(f"{prompt_id}: {default_to_none}")

    valid_entries, invalid_reasons = [], defaultdict(lambda: 0)
    for entry in raw_entries:


        is_invalid_reason, response_data = validate_entry(entry, task_options, default_to_none=default_to_none)
        if is_invalid_reason != "Valid":
            invalid_reasons[is_invalid_reason] += 1
            continue

        valid_labels = []
        valid_confidences = []
        for item in response_data:
            if item.get("confidence", 1.0) >= conf_threshold:
                labels = item['labels'] if isinstance(item['labels'], list) else [item['labels']]
                clean_labels = [clean_annotation_label(label) for label in labels]
                label_to_canonical_label = constants.ANNOTATION_TAXONOMY_REVERSE_REMAPPER[f"{level_id}_{prompt_id}"]
                
                # Handle case-insensitive mapping
                mapped_labels = []
                for label in clean_labels:
                    if label in label_to_canonical_label:
                        mapped_labels.append(label_to_canonical_label[label])
                    else:
                        # Try case-insensitive matching
                        found = False
                        for key, canonical in label_to_canonical_label.items():
                            if key.lower() == label.lower():
                                mapped_labels.append(canonical)
                                found = True
                                break
                        if not found:
                            # If still not found, try to find a partial match
                            for key, canonical in label_to_canonical_label.items():
                                if label.lower() in key.lower() or key.lower() in label.lower():
                                    mapped_labels.append(canonical)
                                    found = True
                                    break
                        if not found:
                            print(f"Warning: Could not map label '{label}' to canonical label for {level_id}_{prompt_id}")
                            # Skip this label
                            continue
                
                clean_labels = mapped_labels

                confidences = item['confidence'] if isinstance(item['confidence'], list) else [item['confidence']]
                valid_labels.extend(clean_labels)
                valid_confidences.extend(confidences)

        # if entry["ex_id"] == "sharegpt_3fTHHS8" and "gpt4o-free" in fpath and f"{level_id}_{prompt_id}" == "turn_sensitive_use_flags" and entry["turn"] == 12:
            # print(entry)
            # print(valid_labels)
            # print(response_data)

        new_entry = copy.deepcopy(entry)
        new_entry.update({"parsed_response": valid_labels, "parsed_confidence": valid_confidences})
        valid_entries.append(new_entry)

    # FOR TESTING WHAT ANNOTATIONS DO NOT MAP:
    # ------------------------------------------------
    # label_to_canonical_label = constants.ANNOTATION_TAXONOMY_REVERSE_REMAPPER[f"{level_id}_{prompt_id}"]
    # unmapped = set()
    # for entry in valid_entries:
    #     for label in entry["parsed_response"]:
    #         if label not in label_to_canonical_label:
    #             unmapped.add(label)
    # print(prompt_id)
    # print(unmapped)
    # ------------------------------------------------

    if verbose:
        total_invalid = sum(invalid_reasons.values())
        print(f"{level_id}-{prompt_id}: {total_invalid} / {len(raw_entries)} failed due to invalid annotations.")
    return valid_entries
