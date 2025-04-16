import os
import sys
import typing
import json
import re
import math
import copy
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Optional

sys.path.append("./")

from src.helpers import io
from src.scripts.evaluator.taxonomy import OPTIONS


# Helper Functions

def extract_options(options_dict: Dict[str, Dict[str, str]], level_key: str, prompt_key: str) -> List[str]:
    """Extract labels from a predefined options dictionary."""
    # print(options_dict.keys())
    if level_key not in options_dict or prompt_key not in options_dict[level_key]:
        return []
        # raise ValueError(f"Invalid keys: {level_key}:{prompt_key} not found.")

    raw = options_dict[level_key][prompt_key]
    return [line.strip()[2:].split(':', 1)[0].strip()
            for line in raw.strip().splitlines() if line.strip().startswith('- ')]


def extract_json(response_str: str) -> Optional[List[Dict[str, Any]]]:
    """Extract JSON list from a response string."""
    try:
        match = re.search(r'```json\s*(.*?)\s*```', response_str, re.DOTALL)
        json_str = match.group(1) if match else response_str
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def label_in_options(label: str, options: List[str]) -> bool:
    """Check if a label is valid based on provided options."""
    def normalize_label(label: str) -> str:
        """Normalize labels by removing extra spaces and hyphens."""
        return label.replace('- ', '-').strip()

    norm_label = normalize_label(label)
    return any(norm_label == normalize_label(opt) for opt in options)

# Core Validation Function

def validate_entry(
    entry: Dict[str, Any], 
    options: List[str]
) -> Tuple[bool, str, Optional[List[Dict[str, Any]]]]:
    """Validate an entry against options and extract JSON data if valid."""
    response_data = extract_json(entry.get('response', ''))
    if not isinstance(response_data, list):
        return False, 'Invalid JSON list', None

    for item in response_data:
        if not isinstance(item, dict):
            return False, 'Item not a dictionary', None
        if 'labels' not in item or 'confidence' not in item:
            return False, 'Missing keys', None
        if not isinstance(item['confidence'], (int, float)) or not (0 <= item['confidence'] <= 1):
            return False, 'Confidence out of range', None

        labels = item['labels']
        labels = labels if isinstance(labels, list) else [labels]

        if not all(label_in_options(label, options) for label in labels):
            return False, 'Invalid option', None

    return True, 'Valid', response_data


def parse_automatic_annotations(
    path, 
    conf_threshold: int=0.3,
    verbose: bool=False
):
    raw_entries = io.read_jsonl(path)
    if not raw_entries:
        return []

    level_id = raw_entries[0]["level_id"]
    prompt_id = raw_entries[0]["prompt_id"]
    task_options = extract_options(OPTIONS, level_id, prompt_id)
    if not task_options:
        return []

    valid_entries, invalid_reasons = [], defaultdict(lambda: 0)
    for entry in raw_entries:
        is_valid, reason, response_data = validate_entry(entry, task_options)
        if not is_valid:
            invalid_reasons[reason] += 1
            continue

        valid_labels = []
        for item in response_data:
            if item.get("confidence", 1.0) >= conf_threshold:
                labels = item['labels'] if isinstance(item['labels'], list) else [item['labels']]
                valid_labels.extend(labels)

        new_entry = copy.deepcopy(entry)
        new_entry.update({"parsed_response": valid_labels})
        valid_entries.append(new_entry)

    if verbose:
        total_invalid = sum(invalid_reasons.values())
        print(f"{level_id}-{prompt_id}: {total_invalid} / {len(raw_entries)} failed due to invalid annotations.")
    return valid_entries




