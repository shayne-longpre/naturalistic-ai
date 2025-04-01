import json
import re
import argparse
from taxonomy import OPTIONS
from collections import Counter


def extract_options(level_key, prompt_key):
    if level_key not in OPTIONS or prompt_key not in OPTIONS[level_key]:
        raise ValueError(f"Invalid keys: {level_key}:{prompt_key} not found in OPTIONS.")

    raw = OPTIONS[level_key][prompt_key]
    labels = []
    for line in raw.strip().splitlines():
        if line.strip().startswith('- '):
            label_part = line.strip()[2:].split(':', 1)[0].strip()
            labels.append(label_part)
    return labels


def extract_json_from_response(response_str):
    try:
        match = re.search(r'```json\s*(.*?)\s*```', response_str, re.DOTALL)
        if match:
            response_str = match.group(1)
        return json.loads(response_str)
    except json.JSONDecodeError:
        return None


def validate_entry(args, entry):
    response_data = extract_json_from_response(entry['response'])
    if not isinstance(response_data, list):
        return False, 'Invalid JSON list'

    task_options = extract_options(args.level_id, args.prompt_id)
    for item in response_data:
        if not isinstance(item, dict):
            return False, 'Item is not a dictionary'
        if 'labels' not in item or 'confidence' not in item:
            return False, 'Missing keys'
        if not isinstance(item['confidence'], (int, float)) or not (0 <= item['confidence'] <= 1):
            return False, 'Confidence out of range'
        if item['labels'] not in task_options:
            return False, f'Invalid option'
    return True, 'Valid'


def main(args):
    total_count = 0
    invalid_reasons = Counter()

    with open(args.input, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f, 1):
            entry = json.loads(line)
            total_count += 1
            is_valid, reason = validate_entry(args, entry)
            if not is_valid:
                invalid_reasons[reason] += 1

    print("Total count: ", total_count)

    if invalid_reasons:
        print("=======Invalid Reason Summary=======")
        for reason, count in invalid_reasons.items():
            print(f'{reason}: {count}')
    else:
        print("All entries are correctly formatted.")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input jsonl file.")
    parser.add_argument("--level_id", required=True, default=None)
    parser.add_argument("--prompt_id", required=True, default=None)
    args = parser.parse_args()

    main(args)