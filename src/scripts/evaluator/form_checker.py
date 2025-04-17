import json
import re
import os
import csv
import sys
import math
import argparse
from helpers import io
from collections import Counter

sys.path.append("./")


def extract_options(taxonomy_options, level_key, prompt_key):
    if level_key not in taxonomy_options or prompt_key not in taxonomy_options[level_key]:
        raise ValueError(f"Invalid keys: {level_key}:{prompt_key} not found in taxonomy options.")

    raw = taxonomy_options[level_key][prompt_key]
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


def bucket_confidence(conf):
    if conf < 0.2: return '[0.0-0.2)'
    elif conf < 0.4: return '[0.2-0.4)'
    elif conf < 0.6: return '[0.4-0.6)'
    elif conf < 0.8: return '[0.6-0.8)'
    else: return '[0.8-1.0]'


def compute_entropy(counts):
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def compute_gini(counts):
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return 1 - sum(p ** 2 for p in probs)


def normalize_label(label):
    return label.replace('- ', '-').strip()

def label_in_options(label, options):
    norm_label = normalize_label(label)
    return any(norm_label == normalize_label(opt) or norm_label in normalize_label(opt) for opt in options)


def validate_entry(entry, task_options):
    response_data = extract_json_from_response(entry['response'])
    if not isinstance(response_data, list):
        return False, 'Invalid JSON list', None

    for item in response_data:
        if not isinstance(item, dict):
            return False, 'Item is not a dictionary', None
        if 'labels' not in item or 'confidence' not in item:
            return False, 'Missing keys', None
        if not isinstance(item['confidence'], (int, float)) or not (0 <= item['confidence'] <= 1):
            return False, 'Confidence out of range', None

        if isinstance(item['labels'], list):
            if not all(label_in_options(label, task_options) for label in item['labels']):
                return False, 'Invalid option', item['labels']
        elif isinstance(item['labels'], str):
            if not label_in_options(item['labels'], task_options):
                return False, 'Invalid option', item['labels']
        else:
            return False, 'Invalid option', item['labels']
    return True, 'Valid', None


EXPECTED_INVALID_REASONS = [
    'Invalid JSON list',
    'Item is not a dictionary',
    'Missing keys',
    'Confidence out of range',
    'Invalid option'
]


def main(args, task_options, level_id, prompt_id):
    total_count = 0
    invalid_reasons = Counter()
    confidence_buckets = Counter()
    predictions_per_row = []
    all_labels = []
    label_freq = Counter()

    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            total_count += 1
            is_valid, reason, _ = validate_entry(entry, task_options)

            if not is_valid:
                invalid_reasons[reason] += 1
                continue

            response_data = extract_json_from_response(entry['response'])
            predictions_per_row.append(len(response_data))
            for item in response_data:
                confidence_buckets[bucket_confidence(item['confidence'])] += 1
                labels = item['labels']

                if isinstance(labels, list):
                    for label in labels:
                        all_labels.append(label)
                        label_freq[label] += 1
                else:  # string
                    all_labels.append(labels)
                    label_freq[labels] += 1

    result = {
        'level_id': level_id,
        'prompt_id': prompt_id,
        'total_entries': total_count,
        'avg_preds_per_row': round(sum(predictions_per_row) / len(predictions_per_row), 2) if predictions_per_row else 0,
        'unique_labels': f"{len(set(all_labels))} / {len(task_options)}",
        'label_entropy': round(compute_entropy(label_freq), 4) if label_freq else 0,
        'label_gini': round(compute_gini(label_freq), 4) if label_freq else 0,
    }

    # Add invalid reason counts
    for reason in EXPECTED_INVALID_REASONS:
        result[f'invalid_{reason}'] = invalid_reasons.get(reason, 0)

    # Add confidence bucket distribution
    total_conf = sum(confidence_buckets.values())
    for bucket in ['[0.0-0.2)', '[0.2-0.4)', '[0.4-0.6)', '[0.6-0.8)', '[0.8-1.0]']:
        percent = 100 * confidence_buckets[bucket] / total_conf if total_conf else 0
        result[f'conf_{bucket}'] = round(percent, 2)

    return result
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input .jsonl files.")
    parser.add_argument("--save", type=str, required=True, help="Name of csv file to save.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Provided input path {args.input_dir} is not a directory.")

    jsonl_files = [f for f in os.listdir(args.input_dir) if f.endswith('.jsonl')]
    if not jsonl_files:
        print("No .jsonl files found in the input directory.")
        exit()

    taxonomy_options = io.read_json("src/scripts/taxonomy_options.json")

    all_results = []

    for filename in sorted(jsonl_files):
        filepath = os.path.join(args.input_dir, filename)
        basename = filename[:-6]
        parts = basename.split('_', 1)
        if len(parts) != 2:
            print(f"Skipping file with unexpected format: {filename}")
            continue

        level_id, prompt_id = parts
        try:
            task_options = extract_options(taxonomy_options, level_id, prompt_id)
            class TempArgs:
                def __init__(self, input):
                    self.input = input

            temp_args = TempArgs(filepath)
            result = main(temp_args, task_options, level_id, prompt_id)
            all_results.append(result)

        except Exception as e:
            print(f"Error processing {filename}: {e}")


    output_csv = os.path.join(args.save)
    if all_results:
        dynamic_keys = set()
        for result in all_results:
            for key in result:
                if key.startswith('invalid_') or key.startswith('conf_'):
                    dynamic_keys.add(key)

        preferred_order = [
            'level_id',
            'prompt_id',
            'total_entries',
        ]
        invalid_keys = [f'invalid_{reason}' for reason in EXPECTED_INVALID_REASONS]
        confidence_keys = ['conf_[0.0-0.2)', 'conf_[0.2-0.4)', 'conf_[0.4-0.6)', 'conf_[0.6-0.8)', 'conf_[0.8-1.0]']
        rest_keys = [
            'avg_preds_per_row',
            'unique_labels',
            'label_entropy',
            'label_gini'
        ]

        fieldnames = preferred_order + invalid_keys + confidence_keys + rest_keys

        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_results)
