import json
import re
import math
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


def validate_entry(args, entry):
    response_data = extract_json_from_response(entry['response'])
    if not isinstance(response_data, list):
        return False, 'Invalid JSON list'

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


def main(args, task_options):
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
            is_valid, reason = validate_entry(args, entry)

            if not is_valid:
                invalid_reasons[reason] += 1
                continue

            response_data = extract_json_from_response(entry['response'])
            predictions_per_row.append(len(response_data))
            for item in response_data:
                confidence_buckets[bucket_confidence(item['confidence'])] += 1
                label = item['labels']
                all_labels.append(label)
                label_freq[label] += 1

    # Print invalid reason summary
    print(f"\nTotal entries: {total_count}")
    if invalid_reasons:
        print("\n======= Invalid Reason Summary =======")
        for reason, count in invalid_reasons.items():
            print(f'{reason}: {count}')
    else:
        print("All entries are correctly formatted.")

    # Confidence bucket stats
    if confidence_buckets:
        print("\n======= Confidence Bucket (%) =======")
        for bucket in ['[0.0-0.2)', '[0.2-0.4)', '[0.4-0.6)', '[0.6-0.8)', '[0.8-1.0]']:
            percent = 100 * confidence_buckets[bucket] / sum(confidence_buckets.values())
            print(f"{bucket}: {percent:.2f}%")

    # Multi-label stats
    if predictions_per_row:
        print("\n======= Multilabel Stats =======")
        avg_preds = sum(predictions_per_row) / len(predictions_per_row)
        print(f"Average number of predictions per row: {avg_preds:.2f}")
        print(f"Unique labels predicted: {len(set(all_labels))} / {len(task_options)}")
        print(f"Label entropy: {compute_entropy(label_freq):.4f}")
        print(f"Label Gini index: {compute_gini(label_freq):.4f}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input jsonl file.")
    parser.add_argument("--level_id", required=True, default=None)
    parser.add_argument("--prompt_id", required=True, default=None)
    args = parser.parse_args()

    task_options = extract_options(args.level_id, args.prompt_id)
    main(args, task_options)