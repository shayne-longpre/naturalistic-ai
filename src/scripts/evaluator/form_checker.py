import json
import re
import os
import csv
import math
import argparse
# from taxonomy import OPTIONS
from collections import Counter

sys.path.append("./")

from src.classes import automatic_annotation_parser
from src.helpers import io
from src.scripts.evaluator.taxonomy import OPTIONS


def compute_entropy(counts: Counter) -> float:
    """Calculate entropy given label frequency counts."""
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def compute_gini(counts: Counter) -> float:
    """Calculate Gini impurity given label frequency counts."""
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return 1 - sum(p ** 2 for p in probs)


def bucket_confidence(conf):
    if conf < 0.2: return '[0.0-0.2)'
    elif conf < 0.4: return '[0.2-0.4)'
    elif conf < 0.6: return '[0.4-0.6)'
    elif conf < 0.8: return '[0.6-0.8)'
    else: return '[0.8-1.0]'


def analyze_entries(entries: List[Dict[str, Any]], options: List[str]) -> Dict[str, Any]:
    """Generate statistics about validation outcomes and label distribution."""
    invalid_reasons = Counter()
    confidence_buckets = Counter()
    label_counts = Counter()
    predictions_per_row = []

    for entry in entries:
        is_valid, reason, response_data = automatic_annotation_parser.validate_entry(entry, options)
        if not is_valid:
            invalid_reasons[reason] += 1
            continue

        predictions_per_row.append(len(response_data))

        for item in response_data:
            confidence_buckets[bucket_confidence(item['confidence'])] += 1
            labels = item['labels']
            labels = labels if isinstance(labels, list) else [labels]
            label_counts.update(labels)

    total_confidences = sum(confidence_buckets.values())
    confidence_distribution = {bucket: round(100 * confidence_buckets[bucket] / total_confidences, 2)
                               for bucket in confidence_buckets}

    return {
        'total_entries': len(entries),
        'invalid_reasons': dict(invalid_reasons),
        'avg_predictions_per_entry': round(sum(predictions_per_row) / len(predictions_per_row), 2)
        if predictions_per_row else 0,
        'unique_labels': len(label_counts),
        'label_entropy': round(compute_entropy(label_counts), 4),
        'label_gini': round(compute_gini(label_counts), 4),
        'confidence_distribution': confidence_distribution
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and analyze annotation outputs.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input .jsonl files.")
    parser.add_argument("--options_file", default="src/scripts/evaluator/taxonomy.py", help="JSON file containing task options.")
    parser.add_argument("--output", required=True, help="Path to save analysis results as CSV.")

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Provided input path {args.input_dir} is not a directory.")

    all_results = []
    for filepath in io.listdir_nohidden(args.input_dir):
        entries = io.read_jsonl(filepath)
        level_id = annotations[0]["level_id"]
        prompt_id = annotations[0]["prompt_id"]
        task_options = extract_options(OPTIONS, level_id, prompt_id)

        analysis_result = analyze_entries(entries, task_options)
        analysis_result.update({"level_id": level_id, "prompt_id": prompt_id})
        all_results.append(analysis_result)

    if all_results:
        fieldnames = ["level_id", "prompt_id", "total_entries", "avg_predictions_per_entry",
                      "unique_labels", "label_entropy", "label_gini", "invalid_reasons",
                      "confidence_distribution"]

        with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in all_results:
                result["invalid_reasons"] = json.dumps(result["invalid_reasons"])
                result["confidence_distribution"] = json.dumps(result["confidence_distribution"])
                writer.writerow(result)

    print(f"Analysis results saved to {args.output}")