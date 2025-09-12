"""Form checker."""

import argparse
import csv
import math
from collections import Counter
from typing import Any, Dict, List

from src.classes import automatic_annotation_parser
from src.helpers import io

# === Utility functions ===


def compute_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def compute_gini(counts: Counter) -> float:
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return 1 - sum(p ** 2 for p in probs)


def bucket_confidence(conf: float) -> str:
    if conf < 0.2:
        return '[0.0-0.2)'
    elif conf < 0.4:
        return '[0.2-0.4)'
    elif conf < 0.6:
        return '[0.4-0.6)'
    elif conf < 0.8:
        return '[0.6-0.8)'
    else:
        return '[0.8-1.0]'


# === Main analysis logic ===

def analyze_parsed_entries(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    confidence_buckets = Counter()
    label_counts = Counter()
    predictions_per_row = []

    for entry in entries:
        response_data = entry.get("parsed_response", [])
        if not response_data:
            continue

        predictions_per_row.append(len(response_data))

        # Optional: Count original confidence distribution if raw items kept
        raw_items = entry.get("response_data", [])
        for item in raw_items:
            confidence_buckets[bucket_confidence(item["confidence"])] += 1

        label_counts.update(response_data)

    total_conf = sum(confidence_buckets.values())
    conf_dist = {bucket: round(100 * confidence_buckets[bucket] / total_conf, 2) for bucket in confidence_buckets}

    return {
        'total_entries': len(entries),
        'avg_predictions_per_entry': round(sum(predictions_per_row) / len(predictions_per_row), 2) if predictions_per_row else 0,
        'unique_labels': len(label_counts),
        'label_entropy': round(compute_entropy(label_counts), 4),
        'label_gini': round(compute_gini(label_counts), 4),
        'confidence_distribution': conf_dist
    }


# === Script entry point ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Directory with input .jsonl files")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--conf_threshold", type=float,
                        default=0.3, help="Minimum confidence to accept labels")
    args = parser.parse_args()

    all_results = []

    for filepath in io.listdir_nohidden(args.input_dir):
        entries = automatic_annotation_parser.parse_automatic_annotations(
            filepath, conf_threshold=args.conf_threshold, verbose=True,
        )
        if not entries:
            continue

        level_id = entries[0].get("level_id")
        prompt_id = entries[0].get("prompt_id")

        result = analyze_parsed_entries(entries)
        result.update({"level_id": level_id, "prompt_id": prompt_id})
        all_results.append(result)

    if all_results:
        fieldnames = ["level_id", "prompt_id", "total_entries", "avg_predictions_per_entry",
                      "unique_labels", "label_entropy", "label_gini"]
        fieldnames += [f'conf_{b}' for b in ['[0.0-0.2)',
                                             '[0.2-0.4)', '[0.4-0.6)', '[0.6-0.8)', '[0.8-1.0]']]

        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                for bucket in ['[0.0-0.2)', '[0.2-0.4)', '[0.4-0.6)', '[0.6-0.8)', '[0.8-1.0]']:
                    r[f'conf_{bucket}'] = r.get('confidence_distribution', {}).get(bucket, 0)
                r.pop('confidence_distribution', None)  # <- REMOVE unused field
                writer.writerow(r)
