import json
import random
import sys
import argparse
from collections import defaultdict
from datetime import datetime
from dateutil.parser import parse as parse_date
from typing import List

sys.path.append("./")

from src.classes.conversation import Conversation


def get_month_bin(dt: datetime) -> str:
    return dt.strftime('%Y-%m')


def sample_evenly_by_time(data: List[Conversation], total_sample_size: int, seed: int = 42) -> List[Conversation]:
    random.seed(seed)
    time_bins = defaultdict(list)

    for item in data:
        try:
            dt = parse_date(item.time) if isinstance(item.time, str) else item.time
            if isinstance(dt, datetime):
                bin_key = get_month_bin(dt)
                time_bins[bin_key].append(item)
        except Exception:
            continue

    bins = list(time_bins.values())
    num_bins = len(bins)
    if num_bins == 0:
        raise ValueError("No valid time bins found in the data.")

    per_bin = total_sample_size // num_bins
    remainder = total_sample_size % num_bins

    sampled = []
    for i, bin_data in enumerate(bins):
        n = per_bin + (1 if i < remainder else 0)
        sampled.extend(random.sample(bin_data, min(n, len(bin_data))))

    return sampled


def main(input_path, output_path, sample_size, seed):
    with open(input_path, 'r') as f:
        dataset = json.load(f)

    raw_data = dataset['data']
    conversations = [Conversation(**c) for c in raw_data]  # Convert dicts to Conversation

    sampled = sample_evenly_by_time(conversations, total_sample_size=sample_size, seed=seed)
    output_dict = {
        'dataset_id': dataset['dataset_id'],
        'data': [x.to_dict() for x in sampled]
    }
    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=2)
    print(f"Saved {len(sampled)} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample dataset evenly across time bins.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to input JSON dataset.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save sampled JSON dataset.')
    parser.add_argument('--sample_size', type=int, default=4000, help='Total number of samples to select.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()
    main(args.input_path, args.output_path, args.sample_size, args.seed)
