"""Utililities for scripts."""

import json
import os

import pandas as pd

from src.helpers import io


def char_count(s):
    return len(s)


def read_json(json_file):
    # create file if not exists
    if not os.path.exists(json_file):
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False)
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def batch_generator(data_list, batch_size):
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]


def append_jsonl(data, file_path):
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")


def load_existing_conversation_ids(save_fpath):
    try:
        existing_data = io.read_jsonl(save_fpath)
        return {entry["conversation_id"] for entry in existing_data}
    except FileNotFoundError:
        return set()


def load_existing_exid_turn_pairs(filepath):
    if not os.path.exists(filepath):
        return set()

    existing_pairs = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                pair = (data.get('conversation_id'), data.get('turn'))
                existing_pairs.add(pair)
    return existing_pairs


def valid_turn(text):
    return pd.notna(text) and text.strip() != ""
