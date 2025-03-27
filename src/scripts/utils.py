import sys
import json
import pandas as pd
from helpers import io

sys.path.append("./")
sys.path.append("src/")


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


def load_existing_ex_ids(save_fpath):
    try:
        existing_data = io.read_jsonl(save_fpath)
        return {entry["ex_id"] for entry in existing_data}
    except FileNotFoundError:
        return set()


def valid_turn(text):
    return pd.notna(text) and text.strip() != ""