"""Load benchmarks."""

from typing import Literal

from src.classes.dataset import Dataset

DATASET_LOCATIONS = {
    ### User Datasets ###
    "wildchat_v1": "datasets/wildchat_v1/full.json",
    "wildchat_private": "dataset_downloads/wildchat_1m_full",
    "lmsys_1m": "datasets/lmsys_1m/full.json",
    "sharegpt_v1": "datasets/sharegpt_v1/full.json",

    ### Benchmarks ###
    "chatbot_arena": "datasets/chatbot_arena/full.json",
    "alpaca_eval": "datasets/alpaca_eval/full.json",
    "mmlu": "datasets/mmlu/full.json",
    "hle": "datasets/hle/full.json",
    "gpqa": "datasets/gpqa/full.json",
    "swebench": "datasets/swebench/full.json",
}
DATASET_CATEGORIES = {
    ### User Datasets ###
    "wildchat_v1": "conversation",
    "wildchat_private": "conversation",
    "lmsys_1m": "conversation",
    "sharegpt_v1": "conversation",

    ### Benchmarks ###
    "chatbot_arena": "benchmark",
    "alpaca_eval": "benchmark",
    "mmlu": "benchmark",
    "hle": "benchmark",
    "gpqa": "benchmark",
    "swebench": "benchmark",

}
DATASET_SIZES = {
    ### User Datasets ###
    "wildchat_v1": 1000000,
    "wildchat_private": 1000000,
    "lmsys_1m": 1000000,
    "sharegpt_v1": 90665,

    ### Benchmarks ###
    "chatbot_arena": 66000,
    "alpaca_eval": 805,
    "mmlu": 14042,
    "hle": 2700,
    "gpqa": 546,
    "swebench": 2294,
}


def filter_by_id(ids: list[str]):
    valid_ids = []
    for id in ids:
        if id in DATASET_LOCATIONS.keys():
            valid_ids.append(id)
        else:
            print(f"\n\n**** WARNING: {id} was not found as a valid dataset id. Available datasets include: {DATASET_LOCATIONS.keys()}")
    return valid_ids


def filter_by_categories(categories: list[str]):
    valid_ids = []
    for id, category in DATASET_CATEGORIES.items():
        if category in categories:
            valid_ids.append(id)
    if len(valid_ids) == 0:
        print(f"\n\n**** WARNING: no datasets were found with the provided categories. Available categories include: {list(set(DATASET_LOCATIONS.values()))}")
    return valid_ids


def filter_by_size(size_range: list[int]):
    valid_ids = []
    for id, size in DATASET_SIZES.items():
        if size >= size_range[1] and size <= size_range[0]:
            valid_ids.append(id)
    if len(valid_ids) == 0:
        dataset_sizes_str = ', '.join([f'\n{key}: {value}' for key, value in DATASET_SIZES.items()])
        print(f"\n\n**** WARNING: no datasets were found within the provided size range of {size_range}. The sizes of available datasets are: {dataset_sizes_str}")

    return valid_ids


def load_datasets(by: Literal["id", "category", "size"], ids: list[str] = [], categories: list[str] = [], size_range: list[int] = [], path_to_dataset_downloads: str = ""):
    """
    This function is used to get any dataset.
    At the moment, selection can only be made by 1 feature at a time. For more specific selection, provide a list of the dataset_ids.
    Args:
        -by: how to select the datasets to return. Options are "id", "category", or "size".
        -ids: if selecting by dataset_ids, the dataset_ids to search for.
        -categories: if selecting by categories, the categories to search for.
        -size_range: if selecting by dataset size, the range of sample sizes to filter datasets by.
        -path_to_dataset_downloads: not required, use to set the alternative path to the general datasets folder. Expected format is <path_to_dataset_downloads>/<dataset_id>/{file.ext}.

    Return:
        - datasets: list[Dataset]. list of dataset objects.
    """
    matching_dataset_ids = []
    # Find the dataset ids that match the specifications
    if by == "id":
        if len(ids) == 0:
            raise ValueError("You've specified loading datasets by ID, but have not provided any dataset ids. Try something like: load_datasets(by='id', ids=['mmlu'])")
        matching_dataset_ids = filter_by_id(ids)

    elif by == "category":
        if len(categories) == 0:
            raise ValueError("You've specified loading datasets by categories, but have not provided any categories. Try something like: load_datasets(by='category', categories=['benchmark'])")
        matching_dataset_ids = filter_by_categories(categories)

    elif by == "size":
        if len(size_range) < 2:
            raise ValueError("You've specified loading datasets by size, but have not provided a valid size_range. Try something like: load_datasets(by='size', size_range=[0,10000])")
        matching_dataset_ids = filter_by_size(size_range)

    # Confirm at least one match is found
    if len(matching_dataset_ids) == 0:
        raise ValueError(f"No datasets meet the specifications given. The available datasets include {DATASET_LOCATIONS.keys()}")

    print(f"Found {len(matching_dataset_ids)} datasets that meet the desired specifications: {matching_dataset_ids}. Loading them from data files...")

    # Load the dataset objects and return
    datasets = []
    for dataset_id in matching_dataset_ids: 
        dataset = Dataset(dataset_id=dataset_id, data = [])
        dataset_with_data = dataset.load(json_path =DATASET_LOCATIONS[dataset_id])
        datasets.append(dataset_with_data)

    return datasets
