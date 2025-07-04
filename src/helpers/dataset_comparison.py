import itertools
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Tuple, Union
from src.classes.dataset import Dataset
from src.classes.conversation import Conversation
from scipy.stats import chi2_contingency, wasserstein_distance
from scipy.spatial.distance import jensenshannon
import numpy as np


def split_dataset_by(
    dataset: Dataset,
    group_fn: Callable[[Union[Conversation, object]], str],
    level: str = "conversation"
) -> Dict[str, Dataset]:
    """
    Split a Dataset into multiple Datasets based on a grouping function.

    Args:
        dataset (Dataset): The dataset to split.
        group_fn (Callable): Function that takes a Conversation or Message
                             and returns a group name (str).
        level (str): "conversation" (default) or "message".
                     Defines at which level to group.

    Returns:
        dict[str, Dataset]: A dict of group_name → Dataset,
                            where each Dataset only contains data for that group.
                            (For message-level: conversations only include grouped messages.)
    """
    groups = defaultdict(list)

    if level == "conversation":
        for conv in dataset.data:
            group = group_fn(conv)
            groups[group].append(conv)

    elif level == "message":
        for conv in dataset.data:
            msg_groups = defaultdict(list)
            for msg in conv.conversation:
                group = group_fn(msg)
                msg_groups[group].append(msg)

            for g, msgs in msg_groups.items():
                # Create a shallow copy of the conversation with only the grouped messages
                conv_copy = Conversation(
                    conversation_id=conv.conversation_id,
                    dataset_id=conv.dataset_id,
                    user_id=conv.user_id,
                    time=conv.time,
                    model=conv.model,
                    geography=conv.geography,
                    metadata=conv.metadata.copy(),
                    conversation=msgs
                )
                groups[g].append(conv_copy)
    else:
        raise ValueError(f"Unsupported level: {level}")

    return {
        g: Dataset(dataset_id=f"{dataset.dataset_id}_{g}", data=convs)
        for g, convs in groups.items()
    }


def compare_annotations_to_baseline(
    group_datasets: Dict[str, Dataset],
    baseline_dataset: Dataset,
    annotation_source_tasks: List[Tuple[str, str]]
) -> Dict[str, Dict[Tuple[str, str], Dict]]:
    """
    Compare annotation distributions of group datasets to a baseline,
    returning both differences and statistical measures.

    Args:
        group_datasets (dict[str, Dataset]): Group name → Dataset.
        baseline_dataset (Dataset): Baseline dataset.
        annotation_source_tasks (list[(attribute_name, source)]): 
            e.g. [("prompt_function_purpose", "automatic_v0"), ...]

    Returns:
        dict[group_name, dict[(attribute_name, source), dict]]:
            group_name → {
                (attribute_name, source) → {
                    "differences": [(label, diff, group_pct, base_pct), …],
                    "metrics": {"chi2": …, "jsd": …, "wasserstein": …}
                }
            }
    """
    def get_label_counts(dset: Dataset, attr: str, source: str) -> Counter:
        counter = Counter()
        for conv in dset.data:
            for msg in conv.conversation:
                val = msg.get_attr(source, attr)
                if val and val != "None" and str(val).strip():
                    if isinstance(val, list):
                        counter.update(val)
                    else:
                        counter[str(val).strip()] += 1
        return counter

    baseline_distributions = {
        (attr, src): get_label_counts(baseline_dataset, attr, src)
        for (attr, src) in annotation_source_tasks
    }

    comparison_results = {}

    for group_name, group_dataset in group_datasets.items():
        group_results = {}

        for (attr, src) in annotation_source_tasks:
            key = (attr, src)
            baseline = baseline_distributions[key]
            group = get_label_counts(group_dataset, attr, src)

            total_base = sum(baseline.values())
            total_group = sum(group.values())

            all_labels = sorted(set(baseline.keys()) | set(group.keys()))

            base_counts = np.array([baseline[l] for l in all_labels])
            group_counts = np.array([group[l] for l in all_labels])

            base_pcts = np.array([(c / total_base) * 100 if total_base else 0 for c in base_counts])
            group_pcts = np.array([(c / total_group) * 100 if total_group else 0 for c in group_counts])
            differences = [
                (label, round(gp - bp, 2), gp, bp)
                for label, gp, bp in zip(all_labels, group_pcts, base_pcts)
            ]
            differences.sort(key=lambda x: abs(x[1]), reverse=True)

            # === compute statistical measures ===
            metrics = {}
            if base_counts.sum() and group_counts.sum() and len(all_labels) > 1:
                try:
                    contingency = np.array([group_counts, base_counts])
                    chi2, p, _, _ = chi2_contingency(contingency, correction=False)
                    metrics['chi2'] = chi2
                    metrics['p_value'] = p
                except Exception:
                    metrics['chi2'] = None
                    metrics['p_value'] = None

                try:
                    jsd = jensenshannon(group_pcts + 1e-12, base_pcts + 1e-12)
                    metrics['jsd'] = float(jsd)
                except Exception:
                    metrics['jsd'] = None

                try:
                    w_dist = wasserstein_distance(group_pcts, base_pcts)
                    metrics['wasserstein'] = float(w_dist)
                except Exception:
                    metrics['wasserstein'] = None
            else:
                metrics['chi2'] = None
                metrics['p_value'] = None
                metrics['jsd'] = None
                metrics['wasserstein'] = None

            group_results[key] = {
                "differences": differences,
                "metrics": metrics
            }

        comparison_results[group_name] = group_results

    return comparison_results


