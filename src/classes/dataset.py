import sys
import random
import typing
import json
import os
from torch.utils.data import DataLoader
import pandas as pd 
import itertools
from collections import defaultdict, Counter
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.append("./")

from src.classes.conversation import Conversation
from src.classes.annotation_set import AnnotationSet


"""
This dataset.py file is used to define the Dataset objects. 
A Dataset is used to define the collection of conversations or samples. 
To download a dataset, use the download_datasets.py file. 
To load a dataset, use the load_datasets.py file. 
"""

    

class Dataset(object): 
    """Dataset class used to define the common features / functions of any evaluation or usage dataset."""

    def __init__(
        self, 
        dataset_id: str, 
        data: typing.List[Conversation],
    ):
        self.dataset_id: str = dataset_id
        self.data: typing.List[Conversation] = data
        
    @classmethod
    def load(cls, json_path: str):
        """Alternative constructor that initializes from a JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Create a new instance with the remaining data
        return cls(
            dataset_id=data["dataset_id"],
            data=[Conversation(**x) for x in data["data"]],
        )

    def to_dict(self):
        return {
            "dataset_id": self.dataset_id,
            "data": [x.to_dict() for x in self.data],
        }

    def save_to_json(self, json_path: str):
        """Save the current state to a JSON file."""
        # os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def __len__(self):
        return len(self.data)

    def sample(self, n):
        """Sample n conversations from the dataset."""
        return random.sample(self.data, n)

    def slice(self, start, end):
        """Get a slice of the dataset from start to end."""
        return self.data[start:end]

    def id_lookup(self, ids, level="conversation"):
        if level == "conversation":
            return [cc.to_dict() for cc in self.data if cc.conversation_id in ids]
        else:
            return [m.to_dict() for cc in self.data for m in cc.conversation if f"{cc.conversation_id}-{m.turn}" in ids]

    # def apply_filter(
    #     self, 
    #     filter_fn,
    #     level: str = "conversation",
    # ):
    #     if level == "conversation":
    #         self.data = [conv for conv in self.data if filter_fn(conv)]
    #     else: # level == "message"
    #         self.data = [conv for conv in self.data if filter_fn(conv)]

    def add_annotations(
        self,
        annotation_set: AnnotationSet,
        verbose: bool=False,
    ):
        """Update the dataset `metadata` fields in conversation or messages directly.
        This is used to maintain any type of metadata alongside the dataset.
        """
        # Validation
        assert annotation_set.dataset_id == self.dataset_id
        annotation_conv_ids = [x.target_id.split("-")[0] for x in annotation_set.annotations]
        dset_conv_ids = [cc.conversation_id for cc in self.data]
        
        assert set(annotation_conv_ids) <= set(dset_conv_ids)
        src, name, lvl = annotation_set.source, annotation_set.name, annotation_set.level
        if lvl == "conversation":
            if verbose:
                print(f"Adding AnnotationSet '{src}' for label=`{name}` ({lvl}-level): {len(set(annotation_conv_ids))} of {len(set(dset_conv_ids))} dataset conversations.")
        else:
            dset_num_messages = sum([len(cc.conversation) for cc in self.data])
            annotated_messages = len(annotation_conv_ids)
            if verbose:
                print(f"Adding AnnotationSet '{src}' for label=`{name}` ({lvl}-level): {annotated_messages} of {dset_num_messages} dataset messages, or {len(set(annotation_conv_ids))} of {len(set(dset_conv_ids))} dataset conversations.")

        conv_id_to_idx = {x.conversation_id: i for i, x in enumerate(self.data)}
        # add to conversations
        if annotation_set.level == "conversation":
            for annotation in annotation_set.annotations:
                self.data[conv_id_to_idx[annotation.target_id]].metadata.update({
                    f"{annotation_set.source}-{annotation_set.level}-{annotation_set.name}": annotation})
        # or add to messages
        # elif annotation_set.level == "message":
        #     for annotation in annotation_set.annotations:
        #         conv_id, turn_id = annotation.target_id.split("-")
        #         self.data[conv_id_to_idx[conv_id]].conversation[int(turn_id)].metadata.update({
        #             f"{annotation_set.source}-{annotation_set.name}": annotation})
        elif annotation_set.level in {"message", "prompt", "response", "turn"}:
            for annotation in annotation_set.annotations:
                conv_id, turn_id = annotation.target_id.split("-")
                if conv_id not in conv_id_to_idx:
                    if verbose:
                        print(f"[WARN] conversation_id '{conv_id}' not found in dataset.")
                    continue

                conv_idx = conv_id_to_idx[conv_id]
                conversation = self.data[conv_idx].conversation
                turn_idx = int(turn_id)

                if turn_idx >= len(conversation):
                    if verbose:
                        print(f"[WARN] turn index {turn_idx} out of bounds for conversation '{conv_id}' with {len(conversation)} turns.")
                    continue

                conversation[turn_idx].metadata.update({
                    f"{annotation_set.source}-{annotation_set.level}-{annotation_set.name}": annotation
                })
        else:
            raise Exception

            
    def get_annotation_distribution(
        self, 
        name: str,
        level: str,
        annotation_source: typing.Optional[str] = None,
        annotation_as_list_type: bool = False,
    ) -> typing.Dict[str, int]:
        """
        Get the distribution of annotation values for a specific feature.
        
        Args:
            name: Name of the annotation feature to analyze
            level: 'conversation' or 'message'.
            annotation_source: Source of the annotation (e.g., 'human_labels', 'model_v1_labels')
                If None and name is a built-in attribute, will use that directly
            annotation_as_list_type: If the annotation type is list, turn this to True to
                tally the full list as the value, rather than each element (default).
        
        Returns:
            Dictionary mapping each annotation value to its frequency count
        """
        distribution = {}

        def update_value(distribution, value):
            if isinstance(value, list):
                if annotation_as_list_type:
                    list_val = str(sorted(value))
                    distribution[list_val] = distribution.get(list_val, 0) + 1
                else:
                    for val in value:
                        distribution[val] = distribution.get(val, 0) + 1
            else:
                distribution[value] = distribution.get(value, 0) + 1
            return distribution
        
        # Check if we're looking for a built-in attribute (like 'model')
        if annotation_source is None:
            assert hasattr(self.data[0], name), f"Every conversation should have {name} attribute."
            for conv in self.data:
                value = getattr(conv, name)
                distribution = update_value(distribution, value)
        else:
            for conv in self.data:
                for msg in conv.conversation:
                    if f"{annotation_source}-{level}-{name}" in msg.metadata:
                        value = msg.metadata[f"{annotation_source}-{level}-{name}"].value
                        distribution = update_value(distribution, value)
                    
        return distribution        


    def get_joint_distribution(
        self, 
        annotations1: typing.Tuple[str, typing.Optional[str]], 
        annotations2: typing.Tuple[str, typing.Optional[str]],
        level: str = "conversation",
        compute_disagreement: bool = False,
        verbose: bool = True,
    ):
        """
        Get the joint distribution of two annotation features, supporting list-valued features.
        
        Args:
            annotations1: Tuple of (name, source) for the first annotation
            annotations2: Tuple of (name, source) for the second annotation
            level: The level at which to compute distributions ("conversation" or "message")
            compute_disagreement: Whether to compute disagreement metrics (only for same label sets)
            
        Returns:
            If compute_disagreement is False:
                DataFrame representing the label-level confusion matrix
            If compute_disagreement is True:
                Tuple of (
                    DataFrame representing the label-level confusion matrix,
                    Dictionary of agreement metrics,
                    List of dictionaries containing disagreement details
                )
        """
        name1, source1 = annotations1
        name2, source2 = annotations2
        
        # Format full annotation names
        full_name1 = f"{source1}-{name1}" if source1 else name1
        full_name2 = f"{source2}-{name2}" if source2 else name2
        

        def _create_pairwise_combinations(item_id, val1, val2):
            """vCreate all pairwise combinations of labels between two potentially list-valued features."""
            list1 = val1 if isinstance(val1, list) else [val1]
            list2 = val2 if isinstance(val2, list) else [val2]
            return [(item_id, label1, label2) for label1, label2 in itertools.product(list1, list2)]


        def _create_confusion_matrix(annotation_pairs):
            """
            Create a confusion matrix DataFrame from joint counts.
            """
            joint_counts = defaultdict(int)
            for _, label1, label2 in annotation_pairs:
                joint_counts[(label1, label2)] += 1

            if not joint_counts:
                return pd.DataFrame()
                
            unique_vals1 = sorted(list(set([k[0] for k in joint_counts.keys()])))
            unique_vals2 = sorted(list(set([k[1] for k in joint_counts.keys()])))
            matrix = pd.DataFrame(0, index=unique_vals1, columns=unique_vals2)
            for (val1, val2), count in joint_counts.items():
                matrix.loc[val1, val2] = count
                
            return matrix

        def _compute_annotator_disagreement(annotation_pairs):
            """
            Compute agreement metrics for the annotation pairs.
            """
            if not annotation_pairs:
                return {"error": "No paired values found"}
                
            metrics = {}
            _, labels1, labels2 = zip(*annotation_pairs)
            metrics["label_agreement_rate"] = sum(l1 == l2 for _, l1, l2 in annotation_pairs) / len(annotation_pairs)
            
            try:
                metrics["cohens_kappa"] = cohen_kappa_score([str(x) for x in labels1], [str(x) for x in labels2])
            except Exception as e:
                metrics["cohens_kappa_error"] = str(e)
            return metrics

        
        def _get_annotation_pairs():
            # Get all annotation pairs
            combination_pairs, annotation_pairs = [], []
            total_items, value_exists1, value_exists2 = 0, 0, 0

            def _extract_annotation_pair(obj, obj_id):
                val1 = obj.get_attr(source1, name1)
                val2 = obj.get_attr(source2, name2)
                
                local_combination_pair, local_annotation_pair = None, None
                if val1 is not None and val2 is not None:
                    # Add all pairwise combinations for list values
                    local_combination_pair = _create_pairwise_combinations(
                        obj_id, val1, val2
                    )
                    local_annotation_pair = (
                        obj_id, 
                        sorted(val1) if isinstance(val1, list) else val1, 
                        sorted(val2) if isinstance(val2, list) else val2,
                    )

                return local_annotation_pair, local_combination_pair, int(val1 is not None), (val2 is not None)

            if level == "conversation":
                for conv in self.data:
                    annotation_pair, combination_pair, exists1, exists2 = _extract_annotation_pair(conv, conv.conversation_id)

                    if annotation_pair:
                        annotation_pairs.append(annotation_pair)
                    if combination_pair:
                        combination_pairs.extend(combination_pair)
                    total_items += 1
                    value_exists1 += exists1
                    value_exists2 += exists2

            else:  # level == "message"
                for conv in self.data:
                    for msg in conv.conversation:
                        annotation_pair, combination_pair, exists1, exists2 = _extract_annotation_pair(msg, f"{conv.conversation_id}-{msg.turn}")

                        if annotation_pair:
                            annotation_pairs.append(annotation_pair)
                        if combination_pair:
                            combination_pairs.extend(combination_pair)
                        total_items += 1
                        value_exists1 += exists1
                        value_exists2 += exists2

            if verbose:
                print(f"Found {value_exists1} items with `{source1}-{level}-{name1}`, and {value_exists2} with `{source2}-{level}-{name2}`.")
                print(f"Generated {len(annotation_pairs)} label-level pairs (at level={level}) out of {total_items} total items.")
            return annotation_pairs, combination_pairs

        annotation_pairs, combination_pairs = _get_annotation_pairs()

        # Create confusion matrix DataFrame
        matrix = _create_confusion_matrix(combination_pairs)
        
        # Compute agreement metrics if requested
        if compute_disagreement:
            metrics = _compute_annotator_disagreement(annotation_pairs)
            return matrix, metrics, annotation_pairs
        else:
            return matrix, annotation_pairs



