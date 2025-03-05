import sys
import random
import typing
import json
from torch.utils.data import DataLoader
import pandas as pd 
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

    def add_annotations(
        self,
        annotation_set: AnnotationSet,
    ):
        """Update the dataset `metadata` fields in conversation or messages directly.
        This is used to maintain any type of metadata alongside the dataset.
        """
        # Validation
        assert annotation_set.dataset_id == self.dataset_id
        annotation_conv_ids = [x.target_id.split("-")[0] for x in annotation_set.annotations]
        dset_conv_ids = [cc.conversation_id for cc in self.data]
        # print(dset_conv_ids)
        # print(annotation_conv_ids)
        assert set(annotation_conv_ids) <= set(dset_conv_ids)
        src, name, lvl = annotation_set.source, annotation_set.name, annotation_set.level
        print(f"Adding AnnotationSet '{src}' for label=`{name}` ({lvl}-level): {len(annotation_conv_ids)} of {len(dset_conv_ids)} dataset rows")

        conv_id_to_idx = {x.conversation_id: i for i, x in enumerate(self.data)}
        # add to conversations
        if annotation_set.level == "conversation":
            for annotation in annotation_set.annotations:
                self.data[conv_id_to_idx[annotation.target_id]].metadata.update({
                    f"{annotation_set.source}-{annotation_set.name}": annotation.value})
        # or add to messages
        elif annotation_set.level == "message":
            for annotation in annotation_set.annotations:
                conv_id, turn_id = annotation.target_id.split("-")
                self.data[conv_id_to_idx[conv_id]].conversation[turn_id].metadata.update({
                    f"{annotation_set.source}-{annotation_set.name}": annotation.value})
        else:
            raise Exception

            
    def get_annotation_distribution(
        self, 
        name: str,
        level: str = "conversation",
        annotation_source: typing.Optional[str] = None
    ) -> typing.Dict[str, int]:
        """
        Get the distribution of annotation values for a specific feature.
        
        Args:
            name: Name of the annotation feature to analyze
            level: 'conversation' or 'message'.
            annotation_source: Source of the annotation (e.g., 'human_labels', 'model_v1_labels')
                                If None and name is a built-in attribute, will use that directly
        
        Returns:
            Dictionary mapping each annotation value to its frequency count
        """
        distribution = {}
        
        # Check if we're looking for a built-in attribute (like 'model')
        if level == "conversation" and annotation_source is None:
            assert hasattr(self.data[0], name), f"Every conversation should have {name} attribute."
            for conv in self.data:
                value = getattr(conv, name)
                distribution[value] = distribution.get(value, 0) + 1
        elif level == "message"  and annotation_source is None:
            assert hasattr(self.data[0].conversation[0], name), f"Every message should have {name} attribute."
            for conv in self.data:
                for message in conv.conversation:
                    value = getattr(message, name)
                    distribution[value] = distribution.get(value, 0) + 1
        elif level == "conversation":
            for conv in self.data:
                if f"{annotation_source}-{name}" in conv.metadata:
                    value = conv.metadata[f"{annotation_source}-{name}"]
                    distribution[value] = distribution.get(value, 0) + 1
        else:
            for conv in self.data:
                for msg in conv.conversation:
                    if f"{annotation_source}-{name}" in msg.metadata:
                        value = msg.metadata[f"{annotation_source}-{name}"]
                        distribution[value] = distribution.get(value, 0) + 1
                    
        return distribution
    
    
    def get_joint_distribution(
        self, 
        annotations1: typing.Tuple[str, typing.Optional[str]], 
        annotations2: typing.Tuple[str, typing.Optional[str]],
        level: str = "conversation",
        compute_disagreement: bool = False,
    ):
        """
        Get the joint distribution of two annotation features.
        
        Args:
            annotations1: Tuple of (name, source) for the first annotation
            annotations2: Tuple of (name, source) for the second annotation
            compute_disagreement: Whether to compute disagreement metrics (only for same label sets)
            
        Returns:
            If compute_disagreement is False:
                DataFrame representing the confusion matrix
            If compute_disagreement is True:
                Tuple of (
                    DataFrame representing the confusion matrix,
                    Dictionary of agreement metrics,
                    List of dictionaries containing disagreement details
                )
        """
        name1, source1 = annotations1
        name2, source2 = annotations2
        
        # Format full annotation names
        full_name1 = f"{source1}-{name1}" if source1 else name1
        full_name2 = f"{source2}-{name2}" if source2 else name2
        
        # Prepare data structures for counting
        joint_counts = {}
        paired_values = []
        disagreement_rows = []
        total_items = 0
        value_exists1, value_exists2 = 0, 0
        
        # Process based on annotation levels
        if level == "conversation":
            for conv in self.data:
                total_items += 1
                
                # Handle built-in attributes
                val1 = getattr(conv, name1) if source1 in [None, "conversation"] else conv.metadata.get(full_name1)
                val2 = getattr(conv, name2) if source2 in [None, "conversation"] else conv.metadata.get(full_name2)
                value_exists1 += 1 if val1 is not None else 0
                value_exists2 += 1 if val2 is not None else 0
                
                if val1 is not None and val2 is not None:
                    joint_counts[(val1, val2)] = joint_counts.get((val1, val2), 0) + 1
                    paired_values.append((conv.conversation_id, val1, val2))
        
        else: # level == "message"
            for conv in self.data:
                for msg in conv.conversation:
                    total_items += 1
                    
                    val1 = msg.metadata.get(full_name1)
                    val2 = msg.metadata.get(full_name2)
                    value_exists1 += 1 if val1 is not None else 0
                    value_exists2 += 1 if val2 is not None else 0
                    
                    if val1 is not None and val2 is not None:
                        joint_counts[(val1, val2)] = joint_counts.get((val1, val2), 0) + 1
                        paired_values.append((f"{conv.conversation_id}-{msg.turn}", val1, val2))
        
        print(f"Found {value_exists1} for annotations1, and {value_exists2} for annotations2.")
        print(f"Found {len(paired_values)} items (at level={level}) with both annotations out of {total_items} total.")
        
        # Convert joint counts to DataFrame (confusion matrix)
        unique_vals1 = sorted(list(set([k[0] for k in joint_counts.keys()])))
        unique_vals2 = sorted(list(set([k[1] for k in joint_counts.keys()])))
        
        matrix = pd.DataFrame(0, index=unique_vals1, columns=unique_vals2)
        for (val1, val2), count in joint_counts.items():
            matrix.loc[val1, val2] = count
            
        # Compute agreement metrics if requested
        if compute_disagreement:
            metrics = self._compute_annotator_disagreement(paired_values)
            disagreement_rows = [(uid, val1, val2) for (uid, val1, val2) in paired_values if val1 != val2]
            return matrix, metrics, disagreement_rows
        else:
            return matrix

    def _compute_annotator_disagreement(
        self, paired_values
    ):
        metrics = {}

        # Overall agreement
        uids, vals1, vals2 = zip(*paired_values)
        metrics["agreement_rate"] = sum(v1 == v2 for uid, v1, v2 in paired_values) / len(paired_values)
        
        # Try to compute Cohen's Kappa if possible
        try:
            metrics["cohens_kappa"] = cohen_kappa_score(vals1, vals2)
        except Exception as e:
            metrics["cohens_kappa_error"] = str(e)
            
        # Try to compute F1 scores if the values are binary
        if set(vals1).issubset({0, 1}) and set(vals2).issubset({0, 1}):
            cm = confusion_matrix(vals1, vals2)
            tp = cm[1, 1]
            fp = cm[0, 1]
            fn = cm[1, 0]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1_score"] = f1
        
        return metrics

