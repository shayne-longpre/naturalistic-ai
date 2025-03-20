import os
import sys
import typing
import json
import re
from collections import defaultdict

sys.path.append("./")

from src.classes.annotation_record import AnnotationRecord
from src.helpers import io



class AnnotationSet(object):
    """A set of AnnotationRecords and their metadata."""

    def __init__(
        self,
        source: str,  # identify the annotation set uniquely: model / human / date
        name: str, # the annotation task: "language_id" / "conversation_purpose" / etc
        level: str, # "conversation" or "message"
        dataset_id: str,
        annotations: typing.List[AnnotationRecord],
    ):
        assert "-" not in source and "-" not in name, \
            f"Please do not include '-' in source or name, as we use this character for joining/splitting keys."
        assert level in ["conversation", "message"]
        self.source = source
        self.name = name
        self.level = level
        self.dataset_id = dataset_id
        self.annotations = annotations

    @classmethod
    def load(cls, json_path: str):
        """Alternative constructor that initializes from a JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Create a new instance with the remaining data
        return cls(
            source=data["source"], 
            name=data["name"], 
            level=data["level"],
            dataset_id=data["dataset_id"],
            annotations=[
                AnnotationRecord(
                    value=x["value"], 
                    target_id=x["target_id"],
                    annotator=x.get("annotator"),
                ) 
                for x in data["annotations"]],
        )


    @classmethod
    def load_labelstudio(cls, json_path: str, source: str):
        """Alternative constructor that initializes from LabelStudio file(s)."""
        # Parse all the Label Studio annotations together
        annotations = parse_labelstudio_files(json_path)
        
        # Create a new instance with the remaining data
        return cls(
            source=source, 
            name=data["name"], 
            level=data["level"],
            dataset_id=data["dataset_id"],
            annotations=[
                AnnotationRecord(
                    value=x["value"], 
                    target_id=x["target_id"],
                    annotator=x.get("annotator"),
                ) 
                for x in data["annotations"]],
        )

    @classmethod
    def load_automatic(cls, path: str):
        """Alternative constructor that initializes from AIArchive automatic annotation file(s)."""
        annotations = []

        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                record = json.loads(line.strip())
                instance = cls(
                    source=path, 
                    name=record["model"], 
                    level=record["system_level_id"],
                    dataset_id=record["dataset_id"],
                    annotations=AnnotationRecord(
                        value=record["response"], 
                        target_id=record["system_prompt_id"],
                        annotator=record["model_id"],
                    ),
                )
                annotations.append(instance)
        return annotations

    def to_dict(self):
        return {
            "source": self.source,
            "name": self.name,
            "level": self.level,
            "dataset_id": self.dataset_id,
            "annotations": [x.to_dict() for x in self.annotations],
        }

    def save_to_json(self, json_path: str):
        """Save the current state to a JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def remap(self, annotation_map: typing.Dict[str, str]):
        """Remap annotations using the provided dictionary."""
        remapped_annotations = []
        for x in self.annotations:
            new_value = annotation_map.get(x.value, x.value)
            remapped_annotations.append(AnnotationRecord(value=new_value, target_id=x.target_id))
        self.annotations = remapped_annotations

        

def process_annotations_to_annotation_sets(annotations_list, source: str):
    task_groups = {}
    
    for annotation in annotations_list:
        if "annotation_tasks" not in annotation:
            continue
            
        conv_id = annotation.get("conversation_id", "") or annotation.get("text_in_conversation_turn", {}).get("conversation_id", "")
        turn_idx = annotation.get("text_in_conversation_turn", {}).get("turn", 0)
        annotator = annotation.get("annotator_name", "")
        
        for task_category, task_values in annotation["annotation_tasks"].items():
            if task_category not in task_groups:
                task_groups[task_category] = []

            task_groups[task_category].append({
                "annotation_value": task_values,
                "conversation_id": conv_id,
                "turn_idx": turn_idx,
                "annotator_name": annotator
            })

    return {
        task_name: AnnotationSet(
            source=source,
            name=task_name,
            level="message",
            dataset_id="sample120",
            annotations=[
                AnnotationRecord(
                    value=x["annotation_value"],
                    target_id=f"{x['conversation_id']}-{x['turn_idx']}",
                    annotator=x.get("annotator_name")
                ) for x in data
            ]
        ) for task_name, data in task_groups.items()
    }