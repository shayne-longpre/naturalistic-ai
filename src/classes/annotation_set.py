import sys
import typing
import json

sys.path.append("./")

from src.classes.annotation_record import AnnotationRecord


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
            annotations=[AnnotationRecord(value=x["value"], target_id=x["target_id"]) for x in data["annotations"]],
        )

    @classmethod
    def load_labelstudio(cls, json_path: str):
        """Alternative constructor that initializes from LabelStudio file(s)."""
        pass

    @classmethod
    def load_automatic(cls, path: str):
        """Alternative constructor that initializes from automatic annotation file(s)."""
        pass

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

        