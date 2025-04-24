import sys
import typing


class AnnotationRecord(object):
    """An annotation record."""

    def __init__(
        self,
        value,
        confidence,
        target_id: str, # conversation_id or conversation_id-turn_id
        annotator: str=None,
    ):
        self.value = value
        self.confidence = confidence
        self.target_id = target_id
        self.annotator = annotator

    def to_dict(self):
        return {
            "value": self.value,
            "confidence": self.confidence,
            "target_id": self.target_id,
            "annotator": self.annotator,
        }

    def __str__(self):
        return str(self.to_dict())