import sys
import typing


class AnnotationRecord(object):
    """An annotation record."""

    def __init__(
        self,
        value,
        target_id: str, # conversation_id or conversation_id-turn_id
    ):
        self.value = value
        self.target_id = target_id

    def to_dict(self):
        return {
            "value": self.value,
            "target_id": self.target_id,
        }