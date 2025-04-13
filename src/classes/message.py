import sys
import typing


class Message(object):
    """A message object, with all metadata."""

    def __init__(
        self,
        turn,
        role,
        content,
        timestamp=None,
        metadata={},
    ):
        self.turn = turn
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.metadata = metadata

    def to_dict(self):
        return {
            "turn": self.turn,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": {k: v.to_dict() for k, v in self.metadata.items()}
        }

    def get_attr(
        self,
        source,
        attribute_name,
    ):
        metadata_key = f"{source}-{attribute_name}"
        return self.metadata.get(metadata_key).value if metadata_key in self.metadata else None

