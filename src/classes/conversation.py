import sys
import typing

sys.path.append("./")

from src.classes.message import Message

class Conversation(object):
    """A conversation object, with all metadata."""

    def __init__(
        self,
        conversation_id,
        dataset_id,
        user_id,
        time,
        model,
        conversation,
        geography=None,
        metadata={},
    ):
        assert "-" not in conversation_id, \
            f"Please do not include '-' in the conversation_id, as we use this character for joining/splitting keys."
        self.conversation_id = conversation_id
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.time = time
        self.model = model

        if isinstance(conversation[0], dict):
            self.conversation = [
                Message(
                    turn=m["turn"], 
                    role=m["role"], 
                    content=m["content"], 
                    timestamp=m.get("timestamp"), 
                    metadata=m.get("metadata", {}),
                ) 
                for m in conversation
            ]
        else:
            assert isinstance(self.convseration[0], Message)
            self.conversation = conversation
        self.geography = geography
        self.metadata = metadata

    def to_dict(self, unpack_conversation=False):
        obj = {
            "conversation_id": self.conversation_id,
            "dataset_id": self.dataset_id,
            "user_id": self.user_id,
            "time": self.time,
            "model": self.model,
            "geography": self.geography,
            "metadata": self.metadata
        }
        obj["conversation"] = [m.to_dict() for m in self.conversation]
        return obj