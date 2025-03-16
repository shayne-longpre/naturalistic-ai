import sys
import typing

sys.path.append("./")

from src.classes.message import Message
from src.classes.annotation_record import AnnotationRecord

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
        languages=None, # TODO: Remove.
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
                    metadata={k: AnnotationRecord(
                        value=vs["value"],
                        target=vs["target"],
                        annotator=vs.get("annotator")
                    ) for k, vs in m.get("metadata", {}).items()},
                ) 
                for m in conversation
            ]
        else:
            assert isinstance(self.convseration[0], Message)
            self.conversation = conversation
        self.geography = geography
        self.metadata = metadata

    def to_dict(self):
        obj = {
            "conversation_id": self.conversation_id,
            "dataset_id": self.dataset_id,
            "user_id": self.user_id,
            "time": self.time,
            "model": self.model,
            "geography": self.geography,
            "metadata": {k: v.to_dict() for k, v self.metadata.items()}
        }
        obj["conversation"] = [m.to_dict() for m in self.conversation]
        return obj