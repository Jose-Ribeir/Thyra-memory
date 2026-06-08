"""DeltaEvent — the post-turn learning event written by stop_hook.py."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeltaEvent:
    session_id: str
    turn_id: str
    user_id: str
    agent_id: str
    timestamp: int
    memories_served: list[str] = field(default_factory=list)
    memories_declared: list[str] = field(default_factory=list)
    cues_fired: list[str] = field(default_factory=list)
    raw_user_text: str = ""
    raw_assistant_text: str = ""
    correction_flag: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "DeltaEvent":
        return cls(
            session_id=data.get("session_id", ""),
            turn_id=data.get("turn_id", ""),
            user_id=data.get("user_id", "default"),
            agent_id=data.get("agent_id", "claude-code-global"),
            timestamp=int(data.get("timestamp", 0)),
            memories_served=data.get("memories_served", []),
            memories_declared=data.get("memories_declared", []),
            cues_fired=data.get("cues_fired", []),
            raw_user_text=data.get("raw_user_text", ""),
            raw_assistant_text=data.get("raw_assistant_text", ""),
            correction_flag=bool(data.get("correction_flag", False)),
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "memories_served": self.memories_served,
            "memories_declared": self.memories_declared,
            "cues_fired": self.cues_fired,
            "raw_user_text": self.raw_user_text,
            "raw_assistant_text": self.raw_assistant_text,
            "correction_flag": self.correction_flag,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
