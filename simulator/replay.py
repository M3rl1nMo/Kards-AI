"""Portable action/state replay archive for deterministic simulation debugging."""
import json
from dataclasses import dataclass, field


@dataclass
class Replay:
    initial_state: dict
    entries: list[dict] = field(default_factory=list)
    def record(self, action, state):
        self.entries.append({"action": {"type": type(action).__name__, **action.__dict__}, "state": state.to_dict()})
    def to_json(self): return json.dumps({"initial_state": self.initial_state, "entries": self.entries}, ensure_ascii=False)
    @classmethod
    def from_json(cls, value):
        data=json.loads(value); return cls(data["initial_state"], data["entries"])
