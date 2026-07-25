"""Serializable bounded replay data for policy/value learning."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import pickle, random
from typing import Any


@dataclass
class TrainingExample:
    state: list[float]
    action_features: list[list[float]]
    legal_mask: list[bool]
    policy: list[float]
    value: float


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000, seed: int = 0) -> None:
        self.capacity, self.examples, self.rng = capacity, [], random.Random(seed)
    def add(self, example: TrainingExample) -> None:
        self._compact(example)
        if len(self.examples) >= self.capacity: self.examples.pop(0)
        self.examples.append(example)
    def sample(self, batch_size: int) -> list[TrainingExample]:
        return self.rng.sample(self.examples, min(batch_size, len(self.examples)))
    def __len__(self) -> int: return len(self.examples)
    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump({"capacity": self.capacity, "examples": self.examples, "rng_state": self.rng.getstate()}, handle)
        temporary.replace(path)
    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        with open(path, "rb") as handle: data = pickle.load(handle)
        result = cls(data["capacity"])
        result.examples = data["examples"]
        if data.get("rng_state"):
            result.rng.setstate(data["rng_state"])
        for example in result.examples:
            result._compact(example)
        return result

    @staticmethod
    def _compact(example: TrainingExample) -> None:
        """Drop padded action slots retained by older replay files.

        A previous format stored 512 action rows for every decision even when
        only a few actions were legal.  Keeping only the meaningful prefix
        shrinks resident replay memory and pickle traffic dramatically.
        """
        if not example.legal_mask:
            return
        last_legal = max((index for index, legal in enumerate(example.legal_mask) if legal), default=-1) + 1
        if last_legal <= 0:
            raise ValueError("Training example has no legal actions")
        example.action_features = example.action_features[:last_legal]
        example.legal_mask = example.legal_mask[:last_legal]
        example.policy = example.policy[:last_legal]
