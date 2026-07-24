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
    def __init__(self, capacity: int = 100_000, seed: int = 0) -> None:
        self.capacity, self.examples, self.rng = capacity, [], random.Random(seed)
    def add(self, example: TrainingExample) -> None:
        if len(self.examples) >= self.capacity: self.examples.pop(0)
        self.examples.append(example)
    def sample(self, batch_size: int) -> list[TrainingExample]:
        return self.rng.sample(self.examples, min(batch_size, len(self.examples)))
    def __len__(self) -> int: return len(self.examples)
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle: pickle.dump({"capacity": self.capacity, "examples": self.examples}, handle)
    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        with open(path, "rb") as handle: data = pickle.load(handle)
        result = cls(data["capacity"]); result.examples = data["examples"]; return result
