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
    def __init__(self, capacity: int = 50_000, seed: int = 0, recent_fraction: float = 0.6) -> None:
        self.capacity, self.rng = capacity, random.Random(seed)
        self.recent_fraction = recent_fraction
        self.recent: list[TrainingExample] = []
        self.archive: list[TrainingExample] = []
        self.seen = 0
    @property
    def examples(self) -> list[TrainingExample]:
        """Compatibility view; sampling deliberately mixes recent and old games."""
        return self.recent + self.archive
    def extend(self, examples: list[TrainingExample]) -> None:
        for example in examples:
            self.add(example)
    def add(self, example: TrainingExample) -> None:
        self._compact(example)
        self.seen += 1
        recent_cap = max(1, int(self.capacity * self.recent_fraction))
        archive_cap = max(1, self.capacity - recent_cap)
        if len(self.recent) >= recent_cap:
            displaced = self.recent.pop(0)
            # Reservoir retention prevents a new self-play cycle from erasing
            # all successful older policies.
            if len(self.archive) < archive_cap:
                self.archive.append(displaced)
            else:
                slot = self.rng.randrange(self.seen)
                if slot < archive_cap:
                    self.archive[slot] = displaced
        self.recent.append(example)
    def sample(self, batch_size: int) -> list[TrainingExample]:
        population = self.examples
        return self.rng.sample(population, min(batch_size, len(population)))
    def __len__(self) -> int: return len(self.recent) + len(self.archive)
    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump({"capacity": self.capacity, "recent_fraction": self.recent_fraction,
                         "recent": self.recent, "archive": self.archive, "seen": self.seen,
                         "rng_state": self.rng.getstate()}, handle)
        temporary.replace(path)
    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        with open(path, "rb") as handle: data = pickle.load(handle)
        result = cls(data["capacity"], recent_fraction=float(data.get("recent_fraction", 0.6)))
        # Read the old flat replay format once, then keep it as the archive.
        if "examples" in data:
            result.archive = data["examples"][-result.capacity:]
            result.seen = len(result.archive)
        else:
            result.recent, result.archive = data.get("recent", []), data.get("archive", [])
            result.seen = int(data.get("seen", len(result.recent) + len(result.archive)))
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
