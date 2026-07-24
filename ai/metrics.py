"""Human-readable, append-only metrics for training runs."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any


class RunMetrics:
    """Write JSONL records and print compact live progress to the terminal."""

    def __init__(self, path: str | Path | None, label: str) -> None:
        self.path = Path(path) if path else None
        self.label = label
        self.started_at = time.perf_counter()
        self.previous_episodes = self._previous_episodes() if self.path else 0

    def _previous_episodes(self) -> int:
        if self.path is None or not self.path.exists():
            return 0
        total = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("event") in {"selfplay_progress", "selfplay_complete"}:
                total = max(total, int(item.get("total_episodes", 0)))
        return total

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    def emit(self, event: str, **data: Any) -> dict[str, Any]:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "label": self.label,
            "event": event,
            "elapsed_seconds": round(self.elapsed_seconds(), 3),
            **{key: asdict(value) if is_dataclass(value) else value for key, value in data.items()},
        }
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record

    @staticmethod
    def display_selfplay(completed: int, total: int, cumulative_games: int, wins: int, losses: int, draws: int,
                         examples: int, elapsed_seconds: float) -> None:
        rate = completed / elapsed_seconds if elapsed_seconds else 0.0
        win_rate = wins / completed if completed else 0.0
        print(
            f"[self-play] {completed}/{total} games (total {cumulative_games}) | {rate:.2f} games/s | "
            f"W/L/D {wins}/{losses}/{draws} | win rate {win_rate:.1%} | examples {examples}",
            flush=True,
        )

    @staticmethod
    def display_train(completed: int, total: int, metrics: dict[str, float], elapsed_seconds: float,
                      device: str) -> None:
        rate = completed / elapsed_seconds if elapsed_seconds else 0.0
        print(
            f"[train] {completed}/{total} updates | {rate:.2f} updates/s | {device} | "
            f"loss {metrics['loss']:.4f} (policy {metrics['policy_loss']:.4f}, value {metrics['value_loss']:.4f})",
            flush=True,
        )
