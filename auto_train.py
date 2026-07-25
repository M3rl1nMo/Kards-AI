"""Continuous self-play, training, and evaluation loop with graceful stopping."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from ai.runtime import runs_dir

ROOT = Path(__file__).parent
RUNS = runs_dir()


def execute(arguments: list[str]) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def wait_for_resume(stop_file: Path, pause_file: Path) -> bool:
    """Safe boundary control: never terminate an active self-play/optimizer job."""
    while pause_file.exists():
        if stop_file.exists():
            return False
        time.sleep(2)
    return not stop_file.exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=0, help="0 means run until STOP is created")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--mcts-simulations", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluation-games", type=int, default=50)
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True); stop_file = RUNS / "STOP"; stop_file.unlink(missing_ok=True); pause_file = RUNS / "PAUSE"
    checkpoint, replay, metrics = RUNS / "kardsnet.pt", RUNS / "replay.pkl", RUNS / "training_metrics.jsonl"
    iteration = 0
    while args.cycles == 0 or iteration < args.cycles:
        if not wait_for_resume(stop_file, pause_file): break
        iteration += 1
        model_args = ["--model", str(checkpoint)] if checkpoint.exists() else []
        execute(["selfplay.py", "--episodes", str(args.episodes), "--mcts-simulations", str(args.mcts_simulations), "--workers", str(args.workers), "--device", "cuda", "--replay", str(replay), "--append-replay", "--metrics", str(metrics), "--progress-every", "1", *model_args])
        if stop_file.exists(): break
        if not wait_for_resume(stop_file, pause_file): break
        execute(["train.py", "--replay", str(replay), "--updates", str(args.updates), "--batch-size", str(args.batch_size), "--checkpoint", str(checkpoint), "--metrics", str(metrics), "--device", "cuda", *model_args])
        if stop_file.exists(): break
        (RUNS / "SAVE_CHECKPOINT").unlink(missing_ok=True)
        if not wait_for_resume(stop_file, pause_file): break
        report = RUNS / "evaluations" / f"cycle-{iteration:05d}.json"; report.parent.mkdir(exist_ok=True)
        execute(["evaluate.py", "--model", str(checkpoint), "--games", str(args.evaluation_games), "--opponent", "rule", "--report", str(report)])
        with (RUNS / "cycle_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"cycle": iteration, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "report": str(report)}, ensure_ascii=False) + "\n")
        if stop_file.exists(): break


if __name__ == "__main__": main()
