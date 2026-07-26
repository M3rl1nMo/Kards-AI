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
from ai.config import load_config
import shutil
import subprocess as sp

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
    parser.add_argument("--config", default=ROOT / "configs/formal_training.yaml")
    parser.add_argument("--cycles", type=int, help="0 means run until STOP is created")
    parser.add_argument("--episodes", type=int); parser.add_argument("--mcts-simulations", type=int); parser.add_argument("--workers", type=int)
    parser.add_argument("--updates", type=int); parser.add_argument("--batch-size", type=int); parser.add_argument("--evaluation-games", type=int)
    args = parser.parse_args(); cfg = load_config(args.config); spcfg=cfg["self_play"]; tcfg=cfg["training"]; ecfg=cfg["evaluation"]; ccfg=cfg["checkpoint"]; lcfg=cfg["logging"]
    cycles = 0 if args.cycles is None else args.cycles; episodes=args.episodes or int(spcfg["episodes_per_cycle"]); sims=args.mcts_simulations or int(spcfg["simulations"]); workers=args.workers or int(spcfg["workers"]); updates=args.updates or int(tcfg["updates_per_cycle"]); batch=args.batch_size or int(tcfg["batch_size"]); eval_games=args.evaluation_games or int(ecfg["games_per_opponent"])
    RUNS.mkdir(parents=True, exist_ok=True); stop_file = RUNS / "STOP"; stop_file.unlink(missing_ok=True); pause_file = RUNS / "PAUSE"
    checkpoint, replay, metrics = RUNS / lcfg["latest_checkpoint"], RUNS / lcfg["replay_path"], RUNS / lcfg["metrics_path"]
    checkpoints = RUNS / lcfg["history_dir"]; checkpoints.mkdir(exist_ok=True); (RUNS / "config_snapshot.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    iteration = 0
    while cycles == 0 or iteration < cycles:
        if not wait_for_resume(stop_file, pause_file): break
        iteration += 1
        model_args = lambda: ["--model", str(checkpoint)] if checkpoint.exists() else []
        execute(["selfplay.py", "--episodes", str(episodes), "--mcts-simulations", str(sims), "--workers", str(workers), "--max-actions", str(spcfg["max_actions"]), "--device", "cuda", "--replay", str(replay), "--append-replay", "--replay-capacity", str(spcfg["replay_capacity"]), "--metrics", str(metrics), "--progress-every", "1", "--async-mcts", "--max-pending-leaves", str(spcfg["max_pending_leaves"]), *model_args()])
        if stop_file.exists(): break

        if not wait_for_resume(stop_file, pause_file): break
        execute(["train.py", "--replay", str(replay), "--updates", str(updates), "--batch-size", str(batch), "--learning-rate", str(tcfg["learning_rate"]), "--checkpoint", str(checkpoint), "--metrics", str(metrics), "--device", "cuda", *model_args()])
        if stop_file.exists(): break
        torch = __import__("torch"); payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload.setdefault("metadata", {}).update({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "config_snapshot": cfg,
                                                     "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
        torch.save(payload, checkpoint); step=int(payload.get("training_step", 0)); history=checkpoints / f"checkpoint_step_{step}.pt"; shutil.copy2(checkpoint, history)
        for old in sorted(checkpoints.glob("checkpoint_step_*.pt"))[:-int(ccfg["keep_last"])]: old.unlink()
        (RUNS / "SAVE_CHECKPOINT").unlink(missing_ok=True)
        if not wait_for_resume(stop_file, pause_file): break
        reports=[]
        for opponent in ecfg["opponents"]:
            report = RUNS / "evaluations" / f"cycle-{iteration:05d}-{opponent}.json"; report.parent.mkdir(exist_ok=True)
            execute(["evaluate.py", "--model", str(checkpoint), "--games", str(eval_games), "--opponent", opponent, "--report", str(report)]); reports.append(report)
        with (RUNS / lcfg["evaluation_history"]).open("a", encoding="utf-8") as handle:
            for report in reports: handle.write(json.dumps({"cycle": iteration, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "training_step": step, "report": str(report), **json.loads(report.read_text(encoding="utf-8"))}, ensure_ascii=False) + "\n")
        if stop_file.exists(): break

    (RUNS / "training.lock").unlink(missing_ok=True); (RUNS / "training.pid").unlink(missing_ok=True)


if __name__ == "__main__": main()
