"""Train KARDSNet on saved self-play data."""
from __future__ import annotations
import argparse
from pathlib import Path
from ai.config import load_config
from ai.network import KARDSNet
from ai.replay_buffer import ReplayBuffer
from ai.trainer import Trainer
from ai.metrics import RunMetrics

ROOT = Path(__file__).parent
def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=ROOT / "configs/training.yaml"); parser.add_argument("--replay", default="runs/replay.pkl"); parser.add_argument("--model"); parser.add_argument("--batch-size", type=int); parser.add_argument("--updates", type=int); parser.add_argument("--learning-rate", type=float); parser.add_argument("--checkpoint"); parser.add_argument("--device"); parser.add_argument("--metrics", default="runs/training_metrics.jsonl"); parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args(); cfg = load_config(args.config); buffer = ReplayBuffer.load(args.replay); model = KARDSNet.load_checkpoint(args.model) if args.model else KARDSNet()
    trainer = Trainer(model, args.learning_rate or float(cfg["learning_rate"]), args.device); latest = {}; updates = args.updates or int(cfg["updates"]); metrics = RunMetrics(args.metrics, "train")
    for update in range(1, updates + 1):
        latest = trainer.train_batch(buffer, args.batch_size or int(cfg["batch_size"]))
        if update % max(1, args.progress_every) == 0 or update == updates:
            elapsed = metrics.elapsed_seconds(); metrics.display_train(update, updates, latest, elapsed, trainer.device)
            metrics.emit("train_progress", completed_updates=update, requested_updates=updates, updates_per_second=update / max(elapsed, 1e-9), device=trainer.device, replay_examples=len(buffer), **latest)
    trainer.model.save_checkpoint(args.checkpoint or cfg["checkpoint"], device=trainer.device, **latest); metrics.emit("train_complete", checkpoint=str(args.checkpoint or cfg["checkpoint"]), device=trainer.device, replay_examples=len(buffer), **latest); print(latest)
if __name__ == "__main__": main()
