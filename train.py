"""Train KARDSNet on saved self-play data."""
from __future__ import annotations
import argparse
from pathlib import Path
from ai.config import load_config
from ai.network import KARDSNet
from ai.replay_buffer import ReplayBuffer
from ai.trainer import Trainer

ROOT = Path(__file__).parent
def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=ROOT / "configs/training.yaml"); parser.add_argument("--replay", default="runs/replay.pkl"); parser.add_argument("--model"); parser.add_argument("--batch-size", type=int); parser.add_argument("--updates", type=int); parser.add_argument("--learning-rate", type=float); parser.add_argument("--checkpoint"); parser.add_argument("--device")
    args = parser.parse_args(); cfg = load_config(args.config); buffer = ReplayBuffer.load(args.replay); model = KARDSNet.load_checkpoint(args.model) if args.model else KARDSNet()
    trainer = Trainer(model, args.learning_rate or float(cfg["learning_rate"]), args.device); metrics = {}
    for _ in range(args.updates or int(cfg["updates"])): metrics = trainer.train_batch(buffer, args.batch_size or int(cfg["batch_size"]))
    trainer.model.save_checkpoint(args.checkpoint or cfg["checkpoint"], device=trainer.device, **metrics); print(metrics)
if __name__ == "__main__": main()
