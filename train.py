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
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=ROOT / "configs/training.yaml"); parser.add_argument("--replay", default="runs/replay.pkl"); parser.add_argument("--model")
    args = parser.parse_args(); cfg = load_config(args.config); buffer = ReplayBuffer.load(args.replay); model = KARDSNet.load_checkpoint(args.model) if args.model else KARDSNet()
    trainer = Trainer(model, float(cfg["learning_rate"])); metrics = {}
    for _ in range(int(cfg["updates"])): metrics = trainer.train_batch(buffer, int(cfg["batch_size"]))
    trainer.model.save_checkpoint(cfg["checkpoint"], device=trainer.device, **metrics); print(metrics)
if __name__ == "__main__": main()
