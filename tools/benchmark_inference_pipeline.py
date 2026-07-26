"""Measure real multi-worker self-play inference batching without changing training data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import torch

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.network import KARDSNet
from ai.observation import ObservationEncoder
from ai.replay_buffer import ReplayBuffer
from ai.selfplay import SelfPlayRunner
from simulator.cards.loader import CardDatabase

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--max-actions", type=int, default=120)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
    runner = SelfPlayRunner(cards, ObservationEncoder(cards), ReplayBuffer(), args.max_actions, seed=701)
    model = KARDSNet(hidden_dim=128).to("cuda").eval()
    started = perf_counter()
    report = runner.run_parallel(args.episodes, model, args.simulations, ROOT / "data/source/kards_info_cards.json",
                                 workers=args.workers, device="cuda", measure_inference=True)
    elapsed = perf_counter() - started; stats = runner.last_inference_stats
    print(" ".join((
        f"games_per_second={report.episodes / elapsed:.4f}",
        f"examples_per_second={report.examples / elapsed:.4f}",
        f"gpu_forwards={stats.get('gpu_forwards', 0):.0f}",
        f"inference_requests={stats.get('inference_requests', 0):.0f}",
        f"average_batch_size={stats.get('average_batch_size', 0):.3f}",
        f"inferences_per_second={stats.get('inferences_per_second', 0):.2f}",
        f"gpu_seconds={stats.get('gpu_seconds', 0):.4f}",
    )))


if __name__ == "__main__":
    main()
