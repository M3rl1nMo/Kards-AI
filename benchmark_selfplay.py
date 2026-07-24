"""Small reproducible CUDA self-play throughput benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch

from ai.agents import MCTSAgent
from ai.network import KARDSNet
from ai.observation import ObservationEncoder
from ai.replay_buffer import ReplayBuffer
from ai.selfplay import SelfPlayRunner
from simulator.cards.loader import CardDatabase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--max-actions", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    root = Path(__file__).parent; cards = CardDatabase.from_file(root / "data/source/kards_info_cards.json")
    model = KARDSNet(hidden_dim=32).to("cuda").eval()
    runner = SelfPlayRunner(cards, ObservationEncoder(cards), ReplayBuffer(), args.max_actions, seed=301)
    started = perf_counter()
    if args.workers == 1:
        runner.run(args.episodes, MCTSAgent(model, runner.encoder, args.simulations, 302),
                   MCTSAgent(model, runner.encoder, args.simulations, 303))
    else:
        runner.run_parallel(args.episodes, model, args.simulations, root / "data/source/kards_info_cards.json",
                            workers=args.workers, device="cuda")
    elapsed = perf_counter() - started
    print(f"workers={args.workers} elapsed={elapsed:.3f}s games_per_second={args.episodes / elapsed:.4f}")


if __name__ == "__main__":
    main()
