"""Measure legacy pickle queues against shared-memory inference transport."""
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


def run_case(name: str, shared: bool, args, cards, model) -> None:
    runner = SelfPlayRunner(cards, ObservationEncoder(cards), ReplayBuffer(), args.max_actions, seed=911)
    started = perf_counter()
    report = runner.run_parallel(
        args.episodes, model, args.simulations, ROOT / "data/source/kards_info_cards.json", workers=args.workers,
        device="cuda", measure_inference=True, shared_inference=shared,
        mcts_options={"async_inference": True, "max_pending_leaves": args.max_pending_leaves},
    )
    elapsed = perf_counter() - started
    transport = runner.last_transport_stats
    print(" ".join((
        f"transport={name}",
        f"games_per_second={report.episodes / elapsed:.4f}",
        f"examples_per_second={report.examples / elapsed:.4f}",
        f"request_transport_ms={transport.get('average_request_transport_ms', 0):.4f}",
        f"serialization_ms={transport.get('average_serialization_ms', 0):.4f}",
        f"deserialization_ms={transport.get('average_deserialization_ms', 0):.4f}",
        f"inference_wait_ms={transport.get('average_inference_wait_ms', 0):.4f}",
        f"requests={transport.get('requests', 0):.0f}",
        f"average_batch_size={runner.last_inference_stats.get('average_batch_size', 0):.3f}",
    )))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--max-actions", type=int, default=120)
    parser.add_argument("--max-pending-leaves", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
    model = KARDSNet(hidden_dim=128).to("cuda").eval()
    run_case("pickle", False, args, cards, model)
    run_case("shared_memory", True, args, cards, model)


if __name__ == "__main__":
    main()
