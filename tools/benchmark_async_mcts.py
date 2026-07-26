"""Compare the synchronous and pipelined MCTS inference paths on real games."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
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


def _gpu_sampler(stop: Event, samples: list[float]) -> None:
    """Best-effort NVIDIA utilization sampling; benchmark still runs without it."""
    while not stop.wait(0.5):
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL,
            )
            samples.append(float(output.splitlines()[0].strip()))
        except (OSError, ValueError, subprocess.CalledProcessError):
            return


def run_case(name: str, *, async_inference: bool, args, cards, model) -> None:
    runner = SelfPlayRunner(cards, ObservationEncoder(cards), ReplayBuffer(), args.max_actions, seed=701)
    stop, gpu_samples = Event(), []
    sampler = Thread(target=_gpu_sampler, args=(stop, gpu_samples), daemon=True)
    sampler.start()
    started = perf_counter()
    try:
        report = runner.run_parallel(
            args.episodes, model, args.simulations, ROOT / "data/source/kards_info_cards.json",
            workers=args.workers, device="cuda", measure_inference=True,
            mcts_options={"async_inference": async_inference, "max_pending_leaves": args.max_pending_leaves},
        )
    finally:
        stop.set(); sampler.join(timeout=1)
    elapsed = perf_counter() - started
    inference, async_stats = runner.last_inference_stats, runner.last_async_stats
    print(" ".join((
        f"mode={name}",
        f"games_per_second={report.episodes / elapsed:.4f}",
        f"examples_per_second={report.examples / elapsed:.4f}",
        f"average_batch_size={inference.get('average_batch_size', 0):.3f}",
        f"gpu_forwards={inference.get('gpu_forwards', 0):.0f}",
        f"inferences_per_second={inference.get('inferences_per_second', 0):.2f}",
        f"gpu_seconds={inference.get('gpu_seconds', 0):.4f}",
        f"gpu_utilization_percent={sum(gpu_samples) / len(gpu_samples) if gpu_samples else 0:.1f}",
        f"inference_latency_ms={async_stats.get('average_inference_latency_ms', 0):.3f}",
        f"pending_queue_peak={async_stats.get('pending_queue_peak', 0):.0f}",
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
    run_case("sync", async_inference=False, args=args, cards=cards, model=model)
    run_case("async", async_inference=True, args=args, cards=cards, model=model)


if __name__ == "__main__":
    main()
