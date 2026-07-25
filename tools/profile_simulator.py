"""Reproducible simulator throughput and cProfile harness.

This exercises the public training-facing API (reset, action generation,
clone/state access, observation, and step) with legal random actions.  It is
intentionally independent from the neural network so reported time is engine
runtime rather than MCTS/GPU inference time.
"""
from __future__ import annotations

import argparse
import cProfile
from collections import defaultdict
from pathlib import Path
import pstats
import random
import sys
from time import perf_counter, process_time
import tracemalloc


ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.testing import build_random_deck


def run(games: int, max_steps: int, seed: int, *, replay: bool = True, training_path: bool = False,
        measure_memory: bool = False) -> dict[str, float]:
    cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
    rng = random.Random(seed)
    elapsed = defaultdict(float)
    total_steps = 0
    completed = 0
    if measure_memory:
        tracemalloc.start()
    started = perf_counter()
    cpu_started = process_time()
    for game_index in range(games):
        deck_one = build_random_deck(cards, seed=rng.randrange(2**31))
        deck_two = build_random_deck(cards, seed=rng.randrange(2**31))
        env = Simulator(cards, record_replay=replay)
        phase = perf_counter()
        env.reset(deck_one, deck_two, seed=rng.randrange(2**31))
        elapsed["reset"] += perf_counter() - phase
        for _ in range(max_steps):
            if env.is_terminal():
                break
            phase = perf_counter(); actions = env.get_available_actions(); elapsed["actions"] += perf_counter() - phase
            if not actions:
                break
            if not training_path:
                phase = perf_counter(); env.get_state(); elapsed["clone"] += perf_counter() - phase
            phase = perf_counter(); env.get_observation(env.state.current_player)  # type: ignore[union-attr]
            elapsed["observation"] += perf_counter() - phase
            phase = perf_counter()
            action = actions[rng.randrange(len(actions))]
            if training_path:
                env.step_fast(action)
            else:
                env.step(action)
            elapsed["step"] += perf_counter() - phase
            total_steps += 1
        completed += 1
    total = perf_counter() - started
    cpu_seconds = process_time() - cpu_started
    peak_bytes = 0
    if measure_memory:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return {"games": float(completed), "steps": float(total_steps), "total": total,
            "games_per_second": completed / total, "steps_per_second": total_steps / total,
            "process_cpu_percent": 100.0 * cpu_seconds / total,
            "python_peak_memory_mb": peak_bytes / 1_048_576,
            **elapsed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--training-path", action="store_true", help="Use the clone-free internal self-play path")
    parser.add_argument("--memory", action="store_true", help="Measure Python allocations (adds profiler overhead)")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    profiler = cProfile.Profile() if args.profile else None
    kwargs = {"replay": not args.no_replay, "training_path": args.training_path, "measure_memory": args.memory}
    result = profiler.runcall(run, args.games, args.max_steps, args.seed, **kwargs) if profiler else run(args.games, args.max_steps, args.seed, **kwargs)
    print(" ".join(f"{key}={value:.6f}" for key, value in result.items()))
    if profiler:
        pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(35)


if __name__ == "__main__":
    main()
