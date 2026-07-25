"""Training-path throughput benchmark: simulator, MCTS, and neural inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from time import perf_counter

import torch

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.action_encoder import ACTION_FEATURE_DIM
from ai.agents import MCTSAgent
from ai.network import KARDSNet
from ai.observation import ObservationEncoder, STATE_DIM
from ai.replay_buffer import ReplayBuffer
from ai.selfplay import SelfPlayRunner
from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.testing import build_random_deck


def _random_games(cards: CardDatabase, games: int, steps: int, seed: int, fast: bool) -> dict[str, float]:
    rng = random.Random(seed); count = candidates = 0; started = perf_counter()
    for _ in range(games):
        env = Simulator(cards, record_replay=False)
        env.reset(build_random_deck(cards, seed=rng.randrange(2**31)), build_random_deck(cards, seed=rng.randrange(2**31)), seed=rng.randrange(2**31))
        for _ in range(steps):
            if env.is_terminal():
                break
            legal = env.get_available_actions(); candidates += len(legal)
            action = legal[rng.randrange(len(legal))]
            if fast:
                env.step_fast(action)
            else:
                env.step(action)
            count += 1
    elapsed = perf_counter() - started
    return _rates(games, count, candidates, elapsed)


def _rates(games: int, steps: int, actions: int, elapsed: float) -> dict[str, float]:
    return {"games_per_second": games / elapsed, "steps_per_second": steps / elapsed,
            "actions_per_second": actions / elapsed, "elapsed_seconds": elapsed}


def _mcts(cards: CardDatabase, games: int, max_actions: int, simulations: int, device: str) -> dict[str, float]:
    model = KARDSNet(hidden_dim=32).to(device).eval(); encoder = ObservationEncoder(cards)
    runner = SelfPlayRunner(cards, encoder, ReplayBuffer(), max_actions=max_actions, seed=301)
    started = perf_counter(); report = runner.run(games, MCTSAgent(model, encoder, simulations, 302), MCTSAgent(model, encoder, simulations, 303)); elapsed = perf_counter() - started
    return _rates(games, report.examples, report.examples * 1, elapsed)


@torch.inference_mode()
def _inference(batch: int, iterations: int, device: str) -> dict[str, float]:
    model = KARDSNet(hidden_dim=32).to(device).eval()
    states = torch.zeros((batch, STATE_DIM), device=device)
    actions = torch.zeros((batch, 32, ACTION_FEATURE_DIM), device=device)
    mask = torch.ones((batch, 32), dtype=torch.bool, device=device)
    for _ in range(10):
        model(states, actions, mask)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = perf_counter()
    for _ in range(iterations):
        model(states, actions, mask)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = perf_counter() - started
    return {"inferences_per_second": batch * iterations / elapsed, "batches_per_second": iterations / elapsed, "elapsed_seconds": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=16); parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--simulations", type=int, default=16); parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inference-batch", type=int, default=32); parser.add_argument("--inference-iterations", type=int, default=500)
    args = parser.parse_args(); cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
    results = {
        "A_random_public": _random_games(cards, args.games, args.max_steps, 41, False),
        "B_step_fast": _random_games(cards, args.games, args.max_steps, 41, True),
        "C_mcts": _mcts(cards, 1, args.max_steps, args.simulations, args.device),
        "D_network": _inference(args.inference_batch, args.inference_iterations, args.device),
    }
    for name, metrics in results.items():
        print(name, " ".join(f"{key}={value:.4f}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()
