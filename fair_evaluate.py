"""Run fair, seat-balanced MCTS benchmark of a KARDS checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from ai.fair_evaluator import FairEvaluator
from ai.network import KARDSNet
from ai.observation import ObservationEncoder
from simulator.cards.loader import CardDatabase

ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--opponent", choices=("random", "rule", "model"), default="rule")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--nations", default="Britain,France,Germany,Japan,Soviet,USA")
    parser.add_argument("--deck-pool", help="JSON list of {main, ally} official deck specifications")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    nations = [nation.strip() for nation in args.nations.split(",") if nation.strip()]
    raw_pool = json.loads(args.deck_pool) if args.deck_pool else []
    deck_pool = [(item["main"], item.get("ally")) for item in raw_pool] or None
    cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
    evaluator = FairEvaluator(cards, ObservationEncoder(cards), seed=args.seed)
    report = evaluator.evaluate(KARDSNet.load_checkpoint(args.model, device=args.device), args.opponent,
                                games=args.games, simulations=args.simulations, nations=nations, deck_pool=deck_pool)
    evaluator.save(report, args.report)
    print(report)


if __name__ == "__main__":
    main()
