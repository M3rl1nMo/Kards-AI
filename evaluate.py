"""Evaluate a checkpoint against RandomAgent or RuleBasedAgent."""
from __future__ import annotations
import argparse
from pathlib import Path
from ai.agents import NeuralAgent, RandomAgent, RuleBasedAgent
from ai.evaluator import Evaluator
from ai.network import KARDSNet
from ai.observation import ObservationEncoder
from simulator.cards.loader import CardDatabase

ROOT = Path(__file__).parent
def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--games", type=int, default=20); parser.add_argument("--opponent", choices=("random", "rule"), default="random"); parser.add_argument("--report")
    args = parser.parse_args(); cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json"); encoder = ObservationEncoder(cards)
    candidate = NeuralAgent(KARDSNet.load_checkpoint(args.model), encoder); opponent = RandomAgent() if args.opponent == "random" else RuleBasedAgent()
    evaluator = Evaluator(cards, encoder); report = evaluator.evaluate(candidate, opponent, args.games)
    if args.report: evaluator.save_report(report, args.report)
    print(report)
if __name__ == "__main__": main()
