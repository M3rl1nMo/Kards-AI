"""Generate AlphaZero-style self-play training records."""
from __future__ import annotations
import argparse
from pathlib import Path
from ai.agents import MCTSAgent
from ai.config import load_config
from ai.network import KARDSNet
from ai.observation import ObservationEncoder
from ai.replay_buffer import ReplayBuffer
from ai.selfplay import SelfPlayRunner
from simulator.cards.loader import CardDatabase

ROOT = Path(__file__).parent
def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=ROOT / "configs/selfplay.yaml"); parser.add_argument("--episodes", type=int); parser.add_argument("--model")
    args = parser.parse_args(); cfg = load_config(args.config); cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json"); encoder = ObservationEncoder(cards)
    model = KARDSNet.load_checkpoint(args.model) if args.model else KARDSNet()
    sims = int(cfg["mcts_simulations"]); agents = (MCTSAgent(model, encoder, sims, 1), MCTSAgent(model, encoder, sims, 2))
    buffer = ReplayBuffer(); report = SelfPlayRunner(cards, encoder, buffer, int(cfg["max_actions"])).run(args.episodes or int(cfg["episodes"]), *agents, nation=str(cfg["nation"]))
    buffer.save(cfg["replay_path"]); print(report)
if __name__ == "__main__": main()
