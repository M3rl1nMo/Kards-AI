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
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=ROOT / "configs/selfplay.yaml"); parser.add_argument("--episodes", type=int); parser.add_argument("--model"); parser.add_argument("--mcts-simulations", type=int); parser.add_argument("--max-actions", type=int); parser.add_argument("--nation"); parser.add_argument("--replay"); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(); cfg = load_config(args.config); cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json"); encoder = ObservationEncoder(cards)
    model = KARDSNet.load_checkpoint(args.model) if args.model else KARDSNet()
    sims = args.mcts_simulations or int(cfg["mcts_simulations"]); agents = (MCTSAgent(model, encoder, sims, args.seed + 1), MCTSAgent(model, encoder, sims, args.seed + 2))
    buffer = ReplayBuffer(seed=args.seed); report = SelfPlayRunner(cards, encoder, buffer, args.max_actions or int(cfg["max_actions"]), args.seed).run(args.episodes or int(cfg["episodes"]), *agents, nation=args.nation or str(cfg["nation"]))
    buffer.save(args.replay or cfg["replay_path"]); print(report)
if __name__ == "__main__": main()
