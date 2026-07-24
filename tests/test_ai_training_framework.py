"""End-to-end smoke tests for the separate AI training layer."""
from pathlib import Path
import tempfile
import unittest

import torch

from ai.action_encoder import ActionEncoder
from ai.agents import MCTSAgent, RandomAgent, RuleBasedAgent
from ai.mcts import MCTS
from ai.metrics import RunMetrics
from ai.network import KARDSNet
from ai.observation import ObservationEncoder, STATE_DIM
from ai.replay_buffer import ReplayBuffer
from ai.selfplay import SelfPlayRunner
from ai.trainer import Trainer
from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.testing import build_random_deck

ROOT = Path(__file__).parents[1]


class AITrainingFrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
        cls.encoder = ObservationEncoder(cls.cards)

    def _environment(self) -> Simulator:
        deck = build_random_deck(self.cards, seed=91)
        env = Simulator(self.cards); env.reset(deck, deck, nations=("France", "France")); return env

    def test_environment_random_agent_contract(self) -> None:
        env = self._environment(); agent = RandomAgent(1); legal = env.get_available_actions()
        action = agent.select_action(env.get_observation("p1"), legal)
        env.step(action); self.assertIsNotNone(env.state)

    def test_fixed_observation_and_action_mask(self) -> None:
        env = self._environment(); vector = self.encoder.encode(env.get_state(), "p1")
        features, mask = ActionEncoder().encode_legal_actions(env.get_available_actions())
        self.assertEqual(tuple(vector.shape), (STATE_DIM,)); self.assertEqual(features.shape[0], mask.shape[0]); self.assertTrue(mask.any())

    def test_network_forward_and_checkpoint(self) -> None:
        env = self._environment(); features, mask = ActionEncoder().encode_legal_actions(env.get_available_actions()); model = KARDSNet(hidden_dim=32)
        logits, value = model(self.encoder.encode(env.get_state(), "p1"), features, mask)
        self.assertEqual(logits.shape[-1], ActionEncoder().max_actions); self.assertEqual(tuple(value.shape), (1,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"; model.save_checkpoint(path); restored = KARDSNet.load_checkpoint(path)
            self.assertIsInstance(restored, KARDSNet)

    def test_mcts_agent_runs(self) -> None:
        env = self._environment(); agent = MCTSAgent(None, self.encoder, simulations=2); agent.set_state(env.get_state(), "p1", self.cards)
        action = agent.select_action(env.get_observation("p1"), env.get_available_actions()); env.step(action)

    def test_selfplay_replay_and_one_training_batch(self) -> None:
        buffer = ReplayBuffer(); runner = SelfPlayRunner(self.cards, self.encoder, buffer, max_actions=30, seed=3)
        report = runner.run(episodes=1, player_one=RandomAgent(1), player_two=RuleBasedAgent())
        self.assertGreater(report.examples, 0)
        metrics = Trainer(KARDSNet(hidden_dim=32), device="cpu").train_batch(buffer, 4)
        self.assertIn("loss", metrics)

    def test_random_agent_can_start_one_thousand_games(self) -> None:
        """Required scale check: agent/environment contract survives 1,000 resets."""
        buffer = ReplayBuffer()
        report = SelfPlayRunner(self.cards, self.encoder, buffer, max_actions=1, seed=31).run(
            episodes=1000, player_one=RandomAgent(1), player_two=RandomAgent(2))
        self.assertEqual(report.episodes, 1000)
        self.assertEqual(report.examples, 1000)
        self.assertGreaterEqual(report.average_turns, 1.0)

    def test_metrics_persist_cumulative_selfplay_games(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            first = RunMetrics(path, "selfplay"); first.emit("selfplay_complete", total_episodes=3)
            second = RunMetrics(path, "selfplay")
            self.assertEqual(second.previous_episodes, 3)


if __name__ == "__main__": unittest.main()
