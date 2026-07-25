"""Regression coverage for the clone-free, replay-free self-play path."""

from pathlib import Path
import unittest

from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.testing import build_random_deck


ROOT = Path(__file__).parents[1]


class TrainingPerformanceModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")

    def _environment(self, *, record_replay: bool) -> Simulator:
        deck = build_random_deck(self.cards, seed=180)
        env = Simulator(self.cards, record_replay=record_replay)
        env.reset(deck, deck, nations=("France", "France"), seed=81)
        return env

    def test_debug_replay_remains_the_default(self) -> None:
        env = self._environment(record_replay=True)
        self.assertIsNotNone(env.replay)
        action = env.get_available_actions()[0]
        env.step(action)
        self.assertEqual(len(env.replay.entries), 1)  # type: ignore[union-attr]

    def test_fast_step_matches_public_step_state_transition(self) -> None:
        public_env = self._environment(record_replay=False)
        fast_env = self._environment(record_replay=False)
        action = public_env.get_available_actions()[0]
        public_state, public_reward, public_terminal = public_env.step(action)
        fast_reward, fast_terminal = fast_env.step_fast(action)
        self.assertEqual((fast_reward, fast_terminal), (public_reward, public_terminal))
        self.assertEqual(fast_env.state.to_dict(), public_state.to_dict())  # type: ignore[union-attr]
        self.assertIsNone(fast_env.replay)


if __name__ == "__main__":
    unittest.main()
