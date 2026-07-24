"""Regression coverage for the no-training admission gate."""

import unittest

from training.preflight import eligible_card_ids, run_preflight
from simulator.cards.loader import CardDatabase
from pathlib import Path


ROOT = Path(__file__).parents[1]


class TrainingPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")

    def test_active_catalog_is_eligible(self) -> None:
        eligible = eligible_card_ids(self.cards)
        self.assertEqual(len(eligible), 1486)
        self.assertNotIn("fortunes_of_war", eligible)
        self.assertNotIn("night_bombing", eligible)

    def test_seeded_preflight_executes_legal_trajectories(self) -> None:
        report = run_preflight(episodes=2, max_actions=100, seed=17)
        self.assertEqual(report.episodes, 2)
        self.assertGreater(report.actions_executed, 0)


if __name__ == "__main__":
    unittest.main()
