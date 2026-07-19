"""High-volume, deterministic stability checks for the training environment."""

from pathlib import Path
import unittest

from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.testing import build_random_deck, run_large_scale_smoke
from simulator.cards.availability import RETIRED_CARD_IDS


ROOT = Path(__file__).parents[1]


class TrainingStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")

    def test_reset_rejects_illegal_deck(self) -> None:
        card = next(card for card in self.cards if card.nation == "France" and not card.is_token)
        with self.assertRaisesRegex(ValueError, "Illegal deck"):
            Simulator(self.cards).reset([card.id] * 40, [card.id] * 40, nations=("France", "France"))

    def test_observation_hides_opponent_hand(self) -> None:
        deck = build_random_deck(self.cards, seed=11, main_nation="France")
        simulator = Simulator(self.cards)
        simulator.reset(deck, deck, nations=("France", "France"))
        observation = simulator.get_observation("p1")
        self.assertIsInstance(observation["self"]["hand"], list)
        self.assertIsNone(observation["opponent"]["hand"])

    def test_random_deck_builder_excludes_retired_cards_for_known_bad_seed(self) -> None:
        # Seed 19 previously selected FORTUNES OF WAR for a USA deck, then
        # failed validation after the card was retired.
        deck = build_random_deck(self.cards, main_nation="USA", seed=19)
        self.assertFalse(set(deck) & RETIRED_CARD_IDS)

    def test_one_thousand_legal_smoke_games(self) -> None:
        self.assertEqual(run_large_scale_smoke(self.cards, games=1000, seed=100), 1000)


if __name__ == "__main__":
    unittest.main()
