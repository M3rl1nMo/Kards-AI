"""Regression tests for DILEMMA targeting and repeated swaps."""

from pathlib import Path
import unittest

from simulator.actions.action import PlayCardAction, _operation_cost
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState


class DilemmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cards = CardDatabase.from_file(
            Path(__file__).parents[1] / "data/source/kards_info_cards.json"
        )

    def test_friendly_and_enemy_targets_restore_stats_after_two_swaps(self):
        card = self.cards.get("1_infantry_regiment")
        for owner in ("p1", "p2"):
            for attack in (0, 4, 7):
                with self.subTest(owner=owner, attack=attack):
                    unit = UnitState("target", card.id, attack, card.defense, owner)
                    # An unrelated modifier must survive replacement of the set cost.
                    unit.modifiers.append({"type": "modify_attack", "amount": 1})
                    state = GameState(
                        current_player="p1",
                        players={
                            "p1": PlayerState("p1", "Britain", hand=["dilemma"] * 2,
                                              resources=ResourceState(10, 10)),
                            "p2": PlayerState("p2", "Germany"),
                        },
                        battlefield={"frontline": [], "support_line": ["target"]},
                        graveyard={"p1": [], "p2": []},
                        rng_seed=1,
                    )
                    state.players[owner].units.append(unit)
                    original = (unit.attack, _operation_cost(unit, card), unit.defense)
                    for expected in ((original[1], original[0], original[2]), original):
                        PlayCardAction("p1", "dilemma", target_unit_id="target").execute(state, self.cards)
                        self.assertEqual((unit.attack, _operation_cost(unit, card), unit.defense), expected)
                        self.assertEqual(sum(m.get("type") == "set_operation_cost"
                                             for m in unit.modifiers), 1)
                        self.assertIn({"type": "modify_attack", "amount": 1}, unit.modifiers)
                    self.assertFalse(any(e.get("event") == "unsupported_effect"
                                         for e in state.event_log))


if __name__ == "__main__":
    unittest.main()
