"""Focused regression tests for the Milestone 1 cost / resource mechanics.

These assert final GameState and event log, not merely "no exception".
"""

from pathlib import Path
import unittest

from simulator.actions.action import PlayCardAction, card_play_cost, _operation_cost
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.rules.native import NativeRuleEngine


ROOT = Path(__file__).parents[1]


class CostMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
        cls.engine = NativeRuleEngine(cls.cards)

    def _state(self, p1_hand=("active_sonar",), p2_hand=("garrison",), p1_units=(), p2_units=()) -> GameState:
        return GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Britain", deck=["garrison"], hand=list(p1_hand),
                                   resources=ResourceState(10, 10),
                                   units=[UnitState(uid, cid, 2, 3, "p1", "support_line") for uid, cid in p1_units]),
                "p2": PlayerState("p2", "Germany", deck=["garrison"], hand=list(p2_hand),
                                   resources=ResourceState(10, 10),
                                   units=[UnitState(uid, cid, 2, 3, "p2", "support_line") for uid, cid in p2_units]),
            },
        )

    def test_active_sonar_raises_enemy_hand_cost(self) -> None:
        state = self._state(p1_hand=("active_sonar",), p2_hand=("garrison",))
        p2_card = self.cards.get("garrison")
        base = p2_card.kredits or 0
        PlayCardAction("p1", "active_sonar").execute(state, self.cards)
        # Enemy hand cost increased by 3 via a persistent modifier.
        self.assertEqual(len(state.players["p2"].cost_modifiers), 1)
        self.assertEqual(card_play_cost(state, "p2", p2_card, self.cards), base + 3)

    def test_in_hour_of_need_gains_kredit_slots(self) -> None:
        state = self._state(p1_hand=("in_hour_of_need",))
        before = state.players["p1"].resources.max_kredits
        PlayCardAction("p1", "in_hour_of_need").execute(state, self.cards)
        self.assertEqual(state.players["p1"].resources.max_kredits, before + 2)

    def test_daylight_bombing_loses_enemy_kredit_slot(self) -> None:
        state = self._state(p1_hand=("daylight_bombing",), p2_units=(("enemy", "hurricane_mk_i"),))
        before = state.players["p2"].resources.max_kredits
        PlayCardAction("p1", "daylight_bombing", target_unit_id="enemy").execute(state, self.cards)
        self.assertEqual(state.players["p2"].resources.max_kredits, before - 1)

    def test_beriev_op_cost_rule_applies_on_deploy(self) -> None:
        # beriev_be4: "Your units cost 2 less to deploy." → modify_hand_cost (correct)
        rule = self.engine.rule_for("beriev_be4")
        self.assertIn("modify_hand_cost", [a.kind for a in rule.actions])
        self.assertEqual(rule.status, "implemented")

    def test_named_deployment_aura_reduces_cost_and_buffs_future_unit(self) -> None:
        """A name-scoped aura is continuous, and affects only later matches."""
        state = self._state(p1_hand=("raf_ground_crew", "spitfire_mk_ia"))
        crew = self.cards.get("raf_ground_crew")
        spitfire = self.cards.get("spitfire_mk_ia")
        PlayCardAction("p1", crew.id).execute(state, self.cards)

        self.assertEqual(card_play_cost(state, "p1", spitfire, self.cards), (spitfire.kredits or 0) - 2)
        PlayCardAction("p1", spitfire.id).execute(state, self.cards)
        deployed = next(unit for unit in state.players["p1"].units if unit.card_id == spitfire.id)
        self.assertEqual(deployed.attack, (spitfire.attack or 0) + 1)
        self.assertEqual(deployed.defense, (spitfire.defense or 0) + 1)
        self.assertTrue(any(event.get("event") == "deployment_name_aura_applied" for event in state.event_log))

    def test_land_of_the_free_sets_operation_cost_to_zero(self) -> None:
        state = self._state(p1_hand=("land_of_the_free",), p1_units=(("ally", "hurricane_mk_i"),))
        PlayCardAction("p1", "land_of_the_free", target_unit_id="ally").execute(state, self.cards)
        unit = state.players["p1"].units[0]
        card = self.cards.get(unit.card_id)
        self.assertTrue(any(m.get("type") == "set_operation_cost" and m.get("value") == 0 for m in unit.modifiers))
        self.assertEqual(_operation_cost(unit, card), 0)

    def test_cost_modifier_excludes_nation(self) -> None:
        # White-box check of the exclude_nation filter used by e.g. layforce.
        state = self._state(p1_hand=())
        state.players["p1"].cost_modifiers.append(
            {"kind": "hand_cost", "amount": 1, "set_cost": None,
             "scope": "exclude_nation", "filter_value": "Britain", "min_cost": 0, "expires_turn": None}
        )
        britain = self.cards.get("garrison")  # Britain card in this fixture
        non_britain = self.cards.get("t26_fi")  # Finland card -> not excluded by nation
        base_b = britain.kredits or 0
        base_n = non_britain.kredits or 0
        self.assertEqual(card_play_cost(state, "p1", britain, self.cards), base_b)
        self.assertEqual(card_play_cost(state, "p1", non_britain, self.cards), base_n + 1)


if __name__ == "__main__":
    unittest.main()
