"""Operation-cost rules must affect matching current/future units and expire."""

from __future__ import annotations

from pathlib import Path
import unittest

from simulator.actions.action import _operation_cost, apply_op_cost_rules
from simulator.actions.action import PassAction, PlayCardAction
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.effects.resolver import EffectContext
from simulator.rules.native import NativeRuleEngine


ROOT = Path(__file__).parents[1]


class OperationCostRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
        cls.engine = NativeRuleEngine(cls.cards)

    def state(self) -> GameState:
        bomber = UnitState("bomber", "b25_mitchell", 3, 3, "p1", "support_line")
        tank = UnitState("tank", "m3a3_honey", 2, 3, "p1", "support_line")
        return GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Britain", units=[bomber, tank], resources=ResourceState(8, 8)),
                "p2": PlayerState("p2", "Germany"),
            },
            battlefield={"frontline": [], "support_line": ["bomber", "tank"]},
            graveyard={"p1": [], "p2": []},
        )

    def test_temporary_air_rule_applies_only_to_air_and_reverts(self) -> None:
        state = self.state()
        bomber, tank = state.players["p1"].units
        base_bomber = _operation_cost(bomber, self.cards.get(bomber.card_id))
        base_tank = _operation_cost(tank, self.cards.get(tank.card_id))

        self.engine.execute("dawn_operations", "on_play", state, EffectContext("p1", "dawn_operations"))

        self.assertEqual(_operation_cost(bomber, self.cards.get(bomber.card_id)), 0)
        self.assertEqual(_operation_cost(tank, self.cards.get(tank.card_id)), base_tank)
        NativeRuleEngine.revert_temporary(state, "p1", state.turn_number)
        self.assertEqual(_operation_cost(bomber, self.cards.get(bomber.card_id)), base_bomber)

    def test_precise_operation_cost_cards_are_fully_covered(self) -> None:
        for card_id in ("pursuit", "dawn_operations", "a20_havoc", "37_mm_antiaircraft_gun", "p40_n5"):
            self.assertEqual(self.engine.rule_for(card_id).status, "implemented", card_id)

    def test_future_matching_unit_receives_persistent_rule_once(self) -> None:
        state = self.state()
        self.engine.execute("a20_havoc", "on_deploy", state, EffectContext("p1", "a20_havoc", "bomber"))
        fresh = UnitState("fresh", "b25_mitchell", 3, 3, "p1")
        apply_op_cost_rules(state, "p1", fresh, self.cards)
        base = self.cards.get(fresh.card_id).operationCost or 0
        self.assertEqual(_operation_cost(fresh, self.cards.get(fresh.card_id)), max(0, base - 2))
        apply_op_cost_rules(state, "p1", fresh, self.cards)
        self.assertEqual(_operation_cost(fresh, self.cards.get(fresh.card_id)), max(0, base - 2))

    def test_campaign_trail_repairs_current_units_at_end_of_turn(self) -> None:
        state = self.state()
        unit = state.players["p1"].units[0]
        unit.defense = 1
        state.players["p1"].hand = ["campaign_trail"]
        PlayCardAction("p1", "campaign_trail").execute(state, self.cards)
        self.assertLess(_operation_cost(unit, self.cards.get(unit.card_id)), self.cards.get(unit.card_id).operationCost or 0)
        PassAction("p1").execute(state, self.cards)
        self.assertEqual(unit.defense, self.cards.get(unit.card_id).defense)
        self.assertEqual(_operation_cost(unit, self.cards.get(unit.card_id)), self.cards.get(unit.card_id).operationCost or 0)
