"""Focused regression tests for Milestone 3: targeting & multi-target selection.

These assert final GameState and event log, not merely "no exception". The
core M3 deliverable is scoped (type/nation/ability) multi-target resolution,
which is exercised here directly through the engine's scoped combat path and
end-to-end through a real order card.
"""

from pathlib import Path
import unittest

from simulator.actions.action import PlayCardAction
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.effects.resolver import EffectContext
from simulator.rules.native import NativeRuleEngine
from simulator.rules.parser import RuleAction


ROOT = Path(__file__).parents[1]


class TargetingMultiTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")

    def _state(self, p1_units=(), p2_units=(), p1_hand=(), p2_hand=()):
        return GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Germany", deck=["garrison"], hand=list(p1_hand),
                                  resources=ResourceState(10, 10),
                                  units=[UnitState(uid, cid, atk, dfn, "p1", "support_line")
                                         for uid, cid, atk, dfn in p1_units]),
                "p2": PlayerState("p2", "USA", deck=["garrison"], hand=list(p2_hand),
                                  resources=ResourceState(10, 10),
                                  units=[UnitState(uid, cid, atk, dfn, "p2", "support_line")
                                         for uid, cid, atk, dfn in p2_units]),
            },
        )

    def test_deal_bare_damage_to_enemy_hq(self) -> None:
        # "Deal 3 damage." (no explicit target) -> enemy HQ by KARDS convention.
        state = self._state()
        engine = NativeRuleEngine(self.cards)
        before = state.players["p2"].hq.current_health
        engine._execute_action(RuleAction("damage", "enemy_hq", amount=3), state, EffectContext("p1"))
        self.assertEqual(state.players["p2"].hq.current_health, before - 3)
        self.assertTrue(any(e.get("event") == "hq_damaged" for e in state.event_log))

    def test_mass_buff_scoped_by_type(self) -> None:
        # "Friendly infantry has +1 attack." applies only to infantry units.
        state = self._state(p1_units=(("u1", "10th_para_battalion", 2, 3), ("u2", "greif", 3, 4)))
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(
            RuleAction("modify_attack", "friendly_units", amount=1, scope="infantry"),
            state, EffectContext("p1"),
        )
        inf = next(u for u in state.players["p1"].units if u.instance_id == "u1")
        tank = next(u for u in state.players["p1"].units if u.instance_id == "u2")
        self.assertEqual(inf.attack, 3)   # 2 + 1
        self.assertEqual(tank.attack, 3)  # unchanged (tank, not infantry)

    def test_mass_buff_scoped_by_nation(self) -> None:
        # "Your German tanks get +2+1." applies only to German units.
        state = self._state(p1_units=(("u1", "greif", 3, 4), ("u2", "10th_para_battalion", 2, 3)))
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(
            RuleAction("buff", "friendly_units", attack=2, defense=1, card_name="Germany"),
            state, EffectContext("p1"),
        )
        ger = next(u for u in state.players["p1"].units if u.instance_id == "u1")
        other = next(u for u in state.players["p1"].units if u.instance_id == "u2")
        self.assertEqual(ger.attack, 5)
        self.assertEqual(ger.defense, 5)
        self.assertEqual(other.attack, 2)  # unchanged (not German)

    def test_pin_all_enemy_air_units_scoped(self) -> None:
        # "Pin all enemy air units." suppresses only enemy air units.
        state = self._state(p2_units=(("e1", "gladiator_escort", 3, 4), ("e2", "10th_para_battalion", 2, 3)))
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(
            RuleAction("suppress", "enemy_units", scope="air"), state, EffectContext("p1"),
        )
        air_unit = next(u for u in state.players["p2"].units if u.instance_id == "e1")
        ground_unit = next(u for u in state.players["p2"].units if u.instance_id == "e2")
        self.assertIn("suppressed", air_unit.status)
        self.assertNotIn("suppressed", ground_unit.status)

    def test_deal_damage_to_all_enemy_units(self) -> None:
        state = self._state(p2_units=(("e1", "greif", 3, 5), ("e2", "10th_para_battalion", 2, 5)))
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(RuleAction("damage", "enemy_units", amount=2), state, EffectContext("p1"))
        for u in state.players["p2"].units:
            self.assertEqual(u.defense, 3)  # 5 - 2
        self.assertEqual(len([e for e in state.event_log if e.get("event") == "damage_dealt"]), 2)

    def test_destroy_all_enemy_units(self) -> None:
        state = self._state(p2_units=(("e1", "greif", 3, 5), ("e2", "10th_para_battalion", 2, 5)))
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(RuleAction("destroy", "enemy_units"), state, EffectContext("p1"))
        self.assertEqual(len(state.players["p2"].units), 0)
        self.assertTrue(any(e.get("event") == "unit_died" for e in state.event_log))

    def test_single_target_scoped_buff(self) -> None:
        # "Give a friendly tank +1+1" (single chosen tank, type-scoped).
        state = self._state(p1_units=(("u1", "greif", 3, 4), ("u2", "10th_para_battalion", 2, 3)))
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(
            RuleAction("buff", "selected_friendly", attack=1, defense=1, scope="tank"),
            state, EffectContext("p1", target_unit_id="u1"),
        )
        tank = next(u for u in state.players["p1"].units if u.instance_id == "u1")
        other = next(u for u in state.players["p1"].units if u.instance_id == "u2")
        self.assertEqual(tank.attack, 4)
        self.assertEqual(tank.defense, 5)
        self.assertEqual(other.attack, 2)  # unchanged

    def test_integration_order_deals_damage_to_all_enemy_units(self) -> None:
        # forward_observers: "Deal 2 damage to all enemy units."
        state = self._state(p2_units=(("e1", "greif", 3, 6), ("e2", "10th_para_battalion", 2, 6)),
                            p1_hand=("forward_observers",))
        PlayCardAction("p1", "forward_observers").execute(state, self.cards)
        for u in state.players["p2"].units:
            self.assertEqual(u.defense, 4)  # 6 - 2
        self.assertTrue(any(e.get("event") == "damage_dealt" for e in state.event_log))


if __name__ == "__main__":
    unittest.main()
