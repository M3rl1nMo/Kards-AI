"""Focused regression tests for Milestone 4: temporary / duration-bounded effects.

Covers the temporary-effect mechanism: a `duration` on reversible actions is
applied and its reversal is registered, then reverted at the correct turn
boundary (this_turn at end of turn; next_turn after the player's next turn).
Cost modifiers / op-cost rules also honor an expires_turn. Also asserts the
parser attaches durations from 'this turn' / 'next turn' / 'until ...' markers.
"""

from pathlib import Path
import unittest

from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.core.turn import TurnManager
from simulator.effects.resolver import EffectContext
from simulator.rules.native import NativeRuleEngine
from simulator.rules.parser import RuleAction, RuleParser

ROOT = Path(__file__).parents[1]


class TemporaryEffectsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
        cls.parser = RuleParser()

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

    # --- Parser duration attachment ---

    def test_parser_attaches_this_turn(self) -> None:
        rule = self.parser.parse(self.cards.get("rally"))  # "... get +1 attack this turn."
        act = next(a for a in rule.actions if a.kind == "modify_attack")
        self.assertEqual(act.duration, "this_turn")

    def test_parser_attaches_next_turn(self) -> None:
        rule = self.parser.parse(self.cards.get("raid"))  # "Cards in enemy hand cost 2 more next turn."
        act = next(a for a in rule.actions if a.kind == "modify_hand_cost")
        self.assertEqual(act.duration, "next_turn")
        self.assertEqual(act.target, "enemy_hand")

    def test_parser_until_end_of_turn(self) -> None:
        rule = self.parser.parse(self.cards.get("old_hares"))  # "... gets +5 attack until end of turn."
        act = next(a for a in rule.actions if a.kind == "modify_attack")
        self.assertEqual(act.duration, "this_turn")
        self.assertEqual(act.amount, 5)

    def test_parser_set_attack_zero_temporary(self) -> None:
        rule = self.parser.parse(self.cards.get("glide_bombing"))  # "Target unit has 0 attack until your next turn."
        act = next((a for a in rule.actions if a.kind == "set_attack"), None)
        self.assertIsNotNone(act)
        self.assertEqual(act.duration, "next_turn")

    # --- Native temporary application + revert ---

    def test_temporary_buff_reverts_at_turn_end(self) -> None:
        state = self._state(p1_units=(("u1", "greif", 3, 1),))
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(
            RuleAction("modify_attack", "friendly_units", amount=2, duration="this_turn"),
            state, EffectContext("p1"),
        )
        unit = next(u for u in state.players["p1"].units if u.instance_id == "u1")
        self.assertEqual(unit.attack, 5)  # 3 + 2
        TurnManager.end_turn(state, "p1")
        self.assertEqual(unit.attack, 3)  # reverted
        self.assertTrue(any(e.get("event") == "temporary_effect_reverted" for e in state.event_log))

    def test_next_turn_survives_then_reverts(self) -> None:
        state = self._state(p1_units=(("u1", "greif", 3, 1),))
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(
            RuleAction("modify_attack", "friendly_units", amount=2, duration="next_turn"),
            state, EffectContext("p1"),
        )
        unit = next(u for u in state.players["p1"].units if u.instance_id == "u1")
        self.assertEqual(unit.attack, 5)
        # End of the same turn: next_turn effect must still be active.
        NativeRuleEngine.revert_temporary(state, "p1", state.turn_number)
        self.assertEqual(unit.attack, 5)
        # End of the player's next turn: now it reverts.
        NativeRuleEngine.revert_temporary(state, "p1", state.turn_number + 1)
        self.assertEqual(unit.attack, 3)

    def test_temporary_cost_modifier_expires(self) -> None:
        state = self._state()
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(
            RuleAction("modify_hand_cost", "owner", amount=-2, duration="this_turn"),
            state, EffectContext("p1"),
        )
        self.assertEqual(len(state.players["p1"].cost_modifiers), 1)
        self.assertIsNotNone(state.players["p1"].cost_modifiers[0]["expires_turn"])
        NativeRuleEngine.revert_temporary(state, "p1", state.turn_number)
        self.assertEqual(state.players["p1"].cost_modifiers, [])

    def test_temporary_set_attack_reverts(self) -> None:
        state = self._state(p2_units=(("e1", "greif", 4, 1),))
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(
            RuleAction("set_attack", "selected_target", amount=0, duration="this_turn"),
            state, EffectContext("p1", target_unit_id="e1"),
        )
        enemy = next(u for u in state.players["p2"].units if u.instance_id == "e1")
        self.assertEqual(enemy.attack, 0)
        NativeRuleEngine.revert_temporary(state, "p1", state.turn_number)
        self.assertEqual(enemy.attack, 4)  # restored

    def test_expiry_ignores_unit_that_left_play(self) -> None:
        state = self._state(p1_units=(("u1", "greif", 3, 1),))
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(
            RuleAction("modify_attack", "friendly_units", amount=2, duration="this_turn"),
            state, EffectContext("p1"),
        )
        state.players["p1"].units.clear()
        NativeRuleEngine.revert_temporary(state, "p1", state.turn_number)
        self.assertEqual(state.players["p1"].temporary_effects, [])


if __name__ == "__main__":
    unittest.main()
