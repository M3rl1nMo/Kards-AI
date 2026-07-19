"""Focused regression tests for Milestone 6: keyword / attribute / movement sweep.

Covers the unified 'give' parser (stat buff + ability grant with a descriptor
such as 'friendly Guard unit', 'unit with Guard', 'British infantry', 'Veteran
unit') and the native handlers it drives: grant_ability (with scope filtering),
repair, and move_to_support_line (retreat). Also asserts that conditional
'give' sentences stay unresolved for the temporary-effect milestone (M4).
"""

from pathlib import Path
import unittest

from simulator.actions.action import AttackAction, MoveUnitAction, PlayCardAction
from simulator.actions.validator import ActionValidationError
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.effects.resolver import EffectContext
from simulator.rules.native import NativeRuleEngine
from simulator.rules.parser import RuleAction, RuleParser

ROOT = Path(__file__).parents[1]


class KeywordMovementMechanicsTests(unittest.TestCase):
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

    # --- Parser: unified 'give' ---

    def test_parser_give_stat_and_ability(self) -> None:
        # FINEST HOUR: "Give an air unit +1+1 and Blitz." -> buff(air) + grant blitz.
        rule = self.parser.parse(self.cards.get("finest_hour"))
        self.assertEqual(rule.status, "implemented")
        kinds = [(a.kind, a.scope, a.card_name) for a in rule.actions]
        self.assertIn(("buff", "air", ""), kinds)
        self.assertIn(("grant_ability", "air", "blitz"), kinds)

    def test_parser_give_with_guard_descriptor(self) -> None:
        # RESOLUTE DEFENSE: "Give a unit with Guard +1+3." -> buff scope=guard.
        rule = self.parser.parse(self.cards.get("resolute_defense"))
        self.assertEqual(rule.status, "implemented")
        buff = next(a for a in rule.actions if a.kind == "buff")
        self.assertEqual(buff.scope, "guard")
        self.assertEqual((buff.attack, buff.defense), (1, 3))

    def test_parser_give_friendly_guard_unit_multi_grant(self) -> None:
        # ENTRENCHED: "Give a friendly Guard unit +3 attack, Fury and Salvage."
        rule = self.parser.parse(self.cards.get("entrenched"))
        self.assertEqual(rule.status, "implemented")
        granted = {a.card_name for a in rule.actions if a.kind == "grant_ability"}
        self.assertEqual(granted, {"fury", "salvage"})
        self.assertTrue(any(a.kind == "modify_attack" and a.scope == "guard" for a in rule.actions))

    def test_parser_give_veteran_keyword(self) -> None:
        # BREAKOUT: "Give a Veteran unit Fury and Shock." -> grants fury+shock, scope=veteran.
        rule = self.parser.parse(self.cards.get("breakout"))
        self.assertEqual(rule.status, "implemented")
        self.assertEqual({a.card_name for a in rule.actions if a.kind == "grant_ability"}, {"fury", "shock"})
        self.assertTrue(all(a.scope == "veteran" for a in rule.actions if a.kind == "grant_ability"))

    def test_parser_give_conditional_captures_condition(self) -> None:
        # FIRST RESPONDERS carries a trailing 'if' clause. M7 strips the
        # condition into RuleAction.condition instead of leaving it unresolved
        # (fail-loud is replaced by full parse + condition capture).
        rule = self.parser.parse(self.cards.get("first_responders"))
        self.assertNotEqual(rule.status, "unresolved")
        self.assertTrue(
            any(a.condition for a in rule.actions),
            "conditional give must carry its condition in RuleAction.condition",
        )

    # --- Native: grant_ability ---

    def test_native_grant_ability_appends(self) -> None:
        state = self._state(p1_units=(("u1", "greif", 3, 4),))
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(
            RuleAction("grant_ability", "selected_target", card_name="blitz"),
            state, EffectContext("p1", target_unit_id="u1"),
        )
        unit = next(u for u in state.players["p1"].units if u.instance_id == "u1")
        self.assertEqual(unit.status.get("added_abilities"), ["blitz"])

    def test_native_grant_ability_respects_guard_scope(self) -> None:
        state = self._state(
            p1_units=(("guard_unit", "welsh_guards", 2, 3), ("plain", "10th_para_battalion", 2, 3)),
        )
        engine = NativeRuleEngine(self.cards)
        action = RuleAction("grant_ability", "selected_target", card_name="fury", scope="guard")
        engine._execute_action(action, state, EffectContext("p1", target_unit_id="guard_unit"))
        engine._execute_action(action, state, EffectContext("p1", target_unit_id="plain"))
        gu = next(u for u in state.players["p1"].units if u.instance_id == "guard_unit")
        pl = next(u for u in state.players["p1"].units if u.instance_id == "plain")
        self.assertEqual(gu.status.get("added_abilities"), ["fury"])
        self.assertNotIn("added_abilities", pl.status)

    def test_native_buff_respects_guard_scope(self) -> None:
        # RESOLUTE DEFENSE style: +1+3 only to a Guard unit.
        state = self._state(
            p1_units=(("guard_unit", "welsh_guards", 2, 3), ("plain", "10th_para_battalion", 2, 3)),
        )
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(
            RuleAction("buff", "selected_target", attack=1, defense=3, scope="guard"),
            state, EffectContext("p1", target_unit_id="guard_unit"),
        )
        engine._execute_action(
            RuleAction("buff", "selected_target", attack=1, defense=3, scope="guard"),
            state, EffectContext("p1", target_unit_id="plain"),
        )
        gu = next(u for u in state.players["p1"].units if u.instance_id == "guard_unit")
        pl = next(u for u in state.players["p1"].units if u.instance_id == "plain")
        self.assertEqual((gu.attack, gu.defense), (3, 6))
        self.assertEqual((pl.attack, pl.defense), (2, 3))  # unchanged

    # --- Native: repair ---

    def test_native_repair_restores_defense(self) -> None:
        # Damage a unit then fully repair it back to its printed defense.
        state = self._state(p1_units=(("u1", "matilda_mk_iv", 3, 5),))
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(RuleAction("damage", "selected_target", amount=2),
                               state, EffectContext("p1", target_unit_id="u1"))
        damaged = next(u for u in state.players["p1"].units if u.instance_id == "u1")
        self.assertEqual(damaged.defense, 3)  # 5 - 2
        engine._execute_action(RuleAction("repair", "selected_target"),
                               state, EffectContext("p1", target_unit_id="u1"))
        self.assertEqual(damaged.defense, 5)  # restored to card.defense
        self.assertTrue(any(e.get("event") == "unit_repaired" for e in state.event_log))

    # --- Operation sequencing: move versus attack ---

    def test_regular_unit_cannot_attack_after_moving(self) -> None:
        state = self._state(
            p1_units=(("u1", "greif", 3, 4),),
            p2_units=(("enemy", "m3a3_honey", 2, 3),),
        )
        state.battlefield["support_line"] = ["u1", "enemy"]
        MoveUnitAction("p1", "u1").execute(state, self.cards)
        with self.assertRaises(ActionValidationError):
            AttackAction("p1", "u1", "enemy").validate(state, self.cards)

    def test_blitz_tank_can_move_and_attack_on_deployment_turn(self) -> None:
        state = self._state(
            p1_units=(("u1", "panzer_iia", 2, 2),),
            p2_units=(("enemy", "m3a3_honey", 2, 3),),
        )
        state.battlefield["support_line"] = ["u1", "enemy"]
        unit = state.players["p1"].units[0]
        unit.status["deployed_this_turn"] = True
        MoveUnitAction("p1", "u1").execute(state, self.cards)
        AttackAction("p1", "u1", "enemy").validate(state, self.cards)

    def test_explicit_move_and_attack_rule_overrides_operation_limit(self) -> None:
        state = self._state(
            p1_units=(("u1", "59_panzergrenadier", 2, 3),),
            p2_units=(("enemy", "m3a3_honey", 2, 3),),
        )
        state.battlefield["support_line"] = ["u1", "enemy"]
        engine = NativeRuleEngine(self.cards)
        engine.execute("59_panzergrenadier", "on_deploy", state,
                       EffectContext("p1", "59_panzergrenadier", "u1"))
        MoveUnitAction("p1", "u1").execute(state, self.cards)
        AttackAction("p1", "u1", "enemy").validate(state, self.cards)

    # --- Native: move_to_support_line (retreat) ---

    def test_native_retreat_moves_to_support_line(self) -> None:
        state = self._state(p1_units=(("u1", "greif", 3, 4),))
        u = next(un for un in state.players["p1"].units if un.instance_id == "u1")
        u.position = "frontline"
        state.battlefield["frontline"].append("u1")
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(
            RuleAction("move_to_support_line", "selected_target"),
            state, EffectContext("p1", target_unit_id="u1"),
        )
        self.assertEqual(u.position, "support_line")
        self.assertIn("u1", state.battlefield["support_line"])

    def test_integration_give_via_real_card(self) -> None:
        # WELSH GUARDS: "Deployment: Give another target +1 defense." ends
        # implemented and the native engine applies the defense buff on deploy.
        rule = self.parser.parse(self.cards.get("welsh_guards"))
        self.assertEqual(rule.status, "implemented")
        state = self._state(p1_units=(("u1", "10th_para_battalion", 2, 3),))
        engine = NativeRuleEngine(self.cards)
        engine.execute("welsh_guards", "on_deploy", state,
                       EffectContext("p1", source_card_id="welsh_guards", target_unit_id="u1"))
        target = next(u for u in state.players["p1"].units if u.instance_id == "u1")
        self.assertEqual(target.defense, 4)  # 3 + 1


if __name__ == "__main__":
    unittest.main()
