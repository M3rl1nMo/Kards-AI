"""Focused regression tests for Milestone 5: countermeasure parsing.

Covers the 'Counter an order ...' / 'counter it' / 'cancel the effect' parser
templates (which drive CountermeasureResolver into flagging the enemy action as
cancelled) plus the generic countermeasure phrasings that map to existing AST
kinds: 'suppress it', 'destroy it', 'return it to hand', 'give it -N attack',
'enemy discards a random card', and 'it becomes A/D'.
"""

from pathlib import Path
import unittest

from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.effects.resolver import EffectContext
from simulator.rules.native import NativeRuleEngine
from simulator.rules.parser import RuleAction, RuleParser
from simulator.countermeasures import CountermeasureResolver
from simulator.actions.action import PlayCardAction

ROOT = Path(__file__).parents[1]


class CountermeasureMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
        cls.parser = RuleParser()

    def _state(self, p1_units=(), p2_units=(), p1_hand=(), p2_hand=(), p2_active=()):
        return GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Germany", deck=["garrison"], hand=list(p1_hand),
                                  resources=ResourceState(10, 10),
                                  units=[UnitState(uid, cid, atk, dfn, "p1", "support_line")
                                         for uid, cid, atk, dfn in p1_units]),
                "p2": PlayerState("p2", "USA", deck=["garrison"], hand=list(p2_hand),
                                  resources=ResourceState(10, 10),
                                  active_countermeasures=list(p2_active),
                                  units=[UnitState(uid, cid, atk, dfn, "p2", "support_line")
                                         for uid, cid, atk, dfn in p2_units]),
            },
        )

    # --- Parser: the 'Counter an order' cancel family ---

    def test_parser_interception_cancel(self) -> None:
        rule = self.parser.parse(self.cards.get("interception"))
        # cancel is now genuinely executed.
        self.assertEqual(rule.status, "implemented")
        self.assertIn("on_command_played", rule.triggers)
        self.assertTrue(any(a.kind == "cancel" for a in rule.actions))

    def test_parser_secret_operatives_cancel_plus_defense(self) -> None:
        rule = self.parser.parse(self.cards.get("secret_operatives"))
        # cancel is now genuinely executed.
        self.assertEqual(rule.status, "implemented")
        self.assertTrue(any(a.kind == "cancel" for a in rule.actions))
        self.assertTrue(any(a.kind == "modify_defense" and a.amount == 2 for a in rule.actions))

    def test_parser_close_call_cancel_both_triggers(self) -> None:
        rule = self.parser.parse(self.cards.get("close_call"))
        self.assertEqual(rule.status, "implemented")
        self.assertIn("on_command_played", rule.triggers)
        self.assertIn("on_deploy", rule.triggers)
        self.assertTrue(any(a.kind == "cancel" for a in rule.actions))

    def test_parser_lure_counter_it(self) -> None:
        rule = self.parser.parse(self.cards.get("lure"))
        self.assertEqual(rule.status, "implemented")
        self.assertIn("on_command_played", rule.triggers)
        self.assertTrue(any(a.kind == "cancel" for a in rule.actions))

    def test_parser_dowding_system_min_cost(self) -> None:
        rule = self.parser.parse(self.cards.get("dowding_system"))
        self.assertEqual(rule.status, "implemented")
        self.assertTrue(any(a.kind == "cancel" for a in rule.actions))

    def test_parser_ultra_cancel_and_draw(self) -> None:
        rule = self.parser.parse(self.cards.get("ultra"))
        self.assertEqual(rule.status, "implemented")
        self.assertTrue(any(a.kind == "cancel" for a in rule.actions))
        self.assertTrue(any(a.kind == "draw" and a.amount == 1 for a in rule.actions))

    def test_parser_evasive_action_cancel_effect(self) -> None:
        rule = self.parser.parse(self.cards.get("evasive_action"))
        # The trigger clause is descriptive; the cancel action is captured.
        self.assertTrue(any(a.kind == "cancel" for a in rule.actions))
        self.assertNotEqual(rule.status, "unresolved")

    # --- Parser: generic countermeasure phrasings ---

    def test_parser_hit_the_drop_point_suppress_it(self) -> None:
        rule = self.parser.parse(self.cards.get("hit_the_drop_point"))
        self.assertTrue(any(a.kind == "suppress" and a.target == "selected_target" for a in rule.actions))

    def test_c6n_saiun_prevents_active_countermeasures_from_triggering(self) -> None:
        state = self._state(
            p2_units=(("saiun", "c6n_saiun", 1, 1),), p2_hand=("interception",),
            p2_active=({"card_id": "interception", "activated_turn": 1},),
        )
        engine = NativeRuleEngine(self.cards)
        engine.execute("c6n_saiun", "on_deploy", state, EffectContext("p2", "c6n_saiun", "saiun"))
        cancelled = CountermeasureResolver.intercept(
            state, self.cards, "p1", "on_command_played", EffectContext("p1", "critical_damage"),
        )
        self.assertFalse(cancelled)
        self.assertEqual(len(state.players["p2"].active_countermeasures), 1)
        self.assertTrue(any(e["event"] == "countermeasure_trigger_blocked" for e in state.event_log))

    def test_c6n_saiun_blocks_countermeasures_globally(self) -> None:
        state = self._state(
            p1_units=(("saiun", "c6n_saiun", 1, 1),), p2_hand=("interception",),
            p2_active=({"card_id": "interception", "activated_turn": 1},),
        )
        NativeRuleEngine(self.cards).execute("c6n_saiun", "on_deploy", state, EffectContext("p1", "c6n_saiun", "saiun"))
        self.assertFalse(CountermeasureResolver.intercept(
            state, self.cards, "p1", "on_command_played", EffectContext("p1", "critical_damage"),
        ))

    def test_parser_lost_convoy_negative_attack(self) -> None:
        rule = self.parser.parse(self.cards.get("lost_convoy"))
        self.assertEqual(rule.status, "implemented")
        self.assertTrue(any(a.kind == "modify_attack" and a.amount == -2 for a in rule.actions))

    def test_parser_airstrike_return_and_discard(self) -> None:
        rule = self.parser.parse(self.cards.get("airstrike"))
        self.assertEqual(rule.status, "implemented")
        self.assertTrue(any(a.kind == "return_to_hand" for a in rule.actions))
        self.assertTrue(any(a.kind == "discard" and a.target == "enemy_hand" for a in rule.actions))

    def test_parser_sniped_becomes_stats(self) -> None:
        rule = self.parser.parse(self.cards.get("sniped"))
        self.assertEqual(rule.status, "implemented")
        self.assertTrue(any(a.kind == "set_attack" and a.amount == 2 for a in rule.actions))
        self.assertTrue(any(a.kind == "set_defense" and a.amount == 2 for a in rule.actions))

    def test_parser_night_hunters_destroy_it(self) -> None:
        rule = self.parser.parse(self.cards.get("night_hunters"))
        self.assertEqual(rule.status, "implemented")
        self.assertTrue(any(a.kind == "destroy" for a in rule.actions))

    # --- Native: interception actually cancels ---

    def test_native_interception_cancels_order(self) -> None:
        state = self._state(
            p2_units=(("friendly", "german_infantry", 2, 2),),
            p2_active=[{"card_id": "interception", "activated_turn": 0}],
        )
        ctx = EffectContext("p1", "home_guard", target_unit_id="friendly")
        cancelled = CountermeasureResolver.intercept(state, self.cards, "p1", "on_command_played", ctx)
        self.assertTrue(cancelled)
        self.assertFalse(state.players["p2"].active_countermeasures)
        self.assertTrue(any(e.get("event") == "countermeasure_triggered" for e in state.event_log))

    def test_interception_does_not_consume_for_enemy_target(self) -> None:
        state = self._state(
            p1_units=(("enemy", "german_infantry", 2, 2),),
            p2_active=[{"card_id": "interception", "activated_turn": 0}],
        )
        cancelled = CountermeasureResolver.intercept(
            state, self.cards, "p1", "on_command_played",
            EffectContext("p1", "home_guard", target_unit_id="enemy"),
        )
        self.assertFalse(cancelled)
        self.assertEqual(len(state.players["p2"].active_countermeasures), 1)

    def test_dowding_system_respects_intercepted_order_cost(self) -> None:
        active = [{"card_id": "dowding_system", "activated_turn": 0}]
        low = self._state(p2_active=active)
        self.assertFalse(CountermeasureResolver.intercept(
            low, self.cards, "p1", "on_command_played", EffectContext("p1", "radar"),
        ))
        self.assertEqual(len(low.players["p2"].active_countermeasures), 1)
        high_card = next(card.id for card in self.cards if card.type == "order" and card.kredits >= 4)
        high = self._state(p2_active=active)
        self.assertTrue(CountermeasureResolver.intercept(
            high, self.cards, "p1", "on_command_played", EffectContext("p1", high_card),
        ))
        self.assertFalse(high.players["p2"].active_countermeasures)

    def test_native_mismatch_trigger_does_not_cancel(self) -> None:
        # against_the_odds triggers on_attack, not on_command_played.
        state = self._state(p2_active=[{"card_id": "against_the_odds", "activated_turn": 0}])
        ctx = EffectContext("p1", "some_order", target_unit_id="dummy")
        cancelled = CountermeasureResolver.intercept(state, self.cards, "p1", "on_command_played", ctx)
        self.assertFalse(cancelled)

    def test_evasive_action_cancels_deployment_effect_but_not_unit(self) -> None:
        # 40th Cavalry's deployment effect removes a kredit slot.  Evasive
        # Action must leave the unit in play while preventing that mutation.
        state = self._state(
            p1_hand=("40th_cavalry_regiment",),
            p2_active=[{"card_id": "evasive_action", "activated_turn": 0}],
        )
        PlayCardAction("p1", "40th_cavalry_regiment").execute(state, self.cards)
        self.assertEqual(len(state.players["p1"].units), 1)
        self.assertEqual(state.players["p1"].resources.max_kredits, 10)
        self.assertFalse(state.players["p2"].active_countermeasures)
        self.assertTrue(any(entry.get("event") == "deployment_effect_cancelled" for entry in state.event_log))

    # --- Native: generic countermeasure actions ---

    def test_native_sniped_sets_stats(self) -> None:
        state = self._state(p1_units=(("u1", "german_infantry", 5, 5),))
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(
            RuleAction("set_attack", "selected_target", amount=2),
            state, EffectContext("p1", target_unit_id="u1"),
        )
        engine._execute_action(
            RuleAction("set_defense", "selected_target", amount=2),
            state, EffectContext("p1", target_unit_id="u1"),
        )
        unit = next(u for u in state.players["p1"].units if u.instance_id == "u1")
        self.assertEqual((unit.attack, unit.defense), (2, 2))

    def test_native_lost_convoy_applies_negative_attack(self) -> None:
        state = self._state(p1_units=(("u1", "german_infantry", 5, 5),))
        engine = NativeRuleEngine(self.cards)
        engine._execute_action(
            RuleAction("modify_attack", "selected_target", amount=-2),
            state, EffectContext("p1", target_unit_id="u1"),
        )
        unit = next(u for u in state.players["p1"].units if u.instance_id == "u1")
        self.assertEqual(unit.attack, 3)


if __name__ == "__main__":
    unittest.main()
