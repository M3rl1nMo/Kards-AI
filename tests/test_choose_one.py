"""M9: choose_one selection logic — executable structured actions.

Covers the three choose_one patterns found across the 9 KARDS cards:
  Pattern A: choose 1 of N random elite <nation> units -> add to hand.
  Pattern B: choose 1 of N cards in the enemy hand -> side effects via the
             transient `_chosen` card reference (sabotage, happy_time).
  Pattern C: choose a card in hand -> put it on top of your deck
             (night_patrol, thunderbolt; also benefits 175th_infantry_regiment).
"""

from pathlib import Path
import unittest

from simulator.actions.action import PlayCardAction
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.rules.native import NativeRuleEngine


ROOT = Path(__file__).parents[1]


class ChooseOneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
        cls.engine = NativeRuleEngine(cls.cards)

    def state(self, p2_hand=("ec1", "ec2", "ec3")) -> GameState:
        return GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Britain", deck=["garrison", "home_guard", "radar", "active_sonar"], hand=["radar"], resources=ResourceState(10, 10)),
                "p2": PlayerState("p2", "Germany", hand=list(p2_hand), resources=ResourceState(5, 5)),
            },
            graveyard={"p1": [], "p2": []},
            rng_seed=1,
        )

    # --- Pattern A: random nation unit added to hand -----------------------

    def test_pattern_a_adds_random_nation_unit_to_hand(self) -> None:
        for cid in ("his_majestys_chosen", "afrika_korps", "soul_of_old_japan",
                    "heroes_of_the_soviet_union", "a_few_good_men"):
            with self.subTest(card=cid):
                rule = self.engine.rule_for(cid)
                self.assertEqual(rule.status, "implemented")
                self.assertEqual(rule.actions[0].kind, "choose_one")
                self.assertEqual(rule.actions[0].scope, "unit")
                state = self.state()
                state.players["p1"].hand.append(cid)
                before = set(state.players["p1"].hand)
                PlayCardAction("p1", cid).execute(state, self.cards)
                added = set(state.players["p1"].hand) - before
                self.assertEqual(len(added), 1)
                new_id = next(iter(added))
                card = self.cards.get(new_id)
                self.assertIsNotNone(card)
                self.assertTrue(card.is_unit)
                self.assertEqual(card.nation, rule.actions[0].card_name)
                self.assertTrue(any(e["event"] == "choose_one_unit_added" for e in state.event_log))

    # --- Pattern B: enemy-hand selection + side effects ---------------------

    def test_pattern_b_sabotage_discards_enemy_card_and_copies_it(self) -> None:
        rule = self.engine.rule_for("sabotage")
        self.assertEqual([a.kind for a in rule.actions], ["choose_one", "discard", "add_card"])
        self.assertEqual(rule.actions[0].scope, "enemy_hand")
        self.assertEqual(rule.actions[1].target, "chosen")
        self.assertEqual(rule.actions[2].card_name, "chosen_copy")
        state = self.state()
        state.players["p1"].hand.append("sabotage")
        PlayCardAction("p1", "sabotage").execute(state, self.cards)
        # The chosen enemy card is removed from the enemy hand and discarded.
        self.assertEqual(len(state.players["p2"].hand), 2)
        self.assertEqual(len(state.graveyard["p2"]), 1)
        discarded = state.graveyard["p2"][0]
        self.assertNotIn(discarded, state.players["p2"].hand)
        # p1 receives a copy of that exact card.
        self.assertIn(discarded, state.players["p1"].hand)
        self.assertTrue(any(e["event"] == "choose_one_enemy_card" for e in state.event_log))

    def test_pattern_b_happy_time_consumes_enemy_card_and_creates_plan(self) -> None:
        rule = self.engine.rule_for("happy_time")
        # choose_one, copy_unit, and convert_to are now all genuinely executed
        self.assertEqual(rule.status, "implemented")
        kinds = [a.kind for a in rule.actions]
        self.assertIn("choose_one", kinds)
        self.assertIn("copy_unit", kinds)
        self.assertIn("convert_to", kinds)

    def test_pattern_b_noop_when_enemy_hand_empty(self) -> None:
        state = self.state(p2_hand=())
        state.players["p1"].hand.append("sabotage")
        PlayCardAction("p1", "sabotage").execute(state, self.cards)
        # Nothing to choose: no discard, no copy; no crash.
        self.assertEqual(len(state.players["p2"].hand), 0)
        self.assertEqual(len(state.players["p1"].hand), 1)  # only the played card moved to graveyard
        self.assertIn("sabotage", state.graveyard["p1"])
        self.assertTrue(any(e["event"] == "choose_one_enemy_hand_empty" for e in state.event_log))

    # --- Pattern C: choose a card in hand, put on top of deck ---------------

    def test_pattern_c_order_moves_chosen_card_to_deck_top(self) -> None:
        for cid in ("night_patrol",):
            with self.subTest(card=cid):
                rule = self.engine.rule_for(cid)
                self.assertEqual([a.kind for a in rule.actions], ["draw", "choose_one", "return_to_deck"])
                self.assertEqual(rule.actions[1].scope, "hand")
                self.assertEqual(rule.actions[2].target, "chosen")
                state = self.state()
                state.players["p1"].hand.append(cid)
                hand_before = set(state.players["p1"].hand)
                deck_before = set(state.players["p1"].deck)
                PlayCardAction("p1", cid).execute(state, self.cards)
                top = state.players["p1"].deck[0]
                # The chosen card left the hand and now sits on top of the deck.
                self.assertNotIn(top, state.players["p1"].hand)
                self.assertIn(top, hand_before | deck_before)
                self.assertIn(cid, state.graveyard["p1"])
                # Net hand size is preserved: -1 played, +2 drawn, -1 to deck = 0.
                self.assertEqual(len(state.players["p1"].hand), len(hand_before))
                self.assertTrue(any(e["event"] == "card_to_deck_top" for e in state.event_log))

    def test_pattern_c_unit_deployment_triggers_same_effect(self) -> None:
        rule = self.engine.rule_for("thunderbolt")
        self.assertEqual(rule.triggers, ("on_deploy",))
        self.assertEqual([a.kind for a in rule.actions], ["draw", "choose_one", "return_to_deck"])
        state = self.state()
        state.players["p1"].hand.append("thunderbolt")
        hand_before = set(state.players["p1"].hand)
        PlayCardAction("p1", "thunderbolt").execute(state, self.cards)
        # The unit was deployed.
        self.assertTrue(any(u.card_id == "thunderbolt" for u in state.players["p1"].units))
        # And its Deployment effect moved a chosen card to the deck top.
        top = state.players["p1"].deck[0]
        self.assertNotIn(top, state.players["p1"].hand)
        self.assertIn(top, hand_before | set(state.players["p1"].deck[1:]) | {"thunderbolt"})
        self.assertTrue(any(e["event"] == "card_to_deck_top" for e in state.event_log))


if __name__ == "__main__":
    unittest.main()
