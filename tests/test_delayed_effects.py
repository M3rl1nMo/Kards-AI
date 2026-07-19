"""M10: delayed one-shot effects (scheduled discard / return-to-hand) + end_turn.

These cover the new scheduling subsystem in NativeRuleEngine and the turn-loop
hooks in TurnManager. The hard part of delayed effects is *when* they fire
(correct turn phase); the *what* (which unit) is resolved via the action's
target/source unit id captured at schedule time.
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


def _ctx(player_id="p1", source_card_id="x", source_unit_id=None, target_unit_id=None, event="on_play"):
    return EffectContext(
        player_id=player_id,
        source_card_id=source_card_id,
        source_unit_id=source_unit_id,
        target_unit_id=target_unit_id,
        event=event,
        metadata={},
    )


class DelayedEffectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
        cls.engine = NativeRuleEngine(cls.cards)

    def state(self, with_unit=True) -> GameState:
        players = {
            "p1": PlayerState("p1", "Britain", deck=["garrison"], hand=[], resources=ResourceState(5, 5)),
            "p2": PlayerState("p2", "Germany", units=[], resources=ResourceState(5, 5)),
        }
        battlefield = {"frontline": [], "support_line": []}
        if with_unit:
            players["p1"].units.append(UnitState("u1", "hurricane_mk_i", 2, 3, "p1", "frontline"))
            battlefield["frontline"].append("u1")
        return GameState(
            current_player="p1",
            players=players,
            battlefield=battlefield,
            graveyard={"p1": [], "p2": []},
        )

    def test_delayed_return_schedules_then_fires_at_next_turn_start(self) -> None:
        state = self.state()
        ctx = _ctx(source_unit_id="u1")
        # A unit that schedules "Return it at the start of your next turn".
        self.engine._schedule_delayed(
            RuleAction("delayed_return", "owner", duration="start_of_next_turn"), state, ctx
        )
        self.assertEqual(len(state.players["p1"].scheduled), 1)
        # Unit still on battlefield before the trigger fires.
        self.assertIn("u1", [u.instance_id for u in state.players["p1"].units])
        # Start of the player's next turn -> the unit returns to hand.
        NativeRuleEngine.fire_scheduled(state, "p1", "start")
        self.assertNotIn("u1", [u.instance_id for u in state.players["p1"].units])
        self.assertIn("hurricane_mk_i", state.players["p1"].hand)

    def test_delayed_discard_schedules_then_discards_unit_at_turn_end(self) -> None:
        state = self.state()
        ctx = _ctx(source_unit_id="u1")
        self.engine._schedule_delayed(
            RuleAction("delayed_discard", "owner", duration="end_of_turn"), state, ctx
        )
        NativeRuleEngine.fire_scheduled(state, "p1", "end")
        self.assertNotIn("u1", [u.instance_id for u in state.players["p1"].units])
        self.assertIn("hurricane_mk_i", state.graveyard["p1"])

    def test_end_of_next_turn_waits_one_extra_end(self) -> None:
        state = self.state()
        ctx = _ctx(source_unit_id="u1")
        self.engine._schedule_delayed(
            RuleAction("delayed_discard", "owner", duration="end_of_next_turn"), state, ctx
        )
        # First end_turn is the same turn it was scheduled -> must NOT fire yet.
        NativeRuleEngine.fire_scheduled(state, "p1", "end")
        self.assertIn("u1", [u.instance_id for u in state.players["p1"].units])
        # Second end_turn (the player's "next turn" end) -> fires.
        NativeRuleEngine.fire_scheduled(state, "p1", "end")
        self.assertNotIn("u1", [u.instance_id for u in state.players["p1"].units])

    def test_isolation_removes_then_returns_enemy_unit_next_turn(self) -> None:
        # Integration: "Remove target unit... Return it at the start of your next
        # turn." The removed (enemy) unit must come back to the enemy's hand.
        state = self.state(with_unit=False)
        state.players["p2"].units.append(UnitState("victim", "hurricane_mk_i", 2, 3, "p2", "frontline"))
        state.battlefield["frontline"].append("victim")
        state.players["p1"].hand.append("isolation")
        state.players["p1"].resources = ResourceState(5, 5)
        PlayCardAction("p1", "isolation", target_unit_id="victim").execute(state, self.cards)
        # Victim removed from the battlefield and scheduled for return.
        self.assertNotIn("victim", [u.instance_id for u in state.players["p2"].units])
        self.assertEqual(len(state.players["p1"].scheduled), 1)
        # Advance to the start of p1's next turn (their "your next turn").
        NativeRuleEngine.fire_scheduled(state, "p1", "start")
        self.assertIn("hurricane_mk_i", state.players["p2"].hand)
        self.assertNotIn("victim", [u.instance_id for u in state.players["p2"].units])

    def test_end_turn_action_passes_the_turn(self) -> None:
        state = self.state(with_unit=False)
        state.players["p1"].hand.append("calm_before_the_storm")
        state.players["p1"].resources = ResourceState(5, 5)
        before = state.current_player
        PlayCardAction("p1", "calm_before_the_storm").execute(state, self.cards)
        # The card forces an immediate turn pass.
        self.assertNotEqual(state.current_player, before)


if __name__ == "__main__":
    unittest.main()
