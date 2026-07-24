"""Round-trip guarantees required by branchable AI search."""

from __future__ import annotations

import unittest

from simulator.core.state import (
    GameState,
    Headquarters,
    PlayerState,
    ResourceState,
    UnitState,
)


class StateRoundTripTests(unittest.TestCase):
    def test_clone_isolated_and_equivalent(self) -> None:
        state = GameState(current_player="p1", players={"p1": PlayerState("p1", hand=["radar"]), "p2": PlayerState("p2")}, event_log=[{"event": "x"}], pending_cancels=[{"id": "a"}])
        cloned = state.clone()
        self.assertEqual(cloned.to_dict(), state.to_dict())
        cloned.players["p1"].hand.append("new")
        cloned.event_log[0]["event"] = "changed"
        cloned.pending_cancels[0]["id"] = "changed"
        self.assertEqual(state.players["p1"].hand, ["radar"])
        self.assertEqual(state.event_log[0]["event"], "x")
        self.assertEqual(state.pending_cancels[0]["id"], "a")

    def test_json_round_trip_preserves_all_rule_runtime_state(self) -> None:
        state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState(
                    "p1", "Britain", deck=["radar"], hand=["pursuit"],
                    resources=ResourceState(5, 8),
                    hq=Headquarters(nation="Britain", defense_modifier=3,
                                    active_effects=[{"kind": "shield", "turn": 4}]),
                    units=[UnitState("unit-1", "hurricane_mk_i", 3, 4, "p1",
                                     modifiers=[{"type": "modify_operation_cost", "amount": -1}],
                                     status={"added_abilities": ["blitz"]})],
                    active_countermeasures=[{"card_id": "ultra", "activated_turn": 3}],
                    fatigue_damage=2,
                    cost_modifiers=[{"amount": -1, "scope": "order", "expires_turn": 4}],
                    op_cost_rules=[{"amount": -2, "scope": "type", "filter_value": "bomber", "expires_turn": 3}],
                    temporary_effects=[{"turn": 3, "reverts": [{"unit_id": "unit-1", "attr": "attack", "delta": 2}]}],
                    scheduled=[{"trigger": "end_of_turn", "unit_id": "unit-1", "kind": "discard", "wait": 1}],
                    status={"orders_played": 1},
                ),
                "p2": PlayerState("p2", "Germany"),
            },
            battlefield={"frontline": [], "support_line": ["unit-1"]},
            graveyard={"p1": [], "p2": []},
            event_log=[{"event": "unit_deployed", "unit_id": "unit-1"}],
            mulligan_pending=["p2"],
            rng_seed=123,
            pending_cancels=[{"source_card_id": "ultra", "player_id": "p1"}],
        )

        restored = GameState.from_json(state.to_json())

        self.assertEqual(restored.to_dict(), state.to_dict())
        # Restored nested values must be independent for tree-search branches.
        restored.players["p1"].cost_modifiers[0]["amount"] = -9
        self.assertEqual(state.players["p1"].cost_modifiers[0]["amount"], -1)

