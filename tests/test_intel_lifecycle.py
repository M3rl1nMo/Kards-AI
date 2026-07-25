"""Regression coverage for player-view Intel information lifecycle."""

import unittest

from simulator.core.state import GameState, PlayerState
from simulator.rules.intel import refresh, reveal


class IntelLifecycleTests(unittest.TestCase):
    def test_reveal_is_random_and_knowledge_expires_when_card_leaves_hand(self) -> None:
        state = GameState(
            current_player="p1",
            rng_seed=7,
            players={"p1": PlayerState("p1"), "p2": PlayerState("p2", hand=["a", "b", "c"])},
        )
        revealed = reveal(state, "p1", "p2", 2)
        self.assertEqual(len(revealed), 2)
        self.assertEqual(state.players["p1"].status["known_enemy_hand"], revealed)
        state.players["p2"].hand.remove(revealed[0])
        refresh(state)
        self.assertNotIn(revealed[0], state.players["p1"].status["known_enemy_hand"])
