"""Information lifecycle helpers for the public player-view model."""

from __future__ import annotations

from collections import Counter
import random


def reveal(state, observer_id: str, target_id: str, amount: int) -> list[str]:
    """Reveal a deterministic-random sample of the target's current hand."""
    hand = state.players[target_id].hand
    rng = random.Random((state.rng_seed or 0) + len(state.event_log))
    revealed = rng.sample(hand, min(max(0, amount), len(hand)))
    known = state.players[observer_id].status.setdefault("known_enemy_hand", [])
    known.extend(revealed)
    refresh(state)
    return revealed


def refresh(state) -> None:
    """Drop knowledge for cards no longer in the observed enemy hand.

    Card ids are catalogue ids rather than physical-card ids, so multiplicity
    is retained and reconciled with Counter instead of using a set.
    """
    player_ids = tuple(state.players)
    if len(player_ids) != 2:
        return
    for observer_id in player_ids:
        target_id = next(pid for pid in player_ids if pid != observer_id)
        known = state.players[observer_id].status.get("known_enemy_hand")
        if not isinstance(known, list):
            continue
        available = Counter(state.players[target_id].hand)
        retained = []
        for card_id in known:
            if available[card_id]:
                retained.append(card_id)
                available[card_id] -= 1
        state.players[observer_id].status["known_enemy_hand"] = retained
