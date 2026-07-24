"""Fixed-shape, player-relative observation tensor encoder."""
from __future__ import annotations

import torch
import hashlib

from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, UnitState

MAX_HAND = 12
MAX_UNITS_PER_PLAYER = 8
CARD_FEATURE_DIM = 8
STATE_DIM = 2 + 8 + (2 * MAX_UNITS_PER_PLAYER * CARD_FEATURE_DIM) + (MAX_HAND * CARD_FEATURE_DIM)
_TYPES = ("infantry", "fighter", "tank", "bomber", "artillery", "order", "countermeasure")


class ObservationEncoder:
    """Encodes only information observable by ``player_id``; no enemy cards leak."""
    def __init__(self, cards: CardDatabase) -> None:
        self.cards = cards

    def encode(self, state: GameState, player_id: str) -> torch.Tensor:
        enemy_id = "p2" if player_id == "p1" else "p1"
        own, enemy = state.players[player_id], state.players[enemy_id]
        values: list[float] = [min(state.turn_number, 30) / 30.0, float(state.current_player == player_id)]
        for player in (own, enemy):
            values.extend([player.hq.current_health / 20.0, player.hq.defense_modifier / 30.0,
                           player.resources.kredits / 12.0, player.resources.max_kredits / 12.0])
        for player in (own, enemy):
            for unit in list(player.units)[:MAX_UNITS_PER_PLAYER]:
                values.extend(self._unit_features(unit))
            values.extend([0.0] * (CARD_FEATURE_DIM * (MAX_UNITS_PER_PLAYER - len(player.units[:MAX_UNITS_PER_PLAYER]))))
        for card_id in own.hand[:MAX_HAND]:
            values.extend(self._card_features(card_id))
        values.extend([0.0] * (CARD_FEATURE_DIM * (MAX_HAND - len(own.hand[:MAX_HAND]))))
        if len(values) != STATE_DIM:
            raise AssertionError("Observation encoder shape drift")
        return torch.tensor(values, dtype=torch.float32)

    def _unit_features(self, unit: UnitState) -> list[float]:
        card = self.cards.get(unit.card_id)
        features = self._card_features(unit.card_id)
        features[1] = max(-1.0, min(2.0, unit.attack / 20.0))
        features[2] = max(-1.0, min(2.0, unit.defense / 20.0))
        features[6] = float(unit.position == "frontline")
        features[7] = min(1.0, len(unit.status) / 8.0)
        return features

    def _card_features(self, card_id: str) -> list[float]:
        card = self.cards.get(card_id)
        return [(_TYPES.index(card.type) + 1) / len(_TYPES), (card.attack or 0) / 20.0,
                (card.defense or 0) / 20.0, card.kredits / 12.0, (card.operationCost or 0) / 12.0,
                min(1.0, len(card.abilities) / 5.0), 0.0,
                int.from_bytes(hashlib.blake2b(card.id.encode(), digest_size=2).digest(), "big") / 65535.0]
