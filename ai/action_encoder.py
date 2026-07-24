"""Fixed candidate-action tensors and legal-action masks."""
from __future__ import annotations

from dataclasses import fields
import hashlib
from typing import Sequence

import torch

from simulator.actions.action import Action, AttackAction, MoveUnitAction, MulliganAction, PassAction, PlayCardAction, UseAbilityAction

ACTION_FEATURE_DIM = 16
MAX_ACTIONS = 512
_TYPE = {PassAction: 0, PlayCardAction: 1, AttackAction: 2, MoveUnitAction: 3, UseAbilityAction: 4, MulliganAction: 5}


def action_key(action: Action) -> tuple:
    """Stable, serializable identity used to map MCTS/network slots to actions."""
    return (type(action).__name__,) + tuple(getattr(action, item.name) for item in fields(action))


class ActionEncoder:
    def __init__(self, max_actions: int = MAX_ACTIONS) -> None:
        self.max_actions = max_actions

    def encode(self, action: Action) -> torch.Tensor:
        x = torch.zeros(ACTION_FEATURE_DIM, dtype=torch.float32)
        x[_TYPE.get(type(action), 15)] = 1.0
        # Compact identifiers retain action/card/target distinctions without
        # making a network output depend on a global, ever-changing action ID.
        values = action_key(action)[1:]
        for index, value in enumerate(values[:5]):
            if value is not None:
                digest = hashlib.blake2b(str(value).encode(), digest_size=2).digest()
                x[6 + index * 2] = int.from_bytes(digest, "big") / 65535.0
                x[7 + index * 2] = min(1.0, len(str(value)) / 32.0)
        return x

    def encode_legal_actions(self, actions: Sequence[Action]) -> tuple[torch.Tensor, torch.Tensor]:
        if not actions:
            raise ValueError("A non-terminal state must expose at least one legal action")
        if len(actions) > self.max_actions:
            raise ValueError("Legal action count exceeds configured MAX_ACTIONS")
        features = torch.zeros((self.max_actions, ACTION_FEATURE_DIM), dtype=torch.float32)
        mask = torch.zeros(self.max_actions, dtype=torch.bool)
        for index, action in enumerate(actions):
            features[index] = self.encode(action)
            mask[index] = True
        return features, mask

    @staticmethod
    def action_at(actions: Sequence[Action], index: int) -> Action:
        if index < 0 or index >= len(actions):
            raise ValueError("Network selected an illegal padded action slot")
        return actions[index]
