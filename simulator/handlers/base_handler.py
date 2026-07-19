"""Contracts shared by custom card handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState


@dataclass(frozen=True)
class HandlerContext:
    player_id: str
    source_card_id: str | None = None
    source_unit_id: str | None = None
    target_unit_id: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)


class BaseHandler(ABC):
    """A reusable custom-effect operation, not a per-card implementation."""

    @abstractmethod
    def execute(self, state: GameState, cards: CardDatabase, context: HandlerContext) -> None:
        """Perform the handler operation; callers own validation and dispatch."""
