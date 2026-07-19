"""Registry of reusable rule-layer operations."""

from __future__ import annotations

from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState
from simulator.handlers.base_handler import BaseHandler, HandlerContext
from simulator.handlers.handlers.choice import ChoiceHandler
from simulator.handlers.handlers.conditional import ConditionalHandler
from simulator.handlers.handlers.copy import CopyHandler
from simulator.handlers.handlers.deploy import DeployHandler
from simulator.handlers.handlers.random import RandomHandler
from simulator.handlers.handlers.transform import TransformHandler


class HandlerRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, BaseHandler] = {}

    @classmethod
    def with_standard_handlers(cls) -> "HandlerRegistry":
        registry = cls()
        registry.register("DEPLOY", DeployHandler())
        registry.register("RANDOM", RandomHandler())
        registry.register("CHOICE", ChoiceHandler())
        registry.register("TRANSFORM", TransformHandler())
        registry.register("COPY", CopyHandler())
        registry.register("CONDITIONAL", ConditionalHandler())
        return registry

    def register(self, handler_id: str, handler: BaseHandler) -> None:
        if not handler_id:
            raise ValueError("Handler ID cannot be empty")
        self._handlers[handler_id] = handler

    def get(self, handler_id: str) -> BaseHandler | None:
        return self._handlers.get(handler_id)

    def execute(self, handler_id: str, state: GameState, cards: CardDatabase, context: HandlerContext) -> bool:
        handler = self.get(handler_id)
        if handler is None:
            state.event_log.append({"event": "unimplemented_handler", "handler_id": handler_id, "card_id": context.source_card_id})
            return False
        handler.execute(state, cards, context)
        state.event_log.append({"event": "handler_executed", "handler_id": handler_id, "card_id": context.source_card_id})
        return True

    @property
    def handler_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))
