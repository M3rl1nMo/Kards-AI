import random

from simulator.effects.resolver import EffectContext, EffectResolver
from simulator.handlers.base_handler import BaseHandler, HandlerContext


class RandomHandler(BaseHandler):
    def execute(self, state, cards, context):
        options = list(context.parameters.get("options", ()))
        if not options:
            return
        index = context.parameters.get("choice_index")
        selected = options[int(index)] if index is not None else random.Random(context.parameters.get("seed")).choice(options)
        if isinstance(selected, dict):
            EffectResolver(cards).resolve(selected, state, EffectContext(context.player_id, context.source_card_id, context.source_unit_id, context.target_unit_id))
