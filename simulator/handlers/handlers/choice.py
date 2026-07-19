from simulator.effects.resolver import EffectContext, EffectResolver
from simulator.handlers.base_handler import BaseHandler, HandlerContext


class ChoiceHandler(BaseHandler):
    def execute(self, state, cards, context):
        choices = list(context.parameters.get("choices", ()))
        index = context.parameters.get("choice_index")
        if index is None or not 0 <= int(index) < len(choices):
            state.event_log.append({"event": "choice_required", "card_id": context.source_card_id})
            return
        selected = choices[int(index)]
        if isinstance(selected, dict):
            EffectResolver(cards).resolve(selected, state, EffectContext(context.player_id, context.source_card_id, context.source_unit_id, context.target_unit_id))
