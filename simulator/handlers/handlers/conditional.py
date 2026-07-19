from simulator.effects.resolver import EffectContext, EffectResolver
from simulator.handlers.base_handler import BaseHandler, HandlerContext


class ConditionalHandler(BaseHandler):
    def execute(self, state, cards, context):
        condition = context.parameters.get("condition", {})
        met = _met(condition, state, context)
        selected = context.parameters.get("if_true" if met else "if_false")
        if isinstance(selected, dict):
            EffectResolver(cards).resolve(selected, state, EffectContext(context.player_id, context.source_card_id, context.source_unit_id, context.target_unit_id))


def _met(condition, state, context):
    if not condition:
        return True
    if condition.get("type") == "kredits_at_least":
        return state.players[context.player_id].resources.kredits >= int(condition.get("amount", 0))
    if condition.get("type") == "friendly_unit_exists":
        return bool(state.players[context.player_id].units)
    return False
