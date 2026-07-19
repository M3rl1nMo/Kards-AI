from simulator.actions.validator import find_unit
from simulator.handlers.base_handler import BaseHandler, HandlerContext


class TransformHandler(BaseHandler):
    def execute(self, state, cards, context):
        target_id = context.target_unit_id or context.source_unit_id
        card_id = str(context.parameters.get("card_id", ""))
        if not target_id or card_id not in cards or not cards.get(card_id).is_unit:
            state.event_log.append({"event": "transform_failed", "card_id": card_id})
            return
        target = find_unit(state, target_id)
        card = cards.get(card_id)
        target.card_id, target.attack, target.defense = card_id, card.attack or 0, card.defense or 0
        target.modifiers.clear()
        target.status.clear()
