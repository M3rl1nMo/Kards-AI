from simulator.core.state import UnitState
from simulator.handlers.base_handler import BaseHandler, HandlerContext


class DeployHandler(BaseHandler):
    def execute(self, state, cards, context):
        card_id = str(context.parameters.get("card_id", context.source_card_id or ""))
        position = str(context.parameters.get("position", "support_line"))
        if card_id not in cards or not cards.get(card_id).is_unit or position not in state.battlefield:
            state.event_log.append({"event": "custom_deploy_failed", "card_id": card_id})
            return
        player = state.players[context.player_id]
        if context.parameters.get("from_hand") and card_id in player.hand:
            player.hand.remove(card_id)
        card = cards.get(card_id)
        instance_id = "{0}-custom-{1}".format(card_id, sum(len(p.units) for p in state.players.values()) + 1)
        player.units.append(UnitState(instance_id, card_id, card.attack or 0, card.defense or 0, context.player_id, position))
        state.battlefield[position].append(instance_id)
