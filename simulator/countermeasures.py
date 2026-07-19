"""Activated-countermeasure dispatcher for native card rules."""
from simulator.effects.resolver import EffectContext
from simulator.actions.validator import opponent_id
from simulator.rules.native import engine_for
from simulator.rules.condition import evaluate as condition_evaluate


class CountermeasureResolver:
    @staticmethod
    def intercept(state, cards, acting_player_id, event, context):
        """Try to counter the current action with the opponent's active countermeasures.

        Returns True if the action was cancelled (at least one countermeasure
        fired and had a `cancel` action with its conditions satisfied).

        Conditions (e.g. 'target must be friendly', 'order cost >= 4') are
        evaluated BEFORE the countermeasure is consumed.  If the condition
        fails, the countermeasure stays active and the action proceeds normally.
        """
        enemy = opponent_id(state, acting_player_id)
        cancelled = False
        # Iterate over a snapshot so modifications from execute() do not skew
        # the loop.
        for counter in list(state.players[enemy].active_countermeasures):
            card_id = counter.get("card_id")
            if not isinstance(card_id, str) or card_id not in cards:
                continue
            rule = engine_for(cards).rule_for(card_id)
            if event not in rule.triggers:
                continue

            # Check that at least one action would actually fire (conditions
            # satisfied) BEFORE consuming the countermeasure.
            should_activate = False
            counter_context = EffectContext(
                enemy, card_id,
                target_unit_id=context.source_unit_id,
                event=event,
            )
            for action in rule.actions:
                if action.condition is None:
                    should_activate = True
                    break
                if condition_evaluate(
                    action.condition, event, counter_context, cards, state,
                ):
                    should_activate = True
                    break

            if not should_activate:
                continue

            # Condition passed: activate the countermeasure.
            engine_for(cards).execute(card_id, event, state, counter_context)
            state.players[enemy].active_countermeasures.remove(counter)
            if card_id in state.players[enemy].hand:
                state.players[enemy].hand.remove(card_id)
            state.graveyard.setdefault(enemy, []).append(card_id)
            state.event_log.append({
                "event": "countermeasure_triggered",
                "player_id": enemy,
                "counter_id": card_id,
                "trigger": event,
            })
            if any(action.kind == "cancel" for action in rule.actions):
                cancelled = True

        return cancelled
