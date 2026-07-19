"""Evaluate parsed ``RuleAction.condition`` strings against a firing event.

The parser stores normalized condition clauses (e.g. ``when it damages the enemy
hq.``, ``if enemy has 3 or more units.``). The native engine uses this to decide
whether a conditional action should actually execute for the current event/context.

Modeled phrases are strictly gated to the matching event + (where applicable) the
played card's type/ability. Unrecognized conditions are retained as satisfied
for backward-compatible execution until the parser can distinguish persistent
text from unresolved event predicates.
"""

from __future__ import annotations

import re

from simulator.cards.loader import CardDatabase


def evaluate(
    condition: str | None,
    event: str,
    context,
    cards: CardDatabase,
    state=None,
) -> bool:
    """Return True if the action should execute for ``event``/``context``."""
    if not condition:
        return True
    c = condition.lower()
    # Event-gated self-buffs.
    if "enemy hq" in c:
        return event == "on_enemy_hq_damaged"
    if "survives combat" in c:
        return event == "on_survives_combat"
    if "is dealt damage" in c or "takes damage" in c or "is damaged" in c or "dealt damage" in c:
        return event == "on_damage"
    # Card-play triggers filtered by the played card's type / ability.
    played = _played_card(context, cards)
    if "give an order" in c or "play an order" in c or "gives an order" in c:
        return event == "on_friendly_card_played" and played is not None and played.type == "order"
    if "card with intel" in c:
        return event == "on_friendly_card_played" and _has_ability(played, "intel")
    if "card with bond" in c:
        return event == "on_friendly_card_played" and _has_ability(played, "bond")
    if state is not None and context.player_id in state.players:
        player = state.players[context.player_id]
        opponent = next((candidate for pid, candidate in state.players.items() if pid != context.player_id), None)
        enemy_count = len(opponent.units) if opponent is not None else 0
        own_count = len(player.units)
        match = re.search(r"enemy has (\d+) or more units?", c)
        if match:
            return enemy_count >= int(match.group(1))
        match = re.search(r"you control (\d+) or more units?", c)
        if match:
            return own_count >= int(match.group(1))
        match = re.search(r"you control (?:another )?(?:unit )?with (\d+) or more attack", c)
        if match:
            source_id = getattr(context, "source_unit_id", None)
            units = [unit for unit in player.units if "another" not in c or unit.instance_id != source_id]
            return any(unit.attack >= int(match.group(1)) for unit in units)
        if re.search(r"you control (?:an? )?infantry", c):
            return any(cards.get(unit.card_id).type == "infantry" for unit in player.units if unit.card_id in cards)
        if "you control the frontline" in c:
            frontline = state.battlefield.get("frontline", [])
            return bool(frontline) and all(
                any(unit.instance_id == unit_id and unit.owner_id == context.player_id for unit in player.units)
                for unit_id in frontline
            )
        match = re.search(r"your hq has (\d+) or less defense", c)
        if match:
            return player.hq.current_health <= int(match.group(1))
        if "you have no cards in hand" in c:
            return not player.hand
        if "navy card on top of your deck" in c:
            if not player.deck or player.deck[0] not in cards:
                return False
            top = cards.get(player.deck[0])
            return "navy" in top.name.lower() or any("navy" in ability.lower() for ability in top.abilities)
    # State-based / other conditions: retained leniently for existing parsed
    # persistent effects. Training readiness audits report this as a blocker.
    return True


def _played_card(context, cards: CardDatabase):
    cid = context.metadata.get("played_card_id") if context and context.metadata else None
    return cards.get(cid) if cid else None


def _has_ability(card, ability: str) -> bool:
    if card is None:
        return False
    abilities = card.abilities
    if not isinstance(abilities, (list, tuple)):
        return False
    return any(ability in (a.lower() if isinstance(a, str) else "") for a in abilities)
