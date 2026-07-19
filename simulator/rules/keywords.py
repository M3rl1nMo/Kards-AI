"""Generic keyword predicates; no card-specific keyword logic lives in cards."""
import re

class KeywordEngine:
    @staticmethod
    def matches_type(card, unit, requested_type: str) -> bool:
        """Return whether a unit satisfies a card-type rule at runtime.

        Some effects grant an additional type (for example, "also counts as a
        tank").  Rules must consult this predicate rather than only immutable
        catalog data.
        """
        kind = requested_type.lower()
        if card.type == kind:
            return True
        return kind == "tank" and bool(unit and unit.status.get("also_tank"))

    @staticmethod
    def has(card, keyword, unit=None):
        """Check catalog and runtime-granted abilities with removal overrides."""
        name = keyword.lower()
        removed = {str(value).lower() for value in (unit.status.get("removed_abilities", ()) if unit else ())}
        added = {str(value).lower() for value in (unit.status.get("added_abilities", ()) if unit else ())}
        static = {value.lower().split(":", 1)[0] for value in card.abilities}
        return name not in removed and (name in static or name in added)

    @staticmethod
    def can_attack_on_deploy(card, unit=None):
        return KeywordEngine.has(card, "blitz", unit)

    @staticmethod
    def can_ignore_guard(card):
        return card.type in {"bomber", "artillery"}

    @staticmethod
    def attack_limit(card, unit=None):
        return 2 if KeywordEngine.has(card, "fury", unit) else 1

    @staticmethod
    def heavy_armor(card, unit=None):
        removed = {str(value).lower() for value in (unit.status.get("removed_abilities", ()) if unit else ())}
        if "heavyarmor" in removed or "heavy armor" in removed:
            return 0
        added = tuple(str(value) for value in (unit.status.get("added_abilities", ()) if unit else ()))
        for value in (*card.abilities, *added):
            match = re.search(r"heavy[ _-]?armor\D*(\d+)", value, re.I)
            if match: return int(match.group(1))
        return 0

    @staticmethod
    def can_attack_from_position(card, position):
        return position == "frontline" or card.type in {"artillery", "bomber", "fighter"}

    @staticmethod
    def is_guarded(state, target, attacker_card, cards):
        if KeywordEngine.can_ignore_guard(attacker_card): return False
        line = state.battlefield.get(target.position, [])
        try: index = line.index(target.instance_id)
        except ValueError: return False
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(line):
                unit_id=line[neighbor]
                for player in state.players.values():
                    for unit in player.units:
                        if unit.instance_id == unit_id and unit.owner_id == target.owner_id and KeywordEngine.has(cards.get(unit.card_id), "guard", unit):
                            return True
        return False
