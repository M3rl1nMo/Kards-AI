"""Deck legality checks suitable for deck-building agents."""
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class DeckValidationResult:
    valid: bool
    errors: tuple[str, ...]


class DeckValidator:
    def __init__(self, cards, deck_size=40, copy_limit=4):
        self.cards, self.deck_size, self.copy_limit = cards, deck_size, copy_limit

    def validate(self, deck, main_nation, ally_nation=None):
        errors = []
        if len(deck) != self.deck_size:
            errors.append("Deck must contain {0} cards".format(self.deck_size))
        counts = Counter(deck)
        for card_id, count in counts.items():
            if card_id not in self.cards:
                errors.append("Unknown card: {0}".format(card_id)); continue
            card = self.cards.get(card_id)
            if card.is_token:
                errors.append("OnlySpawnable card cannot be included: {0}".format(card_id))
            rarity_limit = {"standard": 4, "limited": 3, "special": 2, "elite": 1}.get(card.rarity.lower(), self.copy_limit)
            if count > min(self.copy_limit, rarity_limit):
                errors.append("Copy limit exceeded: {0}".format(card_id))
            nation = card.nation
            if nation and nation not in {main_nation, ally_nation, "Neutral"}:
                errors.append("Nation not allowed: {0}".format(card_id))
        return DeckValidationResult(not errors, tuple(errors))
