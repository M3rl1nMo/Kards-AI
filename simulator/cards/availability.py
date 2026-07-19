"""Active-pool policy layered over the immutable kards.info catalog."""

# Retain these records in the source database for replay/history compatibility,
# but disallow them in new decks and actions after their official removal.
RETIRED_CARD_IDS = frozenset({"fortunes_of_war", "night_bombing"})


def is_card_available(card_id: str) -> bool:
    """Return whether a catalog card belongs to the active simulator pool."""
    return card_id not in RETIRED_CARD_IDS
