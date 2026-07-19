"""Static KARDS card definition from the kards.info catalog."""

from __future__ import annotations

from dataclasses import dataclass


UNIT_TYPES = frozenset({"infantry", "fighter", "tank", "bomber", "artillery"})
CARD_TYPES = UNIT_TYPES | frozenset({"order", "countermeasure"})
IMAGE_BASE_URL = "https://kards.info"


@dataclass(frozen=True)
class Card:
    """One unmodified card record from ``kards_info_cards.json``.

    This model intentionally uses the catalog field names. Runtime state (such
    as a damaged unit) is held by ``UnitState``, never by this immutable card.
    """

    id: str
    importId: str
    slug: str
    name: str
    nation: str
    type: str
    rarity: str
    set: str
    kredits: int
    attack: int | None
    defense: int | None
    operationCost: int | None
    text: str
    abilities: tuple[str, ...]
    abilityLabels: tuple[str, ...]
    unknownAttributes: tuple[str, ...]
    reserved: bool
    spawnable: bool
    exile: str | None
    canCreate: tuple[str, ...] | None
    image: str
    imageUrl: str
    thumbUrl: str
    localImageUrl: str
    localThumbUrl: str

    @property
    def is_unit(self) -> bool:
        return self.type in UNIT_TYPES

    @property
    def is_token(self) -> bool:
        """OnlySpawnable is the authoritative deck-ineligible token marker."""
        return self.set == "OnlySpawnable"

    def image_url(self, local: bool = False, thumbnail: bool = False) -> str:
        """Return the absolute kards.info image URL for this card."""
        path = self.localThumbUrl if local and thumbnail else self.localImageUrl if local else self.thumbUrl if thumbnail else self.imageUrl
        return IMAGE_BASE_URL + path
