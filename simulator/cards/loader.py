"""Loader for the single authoritative kards.info card catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulator.cards.card import CARD_TYPES, Card


class CardLoadError(ValueError):
    """Raised when the kards.info catalog is malformed."""


@dataclass(frozen=True)
class CardLoadReport:
    total_cards: int
    metadata_total_cards: int
    type_counts: Mapping[str, int]


class CardDatabase:
    """Read-only, ID-indexed kards.info card catalog."""

    def __init__(self, cards: Iterable[Card], metadata: Mapping[str, Any] | None = None) -> None:
        materialized = tuple(cards)
        if not materialized:
            raise CardLoadError("Card database has no cards")
        self.metadata = dict(metadata or {})
        self._cards = {card.id: card for card in materialized}
        if len(self._cards) != len(materialized):
            raise CardLoadError("kards.info card IDs must be unique")
        invalid_types = sorted({card.type for card in materialized} - CARD_TYPES)
        if invalid_types:
            raise CardLoadError("Unsupported kards.info card types: " + ", ".join(invalid_types))

    @classmethod
    def from_file(cls, cards_path: str | Path) -> "CardDatabase":
        with Path(cards_path).open(encoding="utf-8-sig") as handle:
            document = json.load(handle)
        if not isinstance(document, Mapping) or not isinstance(document.get("metadata"), Mapping) or not isinstance(document.get("cards"), list):
            raise CardLoadError("kards.info catalog must contain metadata and cards")
        cards = tuple(_card_from_record(record) for record in document["cards"])
        database = cls(cards, document["metadata"])
        expected = database.metadata.get("totalCards")
        if isinstance(expected, int) and expected != len(database):
            raise CardLoadError("metadata.totalCards does not match cards array")
        return database

    def get(self, card_id: str) -> Card:
        return self._cards[card_id]

    def __contains__(self, card_id: object) -> bool:
        return card_id in self._cards

    def __len__(self) -> int:
        return len(self._cards)

    def __iter__(self):
        return iter(self._cards.values())

    @property
    def report(self) -> CardLoadReport:
        counts = {card_type: sum(card.type == card_type for card in self._cards.values()) for card_type in sorted(CARD_TYPES)}
        return CardLoadReport(len(self), int(self.metadata.get("totalCards", len(self))), counts)


def _card_from_record(record: Any) -> Card:
    if not isinstance(record, Mapping):
        raise CardLoadError("Every kards.info card must be an object")
    required_strings = ("id", "importId", "slug", "name", "nation", "type", "rarity", "set", "text", "image", "imageUrl", "thumbUrl", "localImageUrl", "localThumbUrl")
    missing = [field for field in required_strings if not isinstance(record.get(field), str)]
    if missing:
        raise CardLoadError("Card is missing required fields: " + ", ".join(missing))
    card_type = record["type"]
    if card_type not in CARD_TYPES:
        raise CardLoadError("Unsupported kards.info card type: " + str(card_type))
    unit = card_type in {"infantry", "fighter", "tank", "bomber", "artillery"}
    attack, defense, operation_cost = record.get("attack"), record.get("defense"), record.get("operationCost")
    if unit and not all(isinstance(value, int) for value in (attack, defense, operation_cost)):
        raise CardLoadError("Unit card must have attack, defense, and operationCost")
    if not unit and any(value is not None for value in (attack, defense, operation_cost)):
        raise CardLoadError("Non-unit card must have null attack, defense, and operationCost")
    return Card(
        id=record["id"], importId=record["importId"], slug=record["slug"], name=record["name"], nation=record["nation"], type=card_type,
        rarity=record["rarity"], set=record["set"], kredits=_integer(record, "kredits"), attack=attack, defense=defense, operationCost=operation_cost,
        text=record["text"], abilities=_string_tuple(record, "abilities"), abilityLabels=_string_tuple(record, "abilityLabels"),
        unknownAttributes=_string_tuple(record, "unknownAttributes"), reserved=_boolean(record, "reserved"), spawnable=_boolean(record, "spawnable"),
        exile=record.get("exile"), canCreate=_optional_string_tuple(record.get("canCreate")), image=record["image"], imageUrl=record["imageUrl"], thumbUrl=record["thumbUrl"],
        localImageUrl=record["localImageUrl"], localThumbUrl=record["localThumbUrl"],
    )


def _integer(record: Mapping[str, Any], field: str) -> int:
    value = record.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise CardLoadError("Card field " + field + " must be an integer")
    return value


def _boolean(record: Mapping[str, Any], field: str) -> bool:
    value = record.get(field)
    if not isinstance(value, bool):
        raise CardLoadError("Card field " + field + " must be a boolean")
    return value


def _string_tuple(record: Mapping[str, Any], field: str) -> tuple[str, ...]:
    return _optional_string_tuple(record.get(field)) or ()


def _optional_string_tuple(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CardLoadError("Card list field must contain strings")
    return tuple(value)
