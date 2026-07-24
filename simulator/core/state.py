"""Serializable game state designed for fast branchable AI simulations."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class GameStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    PLAYER_ONE_WON = "player_one_won"
    PLAYER_TWO_WON = "player_two_won"
    DRAW = "draw"


@dataclass
class ResourceState:
    """KARDS resource pool. Names stay explicit to avoid future rule ambiguity."""

    kredits: int = 0
    max_kredits: int = 0


@dataclass
class Headquarters:
    """Runtime headquarters model; all HQ changes go through HQResolver."""

    max_health: int = 20
    current_health: int = 20
    nation: str | None = None
    defense_modifier: int = 0
    damage_cap_per_turn: int = 0
    immune_until_end_of_turn: bool = False
    active_effects: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class UnitState:
    instance_id: str
    card_id: str
    attack: int
    defense: int
    owner_id: str
    position: str = "support_line"
    modifiers: list[dict[str, Any]] = field(default_factory=list)
    status: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlayerState:
    player_id: str
    nation: str | None = None
    deck: list[str] = field(default_factory=list)
    hand: list[str] = field(default_factory=list)
    resources: ResourceState = field(default_factory=ResourceState)
    hq: Headquarters = field(default_factory=Headquarters)
    units: list[UnitState] = field(default_factory=list)
    active_countermeasures: list[dict[str, Any]] = field(default_factory=list)
    fatigue_damage: int = 0
    # Reusable, serializable cost-modifier registries (no unstructured blobs).
    # cost_modifiers: kredit-cost changes for cards played from hand.
    #   entry: {"kind": "hand_cost", "amount": int, "set_cost": int|None,
    #           "scope": "all"|"order"|"ability"|"exclude_nation",
    #           "filter_value": str, "min_cost": int, "expires_turn": int|None}
    # op_cost_rules: operation-cost changes applied when a unit is deployed.
    #   entry: {"amount": int, "set_cost": int|None, "scope": "all"|"type"|"ability"|"name",
    #           "filter_value": str}
    cost_modifiers: list[dict[str, Any]] = field(default_factory=list)
    op_cost_rules: list[dict[str, Any]] = field(default_factory=list)
    # Temporary (duration-bounded) effect reversals tracked for native rules.
    # Each entry: {"turn": int (turn at which to revert), "reverts": [revert-op]}.
    # revert-op for a stat change: {"unit_id": str, "attr": "attack"|"defense",
    #   "delta": int}; for a grant: {"unit_id": str, "remove_ability": str}.
    temporary_effects: list[dict[str, Any]] = field(default_factory=list)
    # Scheduled one-shot actions deferred to a future turn phase (delayed
    # discard / return-to-hand). Entry:
    #   {"trigger": "end_of_turn"|"start_of_next_turn"|"end_of_next_turn",
    #    "unit_id": str, "kind": "discard"|"return_to_hand",
    #    "source_card_id": str, "wait": int}
    # `wait` counts how many of the owner's end_turns to skip before firing
    # (used by "end of your next turn" so it fires on the 2nd end_turn).
    scheduled: list[dict[str, Any]] = field(default_factory=list)
    status: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hq.nation is None:
            self.hq.nation = self.nation


@dataclass
class GameState:
    """Complete mutable state; actions in later phases are its sole mutator."""

    current_player: str
    players: dict[str, PlayerState]
    turn_number: int = 1
    battlefield: dict[str, list[str]] = field(default_factory=lambda: {"frontline": [], "support_line": []})
    graveyard: dict[str, list[str]] = field(default_factory=dict)
    removed_cards: list[str] = field(default_factory=list)
    game_status: GameStatus = GameStatus.IN_PROGRESS
    event_log: list[dict[str, Any]] = field(default_factory=list)
    mulligan_pending: list[str] = field(default_factory=list)
    rng_seed: int | None = None
    pending_cancels: list[dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "GameState":
        """Return an isolated deep copy suitable for tree search branching."""
        # Keep the branchable runtime state fully isolated, but avoid generic
        # dataclass reconstruction overhead on every MCTS edge expansion.
        return GameState(
            current_player=self.current_player,
            players={player_id: _clone_player(player) for player_id, player in self.players.items()},
            turn_number=self.turn_number,
            battlefield=copy.deepcopy(self.battlefield),
            graveyard=copy.deepcopy(self.graveyard),
            removed_cards=list(self.removed_cards),
            game_status=self.game_status,
            event_log=copy.deepcopy(self.event_log),
            mulligan_pending=list(self.mulligan_pending),
            rng_seed=self.rng_seed,
            pending_cancels=copy.deepcopy(self.pending_cancels),
        )

    def clone_for_search(self) -> "GameState":
        """Clone a state for MCTS without copying immutable log-entry dictionaries.

        Simulator actions only append new event records; rule resolution reads
        earlier records but never edits them.  A new list therefore preserves
        every branch's event history and RNG length while avoiding a deep copy
        whose cost grows with game length.  Public ``clone()`` deliberately
        remains fully isolated for callers that may edit log records.
        """
        return GameState(
            current_player=self.current_player,
            players={player_id: _clone_player(player) for player_id, player in self.players.items()},
            turn_number=self.turn_number,
            battlefield={key: list(value) for key, value in self.battlefield.items()},
            graveyard={key: list(value) for key, value in self.graveyard.items()},
            removed_cards=list(self.removed_cards),
            game_status=self.game_status,
            event_log=list(self.event_log),
            mulligan_pending=list(self.mulligan_pending),
            rng_seed=self.rng_seed,
            pending_cancels=copy.deepcopy(self.pending_cancels),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe state snapshot."""
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GameState":
        players = {
            player_id: PlayerState(
                player_id=data["player_id"],
                nation=data.get("nation"),
                deck=list(data.get("deck", ())),
                hand=list(data.get("hand", ())),
                resources=ResourceState(**data.get("resources", {})),
                hq=Headquarters(**data["hq"]) if "hq" in data else Headquarters(current_health=int(data.get("hq_health", 20))),
                units=[UnitState(**unit) for unit in data.get("units", ())],
                active_countermeasures=copy.deepcopy(data.get("active_countermeasures", ())),
                fatigue_damage=int(data.get("fatigue_damage", 0)),
                cost_modifiers=copy.deepcopy(data.get("cost_modifiers", ())),
                op_cost_rules=copy.deepcopy(data.get("op_cost_rules", ())),
                temporary_effects=copy.deepcopy(data.get("temporary_effects", ())),
                scheduled=copy.deepcopy(data.get("scheduled", ())),
                status=copy.deepcopy(data.get("status", {})),
            )
            for player_id, data in payload["players"].items()
        }
        return cls(
            current_player=payload["current_player"],
            players=players,
            turn_number=int(payload.get("turn_number", 1)),
            battlefield={key: list(value) for key, value in payload.get("battlefield", {}).items()},
            graveyard={key: list(value) for key, value in payload.get("graveyard", {}).items()},
            removed_cards=list(payload.get("removed_cards", ())),
            game_status=GameStatus(payload.get("game_status", GameStatus.IN_PROGRESS)),
            event_log=copy.deepcopy(payload.get("event_log", ())),
            mulligan_pending=list(payload.get("mulligan_pending", ())),
            rng_seed=payload.get("rng_seed"),
            pending_cancels=copy.deepcopy(payload.get("pending_cancels", ())),
        )

    @classmethod
    def from_json(cls, payload: str) -> "GameState":
        return cls.from_dict(json.loads(payload))


def _clone_player(player: PlayerState) -> PlayerState:
    """Clone mutable game data without generic dataclass graph traversal."""
    return PlayerState(
        player_id=player.player_id, nation=player.nation, deck=list(player.deck), hand=list(player.hand),
        resources=ResourceState(player.resources.kredits, player.resources.max_kredits),
        hq=Headquarters(player.hq.max_health, player.hq.current_health, player.hq.nation,
                        player.hq.defense_modifier, player.hq.damage_cap_per_turn,
                        player.hq.immune_until_end_of_turn, copy.deepcopy(player.hq.active_effects)),
        units=[UnitState(unit.instance_id, unit.card_id, unit.attack, unit.defense, unit.owner_id,
                         unit.position, copy.deepcopy(unit.modifiers), copy.deepcopy(unit.status)) for unit in player.units],
        active_countermeasures=copy.deepcopy(player.active_countermeasures), fatigue_damage=player.fatigue_damage,
        cost_modifiers=copy.deepcopy(player.cost_modifiers), op_cost_rules=copy.deepcopy(player.op_cost_rules),
        temporary_effects=copy.deepcopy(player.temporary_effects), scheduled=copy.deepcopy(player.scheduled),
        status=copy.deepcopy(player.status),
    )
