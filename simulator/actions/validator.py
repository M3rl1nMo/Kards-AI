"""Shared validation rules for Phase 2 actions."""

from __future__ import annotations

from simulator.core.state import GameState, GameStatus, UnitState


class ActionValidationError(ValueError):
    """An action was invalid for the supplied game state."""


def require_active_player(state: GameState, player_id: str) -> None:
    if state.game_status != GameStatus.IN_PROGRESS:
        raise ActionValidationError("The game is already over")
    if state.current_player != player_id:
        raise ActionValidationError("It is not this player's turn")
    if player_id not in state.players:
        raise ActionValidationError("Unknown player")


def find_unit(state: GameState, unit_id: str) -> UnitState:
    for player in state.players.values():
        for unit in player.units:
            if unit.instance_id == unit_id:
                return unit
    raise ActionValidationError("Unknown unit instance: {0}".format(unit_id))


def opponent_id(state: GameState, player_id: str) -> str:
    opponents = [candidate for candidate in state.players if candidate != player_id]
    if len(opponents) != 1:
        raise ActionValidationError("A two-player game requires exactly one opponent")
    return opponents[0]
