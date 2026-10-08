"""Gym-like facade around the state and action layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from copy import deepcopy
from typing import Iterable
import random

from simulator.actions.action import Action, AttackAction, MoveUnitAction, MulliganAction, PassAction, PlayCardAction
from simulator.replay import Replay
from simulator.actions.validator import opponent_id
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, GameStatus, PlayerState
from simulator.core.turn import TurnManager
from simulator.rules.native import engine_for
from simulator.rules.deck import DeckValidator


@dataclass
class Simulator:
    """Headless training interface. Effects and handlers are connected in later phases."""

    cards: CardDatabase
    state: GameState | None = None
    replay: Replay | None = None
    # Replay snapshots serialize the complete state after every action.  They
    # are useful for debugging, but self-play can safely opt out because it
    # stores its own training examples.
    record_replay: bool = True
    # BUG-08：默认严格检查；异常对局必须丢弃。False 仅供排查规则使用。
    strict_effects: bool = True
    _action_cache: list[Action] | None = field(default=None, init=False, repr=False)
    _public_ids: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _action_history: list[dict] = field(default_factory=list, init=False, repr=False)
    _initial_state: dict | None = field(default=None, init=False, repr=False)
    _invalid_episode: bool = field(default=False, init=False, repr=False)

    def reset(self, player_one_deck: Iterable[str], player_two_deck: Iterable[str], nations: tuple[str | None, str | None] = (None, None), seed: int = 0, auto_mulligan: bool = True, ally_nations: tuple[str | None, str | None] = (None, None)) -> GameState:
        self._action_cache = None
        self._public_ids.clear()
        self._action_history.clear()
        self._invalid_episode = False
        deck_one, deck_two = list(player_one_deck), list(player_two_deck)
        nation_one = nations[0] or self._infer_nation(deck_one)
        nation_two = nations[1] or self._infer_nation(deck_two)
        self._validate_deck(deck_one, nation_one, ally_nations[0])
        self._validate_deck(deck_two, nation_two, ally_nations[1])
        self.state = GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", nation_one, ally_nations[0], deck_one),
                "p2": PlayerState("p2", nation_two, ally_nations[1], deck_two),
            },
            graveyard={"p1": [], "p2": []}, rng_seed=seed,
        )
        rng = random.Random(seed)
        rng.shuffle(self.state.players["p1"].deck)
        rng.shuffle(self.state.players["p2"].deck)
        for _ in range(4): TurnManager.draw_card(self.state, "p1")
        for _ in range(5): TurnManager.draw_card(self.state, "p2")
        self.state.mulligan_pending = ["p1", "p2"]
        # BUG-03（已修复）：首回合由最后一次 MulliganAction 统一启动。
        if auto_mulligan:
            MulliganAction("p1").execute(self.state, self.cards)
            MulliganAction("p2").execute(self.state, self.cards)
        self._initial_state = self.state.to_dict()
        self.replay = Replay(self.state.to_dict()) if self.record_replay else None
        return self.get_state()

    def step(self, action: Action) -> tuple[GameState, float, bool]:
        reward, terminal = self.step_fast(action)
        return self.get_state(), reward, terminal

    def step_fast(self, action: Action) -> tuple[float, bool]:
        """Execute an action without creating the unused public state clone.

        The public ``step`` API intentionally remains unchanged.  This method
        is for internal high-volume self-play callers that already hold the
        simulator-owned state and only need reward/termination information.
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step()")
        if self._invalid_episode:
            raise RuntimeError("Invalid episode: call reset() before continuing")
        if self.strict_effects and self._initial_state is None:
            self._initial_state = self.state.to_dict()
        self._action_cache = None
        opponent = opponent_id(self.state, action.player_id)
        before = self.state.players[opponent].hq.current_health
        event_start = len(self.state.event_log)
        action.execute(self.state, self.cards)
        if self.strict_effects:
            self._action_history.append({"type": type(action).__name__, "fields": asdict(action)})
            unsupported = [event for event in self.state.event_log[event_start:]
                           if event.get("event") == "unsupported_effect"]
            if unsupported:
                self._invalid_episode = True
                raise UnsupportedEffectError(unsupported, self._initial_state,
                                             self._action_history, self.state.rng_seed)
        if self.replay:
            self.replay.record(action, self.state)
        reward = float(before - self.state.players[opponent].hq.current_health)
        terminal = self.is_terminal()
        # BUG-04（已修复）：终局奖励采用行动方视角；掉血奖励仍按原接口计算。
        if terminal and self.state.game_status != GameStatus.DRAW:
            winner = "p1" if self.state.game_status == GameStatus.PLAYER_ONE_WON else "p2"
            reward += 100.0 if winner == action.player_id else -100.0
        return reward, terminal

    def get_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("Call reset() before get_state()")
        return self.state.clone()

    def get_available_actions(self) -> list[Action]:
        if self.state is None or self.is_terminal() or self._invalid_episode:
            return []
        if self._action_cache is not None:
            # Never expose the cached container: callers may reorder/filter
            # legal candidates, while a state has not changed underneath it.
            return list(self._action_cache)
        player_id = self.state.current_player
        if self.state.mulligan_pending:
            if player_id not in self.state.mulligan_pending:
                return []
            self._action_cache = [MulliganAction(player_id, subset) for subset in _subsets(tuple(self.state.players[player_id].hand))]
            return list(self._action_cache)
        player = self.state.players[player_id]
        actions: list[Action] = [PassAction(player_id)]
        rule_engine = engine_for(self.cards)
        hand_ids = tuple(player.hand)
        unit_hand = [(card_id, "hand") for card_id in hand_ids if self.cards.get(card_id).is_unit]
        convert_choices = [(card_id, "hand") for card_id in hand_ids] + [(card_id, "deck") for card_id in player.deck]
        all_units: list[str] | None = None
        has_high_attack = any(unit.attack >= 4 for unit in player.units)
        for card_id in hand_ids:
            positions = ("support_line",)  # KARDS units deploy to the support line.
            spec = rule_engine.action_spec_for(card_id)
            if spec.needs_target:
                if all_units is None:
                    all_units = [unit.instance_id for candidate in self.state.players.values() for unit in candidate.units]
                targets = all_units
            else:
                targets = (None,)
            selected_cards = (
                [(candidate_id, zone) for candidate_id, zone in unit_hand if candidate_id != card_id]
                if spec.swap_hand_unit
                else [(candidate_id, zone) for candidate_id, zone in convert_choices if zone != "hand" or candidate_id != card_id]
                if spec.convert_to
                else [(None, None)]
            )
            # BUG-05（已修复）：高攻击力判断只作用于 shock_tactics。
            selected_options = (
                ([None] if has_high_attack else ["blitz", "shock"])
                if spec.shock_tactics
                else (None, *spec.develop_options)
            )
            for position in positions:
                for target_unit_id in targets:
                    for selected_card_id, selected_card_zone in selected_cards:
                        for selected_option in selected_options:
                            action = PlayCardAction(player_id, card_id, position, target_unit_id, selected_card_id, selected_option, selected_card_zone)
                            if _is_valid(action, self.state, self.cards):
                                actions.append(action)
        enemy_id = opponent_id(self.state, player_id)
        for unit in player.units:
            move = MoveUnitAction(player_id, unit.instance_id)
            if _is_valid(move, self.state, self.cards): actions.append(move)
            for target in [None] + [enemy.instance_id for enemy in self.state.players[enemy_id].units]:
                action = AttackAction(player_id, unit.instance_id, target)
                if _is_valid(action, self.state, self.cards):
                    actions.append(action)
        self._action_cache = actions
        return list(self._action_cache)

    def is_terminal(self) -> bool:
        return self.state is not None and self.state.game_status != GameStatus.IN_PROGRESS

    def _validate_deck(self, deck: list[str], nation: str | None, ally_nation: str | None = None) -> None:
        if nation is None:
            raise ValueError("Deck nation could not be inferred")
        result = DeckValidator(self.cards).validate(deck, nation, ally_nation)
        if not result.valid:
            raise ValueError("Illegal deck: " + "; ".join(result.errors))

    def _infer_nation(self, deck: list[str]) -> str | None:
        for card_id in deck:
            if card_id in self.cards and self.cards.get(card_id).nation != "Neutral":
                return self.cards.get(card_id).nation
        return None

    # BUG-06/07：补齐己方状态；观测和玩家动作使用不带卡名的稳定编号。
    def get_observation(self, player_id: str) -> dict:
        """Player-view API that hides the opponent's hand IDs."""
        if self.state is None:
            raise RuntimeError("Call reset() before get_observation()")
        if player_id not in self.state.players:
            raise ValueError("Unknown player")
        player = self.state.players[player_id]
        enemy = self.state.players[opponent_id(self.state, player_id)]
        for owner in self.state.players.values():
            for unit in owner.units:
                self._public_ids.setdefault(unit.instance_id, f"unit-{len(self._public_ids) + 1}")
        opponent_view = _observation_player(enemy, False, self.cards, self._public_ids)
        # Countermeasure activation is deliberately hidden: exposing remaining
        # kredits would reveal the activation payment to the opponent.
        opponent_view["kredits"] = None
        opponent_view["known_hand"] = list(player.status.get("known_enemy_hand", ()))
        return {"turn": self.state.turn_number, "current": self.state.current_player == player_id,
                "self": _observation_player(player, True, self.cards, self._public_ids), "opponent": opponent_view,
                "frontline": [self._public_ids[uid] for uid in self.state.battlefield["frontline"]],
                "support_line": [self._public_ids[uid] for uid in self.state.battlefield["support_line"]],
                "terminal": self.is_terminal()}

    def get_player_actions(self) -> list[Action]:
        """Player-facing actions use the same opaque unit IDs as observations."""
        if self.state is None:
            return []
        self.get_observation(self.state.current_player)
        return [self._map_action(action, self._public_ids) for action in self.get_available_actions()]

    def step_player(self, action: Action) -> tuple[float, bool]:
        """Resolve an opaque player action back to the referee's internal IDs."""
        if action not in self.get_player_actions():
            raise ValueError("Action is not a current legal player action")
        inverse = {public: internal for internal, public in self._public_ids.items()}
        return self.step_fast(self._map_action(action, inverse))

    @staticmethod
    def _map_action(action: Action, mapping: dict[str, str]) -> Action:
        names = {item.name for item in fields(action)}
        changes = {name: mapping[getattr(action, name)]
                   for name in ("unit_id", "attacker_id", "target_unit_id")
                   if name in names and getattr(action, name) is not None}
        return replace(action, **changes)


def _observation_player(player: PlayerState, reveal_hand: bool, cards: CardDatabase, ids: dict) -> dict:
    from simulator.actions.action import _operation_cost
    units = []
    for unit in player.units:
        hidden = bool(unit.status.get("covert") and not reveal_hand)
        visible_status = {key: deepcopy(value) for key, value in unit.status.items()
                          if reveal_hand or key in {"pinned", "suppressed", "deployed_this_turn", "attack_count", "moved_this_turn"}
                          or (not hidden and key in {"keywords", "granted_abilities", "attack_tax", "target_tax"})}
        units.append({"id": ids[unit.instance_id], "card_id": None if hidden else unit.card_id,
                      "attack": None if hidden else unit.attack, "defense": None if hidden else unit.defense,
                      "position": unit.position, "operation_cost": None if hidden else _operation_cost(unit, cards.get(unit.card_id)),
                      "abilities": None if hidden else list(cards.get(unit.card_id).abilities),
                      "modifiers": deepcopy(unit.modifiers) if reveal_hand else None,
                      "status": visible_status})
    return {"hq_health": player.hq.current_health, "hq_defense": player.hq.defense_modifier,
            "kredits": player.resources.kredits, "max_kredits": player.resources.max_kredits,
            "deck_count": len(player.deck), "hand": list(player.hand) if reveal_hand else None,
            "hand_count": len(player.hand), "units": units,
            "active_countermeasures": deepcopy(player.active_countermeasures) if reveal_hand else None,
            "cost_modifiers": deepcopy(player.cost_modifiers) if reveal_hand else None,
            "op_cost_rules": deepcopy(player.op_cost_rules) if reveal_hand else None}


class UnsupportedEffectError(RuntimeError):
    """Reject an invalid trajectory; report is referee-only debugging data."""

    def __init__(self, events, initial_state, actions, seed):
        self.report = deepcopy({"seed": seed, "initial_state": initial_state,
                                "actions": actions, "unsupported_effects": events})
        super().__init__(f"Unsupported effect; discard episode: {events!r}")


def _is_valid(action: Action, state: GameState, cards: CardDatabase) -> bool:
    try:
        action.validate(state, cards)
    except ValueError:
        return False
    return True


def _subsets(values: tuple[str, ...]) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = [()]
    for value in values:
        result += [prefix + (value,) for prefix in tuple(result)]
    return result
