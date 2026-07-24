"""PUCT Monte-Carlo Tree Search with optional KARDSNet priors/value."""
from __future__ import annotations

from dataclasses import dataclass, field
import math, random
from typing import Any
import torch

from ai.action_encoder import ActionEncoder
from ai.observation import ObservationEncoder
from ai.network import KARDSNet
from simulator.actions.action import Action
from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.core.state import GameState


@dataclass
class MCTSNode:
    state: GameState
    player_to_move: str
    parent: "MCTSNode | None" = None
    action: Action | None = None
    prior_probability: float = 1.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    actions: list[Action] = field(default_factory=list)
    action_features: torch.Tensor | None = None
    legal_mask: torch.Tensor | None = None
    visit_count: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    @property
    def value(self) -> float: return self.value_sum / self.visit_count if self.visit_count else 0.0


class MCTS:
    def __init__(self, model: KARDSNet | None, encoder: ObservationEncoder, simulations: int = 64, c_puct: float = 1.5, seed: int = 0) -> None:
        self.model, self.encoder, self.simulations, self.c_puct, self.rng, self.codec = model.eval() if model else None, encoder, simulations, c_puct, random.Random(seed), ActionEncoder()
        self.device = next(model.parameters()).device if model else torch.device("cpu")

    def search(self, state: GameState, cards: CardDatabase, root_player: str) -> tuple[Action, dict[str, float]]:
        root = MCTSNode(state.clone(), state.current_player); self._expand(root, cards)
        for _ in range(self.simulations):
            node = root; path = [node]
            while node.expanded and node.children:
                node = self._select(node); path.append(node)
            value = self._evaluate(node, cards, root_player)
            self._backpropagate(path, value, root_player)
        if not root.children: return root.actions[0], {"0": 1.0}
        visits = {index: child.visit_count for index, child in root.children.items()}
        total = sum(visits.values()) or 1
        best = max(visits, key=visits.get)
        return root.actions[best], {str(index): count / total for index, count in visits.items()}

    def _simulator(self, state: GameState, cards: CardDatabase) -> Simulator:
        env = Simulator(cards); env.state = state.clone(); return env

    def _expand(self, node: MCTSNode, cards: CardDatabase) -> None:
        if node.state.game_status.value != "in_progress": return
        env = self._simulator(node.state, cards); node.actions = env.get_available_actions()
        if not node.actions: return
        node.action_features, node.legal_mask = self.codec.encode_legal_actions(node.actions)
        priors = self._priors(node.state, node.player_to_move, node.action_features, node.legal_mask, len(node.actions))
        for index, action in enumerate(node.actions):
            child_env = self._simulator(node.state, cards)
            # MCTS only needs the successor state. `Simulator.step()` also
            # builds a reward tuple and a cloned public snapshot, neither of
            # which the search consumes. Execute the identical action mutator
            # directly to avoid that unused snapshot allocation.
            assert child_env.state is not None
            action.execute(child_env.state, cards)
            # `child_env` is immediately discarded, therefore its post-step
            # state already has exclusive ownership. Avoiding `get_state()`
            # removes one full deep clone per expanded action without sharing
            # mutable state between MCTS branches.
            child_state = child_env.state
            assert child_state is not None
            node.children[index] = MCTSNode(child_state, child_state.current_player, node, action, priors[index])
        node.expanded = True

    def _priors(self, state: GameState, player_id: str, features: torch.Tensor, mask: torch.Tensor, action_count: int) -> list[float]:
        if self.model is None: return [1.0 / action_count] * action_count
        with torch.no_grad():
            logits, _ = self.model(self.encoder.encode(state, player_id).to(self.device), features.to(self.device), mask.to(self.device))
            return torch.softmax(logits[0], dim=0)[:action_count].tolist()

    def _select(self, node: MCTSNode) -> MCTSNode:
        scale = math.sqrt(node.visit_count + 1)
        return max(node.children.values(), key=lambda child: child.value + self.c_puct * child.prior_probability * scale / (1 + child.visit_count))

    def _evaluate(self, node: MCTSNode, cards: CardDatabase, root_player: str) -> float:
        status = node.state.game_status.value
        if status != "in_progress": return 1.0 if (status == "player_one_won") == (root_player == "p1") else -1.0 if status != "draw" else 0.0
        self._expand(node, cards)
        if self.model is None: return 0.0
        with torch.no_grad():
            assert node.action_features is not None and node.legal_mask is not None
            _, value = self.model(self.encoder.encode(node.state, root_player).to(self.device), node.action_features.to(self.device), node.legal_mask.to(self.device))
            return float(value.item())

    @staticmethod
    def _backpropagate(path: list[MCTSNode], value: float, root_player: str) -> None:
        for node in path:
            node.visit_count += 1; node.value_sum += value if node.player_to_move == root_player else -value
