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
    priors: list[float] = field(default_factory=list)
    prior_order: list[int] = field(default_factory=list)
    virtual_visits: int = 0
    action_features: torch.Tensor | None = None
    legal_mask: torch.Tensor | None = None
    observations: dict[str, torch.Tensor] = field(default_factory=dict)
    evaluations: dict[str, tuple[torch.Tensor, float]] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    @property
    def value(self) -> float: return self.value_sum / self.visit_count if self.visit_count else 0.0


class MCTS:
    def __init__(self, model: KARDSNet | None, encoder: ObservationEncoder, simulations: int = 64, c_puct: float = 1.5, seed: int = 0, inference: Any | None = None,
                 *, initial_expansion: int | None = 16, widening_factor: float = 2.0, virtual_loss: float = 1.0) -> None:
        self.model, self.encoder, self.simulations, self.c_puct, self.rng, self.codec = model.eval() if model else None, encoder, simulations, c_puct, random.Random(seed), ActionEncoder()
        self.device = next(model.parameters()).device if model else torch.device("cpu")
        self.inference = inference
        self.initial_expansion = initial_expansion
        self.widening_factor = widening_factor
        self.virtual_loss = virtual_loss
        self.temperature = 0.0
        self.dirichlet_alpha = 0.0
        self.exploration_fraction = 0.0

    def set_training_policy(self, *, temperature: float = 0.0, dirichlet_alpha: float = 0.0,
                            exploration_fraction: float = 0.0) -> None:
        self.temperature = max(0.0, temperature)
        self.dirichlet_alpha = max(0.0, dirichlet_alpha)
        self.exploration_fraction = min(1.0, max(0.0, exploration_fraction))

    def search(self, state: GameState, cards: CardDatabase, root_player: str) -> tuple[Action, dict[str, float]]:
        root = MCTSNode(state.clone_for_search(), state.current_player); self._expand(root, cards)
        if self.exploration_fraction and self.dirichlet_alpha and root.priors:
            noise = self._dirichlet(len(root.priors), self.dirichlet_alpha)
            root.priors = [(1.0 - self.exploration_fraction) * prior + self.exploration_fraction * sample
                           for prior, sample in zip(root.priors, noise)]
            root.prior_order = sorted(range(len(root.actions)), key=root.priors.__getitem__, reverse=True)
        for _ in range(self.simulations):
            node = root; path = [node]
            # Nodes retain the complete legal-action list, but successor
            # states are materialised only when PUCT selects that action.
            # This keeps the search policy identical while avoiding full
            # state clones for actions that the allotted simulations never
            # visit.
            while node.expanded and node.actions:
                node = self._select(node, cards); path.append(node)
                node.virtual_visits += 1
            value = self._evaluate(node, cards, root_player)
            self._backpropagate(path, value, root_player)
        if not root.children: return root.actions[0], {"0": 1.0}
        visits = {index: child.visit_count for index, child in root.children.items()}
        total = sum(visits.values()) or 1
        best = self._sample_visit(visits)
        return root.actions[best], {str(index): count / total for index, count in visits.items()}

    def _sample_visit(self, visits: dict[int, int]) -> int:
        if self.temperature <= 0.0:
            return max(visits, key=visits.get)
        weights = [count ** (1.0 / self.temperature) for count in visits.values()]
        threshold = self.rng.random() * sum(weights); cumulative = 0.0
        for index, weight in zip(visits, weights):
            cumulative += weight
            if cumulative >= threshold:
                return index
        return next(reversed(visits))

    def _dirichlet(self, size: int, alpha: float) -> list[float]:
        samples = [self.rng.gammavariate(alpha, 1.0) for _ in range(size)]
        total = sum(samples)
        return [sample / total for sample in samples] if total else [1.0 / size] * size

    def _expand(self, node: MCTSNode, cards: CardDatabase) -> None:
        if node.expanded:
            return
        if node.state.game_status.value != "in_progress": return
        # Legal-action generation is read-only. Branch states are still
        # cloned below before executing an action, but cloning merely to ask
        # the validator for candidates is redundant.
        env = Simulator(cards, state=node.state); node.actions = env.get_available_actions()
        if not node.actions: return
        # Candidate rows are scored independently. Padding to MAX_ACTIONS is
        # required in replay/training tensors, but would only run masked GPU
        # work during tree search.
        node.action_features, node.legal_mask = self.codec.encode_legal_actions(node.actions, pad_to_max=False)
        logits, _ = self._network(node, node.player_to_move)
        node.priors = torch.softmax(logits, dim=0)[:len(node.actions)].tolist()
        # Rule legality is never pruned.  This ordering only controls which
        # legal successor is materialised first; progressive widening admits
        # every candidate as search visits grow.
        node.prior_order = sorted(range(len(node.actions)), key=node.priors.__getitem__, reverse=True)
        node.expanded = True

    def _observation(self, node: MCTSNode, player_id: str) -> torch.Tensor:
        if player_id not in node.observations:
            node.observations[player_id] = self.encoder.encode(node.state, player_id)
        return node.observations[player_id]

    def _network(self, node: MCTSNode, player_id: str) -> tuple[torch.Tensor, float]:
        """Return policy and value from one shared model forward per view."""
        cached = node.evaluations.get(player_id)
        if cached is not None:
            return cached
        assert node.action_features is not None and node.legal_mask is not None
        if self.inference is not None:
            logits, value = self.inference.evaluate(self._observation(node, player_id), node.action_features, node.legal_mask)
            result = (logits, float(value.item()))
        elif self.model is None:
            result = (torch.zeros(len(node.actions)), 0.0)
        else:
            with torch.inference_mode():
                logits, value = self.model(self._observation(node, player_id).to(self.device), node.action_features.to(self.device), node.legal_mask.to(self.device))
                result = (logits[0].cpu(), float(value.item()))
        node.evaluations[player_id] = result
        return result

    def _select(self, node: MCTSNode, cards: CardDatabase) -> MCTSNode:
        candidates = self._candidate_indices(node)
        scale = math.sqrt(node.visit_count + node.virtual_visits + 1)
        best_index = max(
            candidates,
            key=lambda index: (
                (node.children[index].value - self.virtual_loss * node.children[index].virtual_visits)
                + self.c_puct * node.children[index].prior_probability * scale / (1 + node.children[index].visit_count + node.children[index].virtual_visits)
                if index in node.children
                else self.c_puct * node.priors[index] * scale
            ),
        )
        child = node.children.get(best_index)
        if child is not None:
            return child

        # Apply the exact same action implementation as before, but only for
        # the branch PUCT has chosen.  `clone_for_search` keeps every mutable
        # game component isolated while sharing immutable history entries.
        child_state = node.state.clone_for_search()
        action = node.actions[best_index]
        action.execute(child_state, cards)
        child = MCTSNode(child_state, child_state.current_player, node, action, node.priors[best_index])
        node.children[best_index] = child
        return child

    def _candidate_indices(self, node: MCTSNode) -> list[int]:
        """Top-prior legal actions, widened as a node receives visits.

        ``initial_expansion=None`` is the exact full-action-space mode used
        by compatibility tests and debugging.  The training default limits
        early branching but never changes simulator legality or action rules.
        """
        if self.initial_expansion is None:
            return node.prior_order
        width = max(self.initial_expansion, int(self.widening_factor * math.sqrt(node.visit_count + node.virtual_visits + 1)))
        return node.prior_order[:min(len(node.prior_order), width)]

    def _evaluate(self, node: MCTSNode, cards: CardDatabase, root_player: str) -> float:
        status = node.state.game_status.value
        if status != "in_progress": return 1.0 if (status == "player_one_won") == (root_player == "p1") else -1.0 if status != "draw" else 0.0
        self._expand(node, cards)
        return self._network(node, root_player)[1]

    @staticmethod
    def _backpropagate(path: list[MCTSNode], value: float, root_player: str) -> None:
        for node in path:
            if node.virtual_visits:
                node.virtual_visits -= 1
            node.visit_count += 1; node.value_sum += value if node.player_to_move == root_player else -value
