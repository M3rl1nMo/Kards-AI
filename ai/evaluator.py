"""Match-based evaluation metrics for AI agents."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from ai.agents import BaseAgent
from ai.observation import ObservationEncoder
from ai.replay_buffer import ReplayBuffer
from ai.selfplay import SelfPlayRunner
from simulator.cards.loader import CardDatabase

@dataclass(frozen=True)
class EvaluationReport:
    games: int; wins: int; losses: int; draws: int; win_rate: float; average_turns: float = 0.0; hq_damage: float = 0.0; cards_played: float = 0.0

class Evaluator:
    def __init__(self, cards: CardDatabase, encoder: ObservationEncoder, seed: int = 0) -> None: self.cards, self.encoder, self.seed = cards, encoder, seed
    def evaluate(self, candidate: BaseAgent, opponent: BaseAgent, games: int = 20) -> EvaluationReport:
        buffer = ReplayBuffer(); report = SelfPlayRunner(self.cards, self.encoder, buffer, seed=self.seed).run(games, candidate, opponent)
        return EvaluationReport(games, report.p1_wins, report.p2_wins, report.draws, report.p1_wins / games)
