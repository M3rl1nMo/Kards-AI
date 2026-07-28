"""Deterministic, seat-balanced MCTS evaluation for checkpoint strength."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

from ai.agents import MCTSAgent, RandomAgent, RuleBasedAgent
from ai.observation import ObservationEncoder
from ai.replay_buffer import ReplayBuffer
from ai.selfplay import SelfPlayRunner
from simulator.cards.loader import CardDatabase


@dataclass(frozen=True)
class FairEvaluationReport:
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    simulations: int
    nations: list[str]
    seed: int
    candidate_as_p1_games: int
    candidate_as_p2_games: int


class FairEvaluator:
    """Evaluate a model with MCTS while alternating candidate seats.

    A single seeded runner supplies deterministic, varied decks.  Candidate
    seats alternate every game, so first-player effects cannot inflate its
    reported win rate.
    """

    def __init__(self, cards: CardDatabase, encoder: ObservationEncoder, *, seed: int = 20260727) -> None:
        self.cards, self.encoder, self.seed = cards, encoder, seed

    def evaluate(self, model, opponent: str, *, games: int = 500, simulations: int = 16,
                 nations: list[str] | None = None, deck_pool: list[tuple[str, str | None]] | None = None) -> FairEvaluationReport:
        if games < 2:
            raise ValueError("Fair evaluation requires at least two games to alternate seats")
        if opponent not in {"random", "rule", "model"}:
            raise ValueError(f"Unsupported opponent: {opponent}")
        nations = nations or ["France"]
        runner = SelfPlayRunner(self.cards, self.encoder, ReplayBuffer(), seed=self.seed)
        wins = losses = draws = p1_games = p2_games = 0
        for index in range(games):
            game_seed = self.seed + index * 17
            candidate = MCTSAgent(model, self.encoder, simulations, game_seed + 1, temperature=0.0)
            if opponent == "random": rival = RandomAgent(game_seed + 2)
            elif opponent == "rule": rival = RuleBasedAgent()
            else: rival = MCTSAgent(model, self.encoder, simulations, game_seed + 2, temperature=0.0)
            candidate_first = index % 2 == 0
            report = runner.run(1, candidate if candidate_first else rival,
                                rival if candidate_first else candidate,
                                nation=nations[index % len(nations)], deck_pool=deck_pool)
            if candidate_first:
                wins += report.p1_wins; losses += report.p2_wins; p1_games += 1
            else:
                wins += report.p2_wins; losses += report.p1_wins; p2_games += 1
            draws += report.draws
        return FairEvaluationReport(games, wins, losses, draws, wins / games, simulations,
                                    list(nations), self.seed, p1_games, p2_games)

    @staticmethod
    def save(report: FairEvaluationReport, path: str | Path) -> None:
        destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
