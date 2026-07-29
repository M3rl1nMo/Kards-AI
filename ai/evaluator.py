"""Match-based evaluation metrics for AI agents."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
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
    def evaluate(self, candidate: BaseAgent, opponent: BaseAgent, games: int = 20,
                 *, deck_pool: list[tuple[str, str | None]] | None = None) -> EvaluationReport:
        """Evaluate on the supplied legal deck distribution, with a fixed seed.

        ``deck_pool`` must match the formal self-play distribution when used
        for Champion promotion.  Falling back to the legacy default is kept
        only for callers that do not provide a training configuration.
        """
        buffer = ReplayBuffer(); report = SelfPlayRunner(self.cards, self.encoder, buffer, seed=self.seed).run(
            games, candidate, opponent, deck_pool=deck_pool)
        return EvaluationReport(games, report.p1_wins, report.p2_wins, report.draws, report.p1_wins / games,
                                report.average_turns, report.average_hq_damage, report.average_cards_played)

    @staticmethod
    def save_report(report: EvaluationReport, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
