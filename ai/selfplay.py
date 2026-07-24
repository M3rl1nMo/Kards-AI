"""Self-play trajectory collection without changing simulator rules."""
from __future__ import annotations

from dataclasses import dataclass
import random

from ai.action_encoder import ActionEncoder
from ai.agents import BaseAgent
from ai.observation import ObservationEncoder
from ai.replay_buffer import ReplayBuffer, TrainingExample
from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.testing import build_random_deck


@dataclass(frozen=True)
class SelfPlayReport:
    episodes: int; examples: int; p1_wins: int; p2_wins: int; draws: int
    average_turns: float; average_hq_damage: float; average_cards_played: float


class SelfPlayRunner:
    def __init__(self, cards: CardDatabase, encoder: ObservationEncoder, buffer: ReplayBuffer, max_actions: int = 300, seed: int = 0) -> None:
        self.cards, self.encoder, self.buffer, self.max_actions, self.rng, self.codec = cards, encoder, buffer, max_actions, random.Random(seed), ActionEncoder()

    def run(self, episodes: int, player_one: BaseAgent, player_two: BaseAgent, nation: str = "France") -> SelfPlayReport:
        totals = {"p1": 0, "p2": 0, "draw": 0}; before = len(self.buffer)
        turns_total = 0; hq_damage_total = 0.0; cards_played_total = 0
        for _ in range(episodes):
            env = Simulator(self.cards)
            decks = [build_random_deck(self.cards, seed=self.rng.randrange(2**31), main_nation=nation) for _ in range(2)]
            env.reset(*decks, nations=(nation, nation), seed=self.rng.randrange(2**31)); pending: list[tuple[TrainingExample, str]] = []
            player_one.reset(); player_two.reset()
            for _ in range(self.max_actions):
                if env.is_terminal(): break
                player_id = env.state.current_player  # type: ignore[union-attr]
                agent = player_one if player_id == "p1" else player_two
                legal = env.get_available_actions(); state = env.get_state()
                if hasattr(agent, "set_state"):
                    try: agent.set_state(state, player_id, self.cards)
                    except TypeError: agent.set_state(state, player_id)
                observation = env.get_observation(player_id); agent.observe(observation)
                action = agent.select_action(observation, legal)
                features, mask = self.codec.encode_legal_actions(legal)
                policy = agent.last_policy or [float(a == action) for a in legal]
                padded_policy = policy + [0.0] * (self.codec.max_actions - len(policy))
                pending.append((TrainingExample(self.encoder.encode(state, player_id).tolist(), features.tolist(), mask.tolist(), padded_policy, 0.0), player_id))
                env.step(action)
            final_state = env.state  # type: ignore[assignment]
            status = final_state.game_status.value
            winner = "p1" if status == "player_one_won" else "p2" if status == "player_two_won" else None
            totals[winner or "draw"] += 1
            turns_total += final_state.turn_number
            hq_damage_total += sum(20 - player.hq.current_health for player in final_state.players.values())
            cards_played_total += sum(1 for event in final_state.event_log if event.get("event") in {"unit_deployed", "command_played", "countermeasure_activated"})
            for example, player_id in pending:
                example.value = 0.0 if winner is None else (1.0 if winner == player_id else -1.0)
                self.buffer.add(example)
        return SelfPlayReport(episodes, len(self.buffer) - before, totals["p1"], totals["p2"], totals["draw"],
                              turns_total / episodes, hq_damage_total / episodes, cards_played_total / episodes)
