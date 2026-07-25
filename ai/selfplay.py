"""Self-play trajectory collection without changing simulator rules."""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import multiprocessing as mp
import random
from typing import Callable

import torch

from ai.action_encoder import ActionEncoder
from ai.agents import BaseAgent
from ai.observation import ObservationEncoder
from ai.replay_buffer import ReplayBuffer, TrainingExample
from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator
from simulator.testing import build_random_deck


_PARALLEL_CONTEXT: dict = {}


def _parallel_worker_init(card_path: str, architecture: dict, state_dict: dict | None, simulations: int,
                          max_actions: int, device: str, inference_requests=None) -> None:
    """Initialize one independent, deterministic self-play worker process."""
    from ai.agents import MCTSAgent
    from ai.network import KARDSNet
    from simulator.cards.loader import CardDatabase

    cards = CardDatabase.from_file(card_path)
    model = None
    if state_dict is not None:
        model = KARDSNet(**architecture)
        model.load_state_dict(state_dict)
        model.to(device).eval()
    _PARALLEL_CONTEXT.update(cards=cards, encoder=ObservationEncoder(cards), model=model,
                             simulations=simulations, max_actions=max_actions, device=device,
                             inference_requests=inference_requests)


def _parallel_episode(index: int, seed: int, nation: str, response_queue=None) -> tuple[int, list[TrainingExample], SelfPlayReport, dict]:
    """Run one episode in a worker; result ordering is restored by the parent."""
    from ai.agents import MCTSAgent

    context = _PARALLEL_CONTEXT
    buffer = ReplayBuffer(seed=seed)
    runner = SelfPlayRunner(context["cards"], context["encoder"], buffer, context["max_actions"], seed)
    inference = None
    if context.get("inference_requests") is not None:
        from ai.inference import RemoteInferenceClient
        assert response_queue is not None
        inference = RemoteInferenceClient(context["inference_requests"], response_queue, context.get("worker_id", 0))
    agents = (
        MCTSAgent(context["model"], context["encoder"], context["simulations"], seed + 1, inference=inference),
        MCTSAgent(context["model"], context["encoder"], context["simulations"], seed + 2, inference=inference),
    )
    report = runner.run(1, *agents, nation=nation)
    return index, buffer.examples, report, runner.game_records[-1]


def _cuda_worker_loop(card_path: str, architecture: dict, simulations: int, max_actions: int,
                      requests, response_queue, worker_id: int, tasks, results) -> None:
    """Run CPU-only rule simulations; CUDA inference remains in the parent."""
    _parallel_worker_init(card_path, architecture, None, simulations, max_actions, "cpu", requests)
    _PARALLEL_CONTEXT["worker_id"] = worker_id
    while True:
        task = tasks.get()
        if task is None:
            return
        try:
            results.put((True, _parallel_episode(*task, response_queue)))
        except BaseException as error:
            results.put((False, repr(error)))


@dataclass(frozen=True)
class SelfPlayReport:
    episodes: int; examples: int; p1_wins: int; p2_wins: int; draws: int
    average_turns: float; average_hq_damage: float; average_cards_played: float


class SelfPlayRunner:
    def __init__(self, cards: CardDatabase, encoder: ObservationEncoder, buffer: ReplayBuffer, max_actions: int = 300, seed: int = 0) -> None:
        self.cards, self.encoder, self.buffer, self.max_actions, self.rng, self.codec = cards, encoder, buffer, max_actions, random.Random(seed), ActionEncoder()
        self.game_records: list[dict] = []

    def run(self, episodes: int, player_one: BaseAgent, player_two: BaseAgent, nation: str = "France",
            on_episode_complete: Callable[[int, dict[str, int], int], None] | None = None) -> SelfPlayReport:
        totals = {"p1": 0, "p2": 0, "draw": 0}; before = len(self.buffer)
        turns_total = 0; hq_damage_total = 0.0; cards_played_total = 0
        for _ in range(episodes):
            # Debug replay serializes every full state; the training replay
            # buffer below already retains the required policy/value samples.
            env = Simulator(self.cards, record_replay=False)
            decks = [build_random_deck(self.cards, seed=self.rng.randrange(2**31), main_nation=nation) for _ in range(2)]
            env.reset(*decks, nations=(nation, nation), seed=self.rng.randrange(2**31)); pending: list[tuple[TrainingExample, str]] = []
            player_one.reset(); player_two.reset()
            for _ in range(self.max_actions):
                if env.is_terminal(): break
                player_id = env.state.current_player  # type: ignore[union-attr]
                agent = player_one if player_id == "p1" else player_two
                legal = env.get_available_actions()
                # Agents consume this state synchronously and MCTS immediately
                # creates its own branchable root. Avoid a public snapshot
                # clone here; no agent is permitted to mutate the environment.
                state = env.state
                assert state is not None
                if hasattr(agent, "set_state"):
                    try: agent.set_state(state, player_id, self.cards)
                    except TypeError: agent.set_state(state, player_id)
                observation = env.get_observation(player_id); agent.observe(observation)
                action = agent.select_action(observation, legal)
                # Replay needs only legal candidates.  Padding to 512 here
                # previously consumed most RAM and was rewritten to disk.
                features, mask = self.codec.encode_legal_actions(legal, pad_to_max=False)
                policy = agent.last_policy or [float(a == action) for a in legal]
                pending.append((TrainingExample(self.encoder.encode(state, player_id).tolist(), features.tolist(), mask.tolist(), policy, 0.0), player_id))
                env.step_fast(action)
            final_state = env.state  # type: ignore[assignment]
            status = final_state.game_status.value
            winner = "p1" if status == "player_one_won" else "p2" if status == "player_two_won" else None
            totals[winner or "draw"] += 1
            turns_total += final_state.turn_number
            hq_damage_total += sum(20 - player.hq.current_health for player in final_state.players.values())
            cards_played_total += sum(1 for event in final_state.event_log if event.get("event") in {"unit_deployed", "command_played", "countermeasure_activated"})
            played = [
                {"player": event.get("player_id"), "card": self.cards.get(event["card_id"]).name}
                for event in final_state.event_log
                if event.get("event") in {"unit_deployed", "command_played", "countermeasure_activated"} and event.get("card_id") in self.cards
            ]
            self.game_records.append({"winner": winner or "draw", "turns": final_state.turn_number,
                                      "cards_played": played, "hq_damage": sum(20 - player.hq.current_health for player in final_state.players.values())})
            for example, player_id in pending:
                example.value = 0.0 if winner is None else (1.0 if winner == player_id else -1.0)
                self.buffer.add(example)
            if on_episode_complete:
                on_episode_complete(sum(totals.values()), totals.copy(), len(self.buffer) - before)
        return SelfPlayReport(episodes, len(self.buffer) - before, totals["p1"], totals["p2"], totals["draw"],
                              turns_total / episodes, hq_damage_total / episodes, cards_played_total / episodes)


class VectorizedSelfPlay:
    """Concurrent, independent environments sharing one batched GPU queue.

    This is deliberately a scheduler rather than a second simulator: every
    lane owns a normal ``Simulator`` and follows ``SelfPlayRunner.run``.  The
    only shared mutable component is the inference queue, which serialises
    model execution and batches requests from independent MCTS trees.
    """

    def __init__(self, cards: CardDatabase, encoder: ObservationEncoder, buffer: ReplayBuffer,
                 max_actions: int = 300, seed: int = 0) -> None:
        self.cards, self.encoder, self.buffer = cards, encoder, buffer
        self.max_actions, self.rng = max_actions, random.Random(seed)
        self.game_records: list[dict] = []

    def run(self, episodes: int, model, simulations: int = 64, *, environments: int = 32,
            nation: str = "France", max_batch_size: int = 32, max_wait_ms: float = 2.0) -> SelfPlayReport:
        if episodes < 1 or environments < 1:
            raise ValueError("episodes and environments must be positive")
        from ai.agents import MCTSAgent
        from ai.inference import BatchedInference

        seeds = [self.rng.randrange(2**31) for _ in range(episodes)]

        def play(index: int, seed: int, inference: BatchedInference):
            local_buffer = ReplayBuffer(seed=seed)
            runner = SelfPlayRunner(self.cards, self.encoder, local_buffer, self.max_actions, seed)
            first = MCTSAgent(model, self.encoder, simulations, seed + 1, inference=inference)
            second = MCTSAgent(model, self.encoder, simulations, seed + 2, inference=inference)
            report = runner.run(1, first, second, nation=nation)
            return index, local_buffer.examples, report, runner.game_records[-1]

        completed: dict[int, tuple[list[TrainingExample], SelfPlayReport, dict]] = {}
        with BatchedInference(model, max_batch_size=max_batch_size, max_wait_ms=max_wait_ms) as inference:
            with ThreadPoolExecutor(max_workers=min(environments, episodes)) as pool:
                futures = [pool.submit(play, index, seed, inference) for index, seed in enumerate(seeds)]
                for future in as_completed(futures):
                    index, examples, report, record = future.result()
                    completed[index] = (examples, report, record)

        totals = {"p1": 0, "p2": 0, "draw": 0}
        turns = hq_damage = 0.0; cards_played = 0
        for index in range(episodes):
            examples, report, record = completed[index]
            self.buffer.examples.extend(examples)
            if len(self.buffer.examples) > self.buffer.capacity:
                del self.buffer.examples[:len(self.buffer.examples) - self.buffer.capacity]
            totals["p1"] += report.p1_wins; totals["p2"] += report.p2_wins; totals["draw"] += report.draws
            turns += report.average_turns; hq_damage += report.average_hq_damage; cards_played += report.average_cards_played
        return SelfPlayReport(episodes, sum(len(item[0]) for item in completed.values()), totals["p1"], totals["p2"], totals["draw"],
                              turns / episodes, hq_damage / episodes, cards_played / episodes)

    def run_parallel(self, episodes: int, model, simulations: int, card_path: str | Path, nation: str = "France", workers: int = 2,
                     device: str = "cuda", on_episode_complete: Callable[[int, dict[str, int], int], None] | None = None) -> SelfPlayReport:
        """Generate independent episodes concurrently without changing MCTS policy or rules.

        Results are merged in episode-index order, keeping replay ordering deterministic
        for a fixed seed even when workers finish at different times.
        """
        if workers < 1:
            raise ValueError("workers must be positive")
        before = len(self.buffer); totals = {"p1": 0, "p2": 0, "draw": 0}
        turns_total = 0; hq_damage_total = 0.0; cards_played_total = 0
        episode_seeds = [self.rng.randrange(2**31) for _ in range(episodes)]
        pending: dict[int, tuple[list[TrainingExample], SelfPlayReport, dict]] = {}
        next_index = 0
        def merge(index: int, examples: list[TrainingExample], report: SelfPlayReport, record: dict) -> None:
            nonlocal next_index, turns_total, hq_damage_total, cards_played_total
            pending[index] = (examples, report, record)
            while next_index in pending:
                examples, report, record = pending.pop(next_index)
                self.buffer.examples.extend(examples)
                if len(self.buffer.examples) > self.buffer.capacity:
                    del self.buffer.examples[:len(self.buffer.examples) - self.buffer.capacity]
                winner = "p1" if report.p1_wins else "p2" if report.p2_wins else "draw"
                totals[winner] += 1; turns_total += report.average_turns
                hq_damage_total += report.average_hq_damage; cards_played_total += report.average_cards_played
                self.game_records.append(record)
                next_index += 1
                if on_episode_complete:
                    on_episode_complete(next_index, totals.copy(), len(self.buffer) - before)

        if str(device).startswith("cuda"):
            # GPU contexts are process-local. Keep all CUDA work in the
            # parent, while workers use separate CPU processes for rules.
            from ai.inference import ProcessInferenceService
            context = mp.get_context("spawn")
            requests, tasks, results = context.Queue(), context.Queue(), context.Queue()
            responses = [context.Queue() for _ in range(workers)]
            service = ProcessInferenceService(model, requests, responses)
            processes = [context.Process(target=_cuda_worker_loop,
                                         args=(str(card_path), model.architecture, simulations, self.max_actions,
                                               requests, responses[index], index, tasks, results), daemon=True)
                         for index in range(workers)]
            for process in processes:
                process.start()
            try:
                for index, seed in enumerate(episode_seeds):
                    tasks.put((index, seed, nation))
                for _ in processes:
                    tasks.put(None)
                completed = 0
                while completed < episodes:
                    ok, payload = results.get()
                    if not ok:
                        raise RuntimeError(f"CUDA self-play worker failed: {payload}")
                    merge(*payload); completed += 1
            finally:
                service.close()
                for process in processes:
                    process.join(timeout=5)
                    if process.is_alive():
                        process.terminate(); process.join()
        else:
            cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            with ProcessPoolExecutor(max_workers=workers, initializer=_parallel_worker_init,
                                     initargs=(str(card_path), model.architecture, cpu_state,
                                               simulations, self.max_actions, device)) as pool:
                futures = [pool.submit(_parallel_episode, index, seed, nation) for index, seed in enumerate(episode_seeds)]
                for future in as_completed(futures):
                    merge(*future.result())
        return SelfPlayReport(episodes, len(self.buffer) - before, totals["p1"], totals["p2"], totals["draw"],
                              turns_total / episodes, hq_damage_total / episodes, cards_played_total / episodes)


# ``run_parallel`` predates VectorizedSelfPlay and remains part of the
# SelfPlayRunner public API.  The implementation is shared because both
# schedulers own the same cards/encoder/buffer/rng state.
SelfPlayRunner.run_parallel = VectorizedSelfPlay.run_parallel
