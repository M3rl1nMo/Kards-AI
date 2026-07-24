# AI Training Readiness Report

Generated: 2026-07-24

## 1. Simulator status

The simulator remains unchanged as the environment.  The AI layer only calls
`Simulator.reset`, `get_observation`, `get_available_actions`, `step`, and
state cloning supplied by the simulator.  The active catalog exposes 1,486
fully implemented, deck-eligible cards; the two partial rules are retired.

## 2. Agent status

`ai.agents` provides `BaseAgent`, `RandomAgent`, `RuleBasedAgent`,
`NeuralAgent`, and `MCTSAgent`.  Every agent consumes the same legal action
list.  MCTS/Neural agents receive a transient cloned state from the runner for
encoding/search; no simulator rules are changed.

## 3. Observation and action spaces

`ObservationEncoder` produces a fixed `float32[234]` player-relative tensor:
turn/current-player, both HQ/resource summaries, 8 unit slots per player, and
12 visible hand-card slots.  Opponent hand identities are never encoded.

`ActionEncoder` produces `float32[512,16]` candidate-action features plus a
`bool[512]` legal mask.  Pass, play/target/choice fields, attacks, movement,
abilities, countermeasures, and mulligans use the simulator's actual action
objects.  Padded slots are masked before softmax.

## 4. Network and MCTS

`KARDSNet` has independent state/action encoders, a candidate-scoring policy
head, and a `tanh` value head.  Checkpoints include architecture metadata and
weights.  `MCTS` uses PUCT selection, network priors/value when supplied,
expansion through cloned simulator states, and signed backpropagation.

## 5. Self-play, replay, and training

`SelfPlayRunner` creates legal decks, records `(state, legal action features,
legal mask, MCTS policy, final value)` records, and writes them through the
bounded pickle `ReplayBuffer`.  `Trainer` combines masked policy cross-entropy
with value MSE and selects CUDA, Apple MPS, then CPU.  `Evaluator` reports
candidate wins, losses, draws, and win rate against Random/RuleBased agents.

## 6. Verification

- Python 3.12 isolated runtime installed with PyTorch 2.13, NumPy, and PyYAML.
- `python -m unittest discover -s tests -v`: 190 passing tests.
- `python -m training.preflight`: 32 seeded episodes, 4,651 legal actions,
  31 terminal episodes, and no unsupported effect.
- `python selfplay.py --episodes 1` smoke configuration: 20 records written.
- AI tests verify agent/environment invocation, fixed shapes/mask, MCTS,
  checkpoint round-trip, self-play data creation, and a completed optimizer
  batch.

## 7. Start the first training run

```bash
cd "/Users/evangong/Documents/Kards AI"
.venv/bin/python selfplay.py
.venv/bin/python train.py
.venv/bin/python evaluate.py --model runs/kardsnet.pt --opponent random
```

The default self-play configuration uses 32 MCTS-guided episodes with 64 tree
simulations each.  Tune episode count, MCTS budget, batch size, updates and
paths in `configs/*.yaml`; checkpoints and replay data are excluded from git.

## Scope boundary

This proves the training framework operates with the current simulator.  It
does not claim live-game rules parity or competitive playing strength before
long self-play runs and evaluation against stronger external fixtures.
