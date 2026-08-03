# KARDS AI

An open-source **KARDS AI** project for building a headless card-game simulator, self-play training pipeline, and reinforcement-learning agents for [KARDS](https://www.kards.com/), the World War II collectible card game.

> **Project status: training is not complete.** The current training run is at approximately **1 million games** and remains an active experiment. The model should not yet be treated as a finished or competitive KARDS player.

## What this project does

KARDS AI combines a JSON-driven rules simulator with AlphaZero-style reinforcement learning:

- Headless KARDS game simulation for repeatable self-play
- Card and rule data loading from the KARDS card catalog
- Legal-action masking and fixed observation/action encoders
- Random, rule-based, neural-network, and PUCT Monte Carlo Tree Search agents
- Parallel self-play, replay-buffer management, policy/value training, and evaluation
- Checkpointing, metrics, monitoring, and training-control utilities
- A modular foundation for improving simulator fidelity and transferring learning toward real KARDS gameplay

## Current training status

The AI training pipeline is runnable, but the overall project is unfinished. Approximately **1,000,000 self-play games** have been reached in the current training effort. This milestone demonstrates training progress, not completion: simulator-rule coverage, real-client parity, strategic strength, evaluation breadth, and long-term training stability still require further work.

Use the readiness reports for the latest limitations and validation evidence:

- [`AI_TRAINING_READINESS_REPORT.md`](AI_TRAINING_READINESS_REPORT.md)
- [`CODEX_HANDOFF_WINDOWS.md`](CODEX_HANDOFF_WINDOWS.md)
- [`docs/simulator_performance.md`](docs/simulator_performance.md)

## Quick start

Create a Python environment and install the training dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-training.txt
```

Run the test suite:

```bash
python -m unittest discover -s tests -q
```

Run a small self-play experiment, train a checkpoint, and evaluate it:

```bash
python selfplay.py --games 10
python train.py --config configs/training.yaml
python evaluate.py --config configs/training.yaml
```

For the long-running training workflow, monitoring, Windows setup, and artifact locations, see [`CODEX_HANDOFF_WINDOWS.md`](CODEX_HANDOFF_WINDOWS.md).

## Scope and limitations

This repository is a research and engineering project. It contains the simulator and AI training code; it does not provide a KARDS game client, online matchmaking, networking automation, or a guarantee of exact real-client behavior. Simulator-side tests and self-play results must not be interpreted as proof of competitive performance in the live game.

## Keywords

KARDS AI · KARDS bot research · KARDS simulator · World War II card game · reinforcement learning · AlphaZero · self-play · Monte Carlo Tree Search · PUCT MCTS · Python · PyTorch · game AI · collectible card game AI

## License

See the repository files for the applicable project and data licensing information.
