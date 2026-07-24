# KARDS AI — Windows Codex Handoff

Read this file before changing or training the project.  It is the durable
replacement for the previous Codex conversation context.

## Repository and verified baseline

- Repository: `EvanProgramming/Kards-AI`
- Branch to use: `main`
- Handoff baseline: `b7331ab` (`feat: make AI training launch-ready`)
- Project root is the directory containing `simulator/`, `ai/`, `tests/`,
  `selfplay.py`, `train.py`, and `evaluate.py`.
- The workspace must be clean before changes.  Preserve unrelated user edits.

The simulator is the environment.  Do **not** rewrite its card data or core
rules in order to change AI behavior.  Add AI-specific behavior in `ai/`.

## What is implemented

| Layer | Location | Status |
|---|---|---|
| Deterministic KARDS environment | `simulator/` | Existing environment; reset, legal actions, step, replay and state cloning |
| Agent interface | `ai/agents.py` | Base, Random, RuleBased, Neural and MCTS agents |
| Fixed player-view observation | `ai/observation.py` | `float32[234]`; opponent hand IDs are hidden |
| Legal candidate action encoding | `ai/action_encoder.py` | `float32[512,16]` + boolean legal mask |
| Policy/value model | `ai/network.py` | PyTorch `KARDSNet`, checkpoint save/load including architecture |
| PUCT search | `ai/mcts.py` | cloned simulator states, priors/value, selection/expansion/backpropagation |
| Self-play and data | `ai/selfplay.py`, `ai/replay_buffer.py` | serializable `(state, policy, value)` examples |
| Training | `ai/trainer.py` | masked policy loss + value MSE; CUDA, then CPU on Windows |
| Evaluation | `ai/evaluator.py` | win/loss/draw, win rate, turns, HQ damage, cards played |
| Entrypoints | `selfplay.py`, `train.py`, `evaluate.py` | usable command-line workflow |

The current card policy excludes retired cards; the two older partial rules are
retired and do not enter legal training decks.  Do not claim that passing
simulator/AI tests proves full parity with the live KARDS client.

## Windows setup

Use Python **3.12**.  The simulator itself is deliberately dependency-light;
the training environment must be isolated in `.venv`.

```powershell
git clone https://github.com/EvanProgramming/Kards-AI.git
cd Kards-AI
git checkout main
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
```

For NVIDIA GPU training, install the PyTorch wheel appropriate to the installed
CUDA version from the official PyTorch selector, then verify:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

`Trainer` automatically uses CUDA when available; otherwise it uses CPU.
Apple MPS is naturally unavailable on Windows.

## Mandatory preflight after moving machines

Run these before starting any real training.  Do not skip failures.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m training.preflight
```

Expected baseline: **191 tests pass**.  Preflight should report 32 seeded
episodes, 1,486 eligible cards, and no unsupported effects.

## Start training

Small first-run command, intended to prove the new machine works:

```powershell
.\.venv\Scripts\python.exe selfplay.py --episodes 32 --mcts-simulations 64 --replay runs\replay.pkl
.\.venv\Scripts\python.exe train.py --replay runs\replay.pkl --updates 100 --batch-size 64 --checkpoint runs\kardsnet.pt
.\.venv\Scripts\python.exe evaluate.py --model runs\kardsnet.pt --games 50 --opponent rule --report runs\evaluation.json
```

`runs/` is deliberately ignored by Git.  Copy it manually between machines if
you need to resume a training run; it contains replay data, checkpoints, and
evaluation reports.

## Useful controls

- `selfplay.py`: `--episodes`, `--mcts-simulations`, `--max-actions`,
  `--nation`, `--replay`, `--model`, `--seed`.
- `train.py`: `--replay`, `--model`, `--batch-size`, `--updates`,
  `--learning-rate`, `--checkpoint`, `--device`.
- `evaluate.py`: `--model`, `--games`, `--opponent random|rule`, `--report`.
- Defaults are versioned in `configs/selfplay.yaml` and
  `configs/training.yaml`.

## Rules for the next Codex

1. Start with `git status`, then run the preflight on the new machine.
2. Do not modify `data/source/` or simulator rules to make a training test
   easier.  Add adapters or AI code instead.
3. Preserve opponent hidden information: never add opponent hand IDs to the
   observation encoder.
4. Never select an action outside `Simulator.get_available_actions()`; the
   action mask is authoritative.
5. Keep checkpoint/replay metadata and evaluation reports when a model is
   trained.  Report actual win rates, not only loss values.
6. Commit focused changes and push only after relevant tests pass.

## Current readiness statement

The repository is ready to start its first self-play training run after the
Windows preflight succeeds.  This is training-framework readiness, not a claim
that a newly trained model is already competitive or that the simulator has
full live-game parity.
