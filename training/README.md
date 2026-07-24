# AI training preparation

This directory intentionally contains no command that starts model training.
It establishes the admission gate for the next phase.

## Target

Train a two-player, imperfect-information KARDS policy through self-play.  The
first model should score the simulator's *current legal action candidates*, not
an artificial global action index: play/target/move/attack choices naturally
have variable arity and the candidate list already performs rules validation.
The model will use a state encoder plus an action encoder and will select only
from that candidate list.  This is compatible with masked policy-gradient
training and later MCTS-guided self-play.

## Before the first optimization step

1. Use Python 3.12 in an isolated `.venv` (the workstation default is Python
   3.14, for which the ML wheels are not part of this setup).
2. Install `requirements-training.txt` into that environment.
3. Run `python -m training.preflight` and retain its printed report with the
   training run metadata.
4. Start with mirrored France-only legal decks, fixed seeds, and the active
   rule pool.  `eligible_card_ids()` fails closed if a non-implemented card is
   ever made deck-legal.
5. Record simulator commit, Python/package versions, config, seeds, win rate
   versus the random baseline, and checkpoint hashes for every run.

## Admission criteria

The preflight must complete without an `unsupported_effect`, invalid action,
empty non-terminal action set, or nondeterministic replay discrepancy.  It is
a safety gate, not a claim of full live-KARDS parity; real-game fixture checks
remain necessary before presenting a trained model as a production bot.

## Explicitly deferred until approval

- Creating the virtual environment and installing PyTorch/NumPy.
- Generating self-play trajectories.
- Updating neural-network weights or writing checkpoints.
