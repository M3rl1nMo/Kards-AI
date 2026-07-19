# Simulator Training Readiness Report

Generated: 2026-07-18

## Current status

This simulator is a **partially executable KARDS environment**, not yet a
rules-parity environment suitable for production MCTS/RL training.

### Native rule coverage

| Status | Cards |
|---|---:|
| Implemented | 426 |
| Partial | 163 |
| Unresolved | 899 |

The coverage counts are parser-derived and conservative: unresolved text is not
silently executed. See `RULE_IMPLEMENTATION_REPORT.md` for the original
breakdown; it must be regenerated after each parser expansion.

### Ability coverage

| Ability | Status |
|---|---|
| Blitz, Fury, Heavy Armor, Guard | Partial combat support |
| Ambush, Smokescreen | Implemented in combat/movement flow |
| Shock, Alpine, Covert, Mobilize, Intel, Salvage, Bond | Not implemented |

### Countermeasures

Activation, hand retention, event matching, trigger discard, and turn expiry
have a base implementation. Trigger predicates, cancellation semantics, target
selection, and the majority of card effects remain incomplete.

### AI environment

- `reset()` now enforces the 40-card/Nation/copy/token checks through
  `DeckValidator`.
- `get_observation(player_id)` hides opponent hand IDs.
- Action generation includes basic plays, targets inferred from supported ASTs,
  combat, movement, pass, and an empty Mulligan action.
- Choice, activated ability, complete countermeasure, and full target action
  interfaces are not complete.

## Stability evidence

- Unit/regression tests: 13 tests passed.
- Deterministic legal smoke games: 1,000 / 1,000 completed without crash.
- This verifies state/action stability only; it does not prove real-rule parity.

## Training decision

**Do not start production Agent/MCTS/RL training yet.** It is appropriate only
for API integration or synthetic-environment experiments. The 899 unresolved
cards and incomplete ability/countermeasure semantics would systematically
teach an agent incorrect transitions.

## Required before production training

1. Convert every remaining card text to a verified AST or reviewed reusable
   handler, with an executable per-card regression test.
2. Complete all catalog abilities and combat event ordering.
3. Complete Countermeasure predicates, cancellation, choices and delayed rules.
4. Add target/hidden-information legality and deterministic action masks.
5. Run parity fixtures against known KARDS interactions, then large-scale games
   with no unresolved rule that can affect a selected deck pool.
