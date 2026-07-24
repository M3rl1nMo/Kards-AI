# KARDS AI Simulator

For a fresh Codex or Windows training-machine setup, read
[`CODEX_HANDOFF_WINDOWS.md`](CODEX_HANDOFF_WINDOWS.md) first.

Core, headless rules simulator for KARDS AI training. This repository deliberately
contains no UI, networking, or client code.

## Card data

`data/source/kards_info_cards.json` is the simulator's only card-data source.
It is the unmodified kards.info catalog (1488 cards); its native names are used
throughout the code: `id`, `type`, `kredits`, `operationCost`, `text`, and
`abilities`. There is no legacy adapter, merged handler file, or parallel card
schema. Relative image paths are made absolute with `https://kards.info`.

Static catalog data is separate from runtime game state. Native `text` and
`abilities` enter the event pipeline through `RuleParser`; `EffectResolver` and
`HandlerRegistry` remain the reusable game-rule layer.

## Development phases

- [x] Phase 1: card loader and serializable game state
- [x] Phase 2: actions and turn flow
- [x] Phase 3: effect engine
- [x] Phase 4: custom-handler framework
- [x] Phase 5: simulation test environment
- [x] Phase 6.1: headquarters model and victory resolution
- [x] Phase 6.2–6.7: battlefield rules, turn triggers, operations, keywords, deck validation, replay, audit, countermeasures, and scale-test tooling

Run the current zero-dependency test suite with `python3 -m unittest discover -s tests -v`.
