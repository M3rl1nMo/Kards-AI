# Handler Ledger — 80 Engine-Executable Action Kinds

Last regenerated: 2026-07-18. Source: `extract_executable_kinds()` in
`tools/build_card_rules.py` (regex scan of `simulator/rules/native.py` +
`simulator/effects/resolver.py`).

## Summary

| Metric | Count |
|---|---|
| Total executable kinds (engine dispatches) | **79** meaningful (+1 string- literal artifact filtered) |
| In `native.py` (NativeRuleEngine) | 47 |
| In `resolver.py` (EffectResolver) | 21 |
| Dispatched in both | 12 |
| Covered by regression tests | 14 |
| Untested | 65 |

## Dispatch Table

Test column: `yes` = at least one test asserts the kind or exercises the handler
in `tests/test_native_rules.py`, `tests/test_choose_one.py`, or `tests/test_cost_mechanics.py`.

| Kind | Dispatch | Tested |
|---|---|---|
| add_card | native.py | yes |
| add_keyword | resolver.py | — |
| attack_bonus_against_higher_attack | native.py | — |
| attack_bonus_against_type | native.py | — |
| attack_equals_defense | native.py | — |
| attack_on_order_played | native.py | — |
| buff | native.py + resolver.py | — |
| buff_nation | native.py | — |
| buff_on_enemy_target | native.py | — |
| buff_on_played_ability | native.py | — |
| cannot_attack_hq | native.py | — |
| cannot_be_targeted_by_enemy_orders | native.py | yes |
| choice | resolver.py | — |
| choose_one | native.py | yes |
| compare_resources | resolver.py | — |
| convert_to | native.py | yes |
| copy_card | resolver.py | — |
| copy_unit | native.py | — |
| damage | native.py + resolver.py | yes |
| damage_bonus_against_air | native.py | — |
| debuff | resolver.py | — |
| defense_nation | native.py | — |
| delayed_discard | native.py | — |
| delayed_return | native.py | — |
| deploy_named | native.py | — |
| destroy | native.py + resolver.py | yes |
| destroy_cost_lte | native.py | — |
| discard | native.py + resolver.py | yes |
| discard_hand | native.py | — |
| double_damage_against_type | native.py | yes |
| draw | resolver.py | yes |
| draw_matching | native.py | — |
| draw_top_matching | native.py | — |
| end_turn | native.py | — |
| enemy_unit_exists | resolver.py | — |
| fight | native.py | — |
| friendly_unit_exists | resolver.py | — |
| gain_hq_defense | resolver.py | — |
| gain_hq_defense_equal_damage | native.py | — |
| gain_kredit_slot | native.py | — |
| gain_kredits | native.py | — |
| grant_ability | native.py | — |
| has_keyword | resolver.py | — |
| has_unit_type | resolver.py | — |
| heal | native.py + resolver.py | — |
| immune_to_ground_in_support | native.py | — |
| lose_kredit_slot | native.py | — |
| mill | native.py | — |
| modify_attack | native.py + resolver.py | — |
| modify_cost | resolver.py | — |
| modify_defense | native.py + resolver.py | — |
| modify_deployment_cost | resolver.py | — |
| modify_hand_cost | native.py | — |
| modify_health | resolver.py | — |
| modify_operation_cost | resolver.py | yes |
| move | resolver.py | — |
| move_to_frontline | native.py + resolver.py | — |
| move_to_support_line | native.py + resolver.py | — |
| op_cost_rule | native.py | — |
| order_damage_bonus | native.py + resolver.py | — |
| random | resolver.py | — |
| remove | native.py | — |
| remove_keyword | resolver.py | — |
| repair | native.py | — |
| reset_operation | native.py | — |
| retreat_and_repair | native.py | — |
| return_to_deck | native.py | yes |
| return_to_hand | native.py | — |
| set_attack | native.py | — |
| set_defense | native.py | yes |
| set_operation_cost | native.py | yes |
| shuffle_named | native.py | — |
| spawn | resolver.py | — |
| spawn_named_same_front | native.py | — |
| spawn_unit | resolver.py | — |
| suppress | native.py + resolver.py | — |
| take_control | native.py | — |
| target_or_attack_tax | native.py | yes |
| transform | resolver.py | — |

## Category Grouping

| Category | Kind count | Examples |
|---|---|---|
| Cost / Kredit | 10 | modify_hand_cost, gain_kredits, op_cost_rule, ... |
| Draw / Discard | 8 | draw, discard, mill, return_to_deck, ... |
| Stat / Combat | 26 | buff, damage, modify_attack, spawn, fight, ... |
| Move / Position | 5 | move, move_to_frontline, reset_operation, ... |
| Target / Choice / Copy | 11 | choose_one, random, convert_to, copy_card, ... |
| HQ | 2 | gain_hq_defense, gain_hq_defense_equal_damage |
| Keyword / Ability | 7 | add_keyword, grant_ability, suppress, ... |
| Temporal / Zone | 9 | end_turn, destroy, delayed_discard, ... |

## Test Coverage Gap

65 of 79 kinds have **no dedicated regression test** asserting their engine
handler produces the correct state mutation and event log. The 14 tested kinds
are: `add_card`, `cannot_be_targeted_by_enemy_orders`, `choose_one`, `convert_to`,
`damage`, `destroy`, `discard`, `double_damage_against_type`, `draw`,
`modify_operation_cost`, `return_to_deck`, `set_defense`, `set_operation_cost`,
`target_or_attack_tax`.

**Per the task book acceptance criteria**: *"每条通用 Action 与目标选择器至少一个测试"*
(each generic Action and target selector requires at least one test). This means
65 kinds currently fall short of the acceptance bar.

## How to Read

- `native.py` dispatch: `NativeRuleEngine._execute_action()` — an `if action.kind == "X"`
  or `if action.kind in {...}` block that mutates state directly.
- `resolver.py` dispatch: `EffectResolver._resolve_action()` — a `if kind == "X"`
  or `elif kind in {...}` block that applies generic effects via target selectors.
- `both`: the kind is recognised in both files. The actual execution path
  depends on which engine handles the event (`NativeRuleEngine.emit()` routes
  to native, `EffectResolver.emit()` to resolver).
- Regenerate this ledger anytime the engine dispatch changes: run
  `extract_executable_kinds()` from `tools/build_card_rules.py` and diff.
