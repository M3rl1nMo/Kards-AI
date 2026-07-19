# KARDS Simulator Event-Order Contract

The deterministic, tested event lifecycle that the rule engine depends on.
Every event below fires exactly once per qualifying action and in the order
listed.  No rule fires without a matching event; no event is skipped silently.

## Full Lifecycle (task book step 5)

```
Draw -> Hand -> Validate legal actions -> Pay kredits -> Play card / Deploy unit
-> Emit events in defined order -> Resolve rules / effects
-> Process deaths & trigger chains -> Graveyard / active zones -> Next action
```

## Event Catalog

### 1. `on_card_drawn`
**Trigger point:** After `basic_effects.draw()` appends a card to the player's hand
and logs `card_drawn`.
**Dispatch:** `NativeRuleEngine.emit_draws_since()` (native.py).
**Rules fired:** Cards in hand / deck with "when drawn" / "for each draw" effects.
**Guarantees:**
- The drawn card is already in the hand when rules execute.
- Only one `on_card_drawn` broadcast per draw event.

### 2. `on_play` (orders)
**Trigger point:** `PlayCardAction.execute()` after validating, paying cost, and
discarding the order card from hand (action.py:122-131).
**Dispatch:** `NativeRuleEngine.emit("on_play")` then `EffectResolver.emit("on_play")`.
**Rules fired:** The played order's own text ("Deployment:" on orders → on_play trigger).
**Guarantees:**
- Cost is already paid; card is out of the hand.
- `on_play` fires BEFORE `on_command_played` (countermeasure intercept).
- Countermeasure intercept (`CountermeasureResolver.intercept`) fires after `on_play`.

### 3. `on_deploy` (units / countermeasures)
**Trigger point:** `PlayCardAction.execute()` after creating the `UnitState`,
adding it to `player.units`, and paying cost (action.py:105-117).
**Dispatch:** `NativeRuleEngine.emit("on_deploy")` then `EffectResolver.emit("on_deploy")`.
**Rules fired:** The deployed card's own text ("Deployment:" on units → on_deploy).
**Guarantees:**
- The unit instance exists in `player.units` with a unique `instance_id`.
- `EffectContext.source_unit_id` is set (used for lifetime-bound cost modifiers).
- `on_deploy` fires BEFORE `on_friendly_card_played`.

### 4. `on_friendly_card_played`
**Trigger point:** After `on_deploy` / `on_play` (action.py:116, 130).
**Dispatch:** `NativeRuleEngine.emit("on_friendly_card_played")`.
**Rules fired:** Cards with "when you play/deploy a card" / "when a friendly card is played."
**Guarantees:**
- The played card is already in play (unit on battlefield, order discarded).
- Fires after the played card's own `on_deploy`/`on_play`.

### 5. `on_command_played`
**Trigger point:** After `on_play`, before `on_friendly_card_played`, for orders only
(action.py:119-121 for countermeasure intercept; action.py:129 for engine).
**Dispatch:** `NativeRuleEngine.emit("on_command_played")`.
**Rules fired:** Countermeasure activation rules, "when an order is played."

### 6. `on_targeted_by_enemy_effect`
**Trigger point:** `PlayCardAction.execute()` when `target_unit_id` is set, after
the card is in play (action.py:114, 127).
**Dispatch:** `NativeRuleEngine.emit("on_targeted_by_enemy_effect")`.
**Rules fired:** Cards with "when this unit is targeted by enemy effect" / target tax.

### 7. `on_attack`
**Trigger point:** `AttackAction.execute()` before combat resolution (action.py).
**Dispatch:** `NativeRuleEngine.emit("on_attack")`.
**Rules fired:** Attacker and defender "on attack" / "when this unit attacks" triggers.
**Guarantees:**
- Both attacker and defender are alive and on the battlefield.
- Fires BEFORE damage and combat resolution.

### 8. `on_targeted_by_enemy_attack`
**Trigger point:** `AttackAction.execute()` when an attack is declared against a
specific unit (action.py).
**Dispatch:** `NativeRuleEngine.emit("on_targeted_by_enemy_attack")`.
**Rules fired:** Defender "when targeted by an attack."

### 9. `on_damage_dealt`
**Trigger point:** After `basic_effects.apply_damage()` records a `damage_dealt`
event (basic_effects.py:13).
**Dispatch:** `NativeRuleEngine.emit("on_damage_dealt")`.
**Rules fired:** Cards with "when this unit deals damage."
**Guarantees:**
- Damage amount is recorded in the event log.
- Fires even if the target dies from the damage.

### 10. `on_damage`
**Trigger point:** `NativeRuleEngine.emit_damage_since()` for surviving units
that took damage (native.py:167-178).
**Dispatch:** `NativeRuleEngine.emit("on_damage")`.
**Rules fired:** Cards with "when this unit takes damage" (surviving only).
**Guarantees:**
- Only fires for units that survived the damage event.
- The unit is still on the battlefield.

### 11. `on_combat_damage_dealt`
**Trigger point:** After combat-specific damage is applied.
**Dispatch:** `NativeRuleEngine.emit("on_combat_damage_dealt")`.
**Rules fired:** Cards with "when this unit deals combat damage."

### 12. `on_survives_combat`
**Trigger point:** After combat, when the attacker survives (action.py).
**Dispatch:** `NativeRuleEngine.emit("on_survives_combat")`.
**Rules fired:** Cards with "after this unit survives combat" / "if this unit
survives combat, ...".

### 13. `on_enemy_hq_damaged`
**Trigger point:** After damage is applied to the enemy HQ.
**Dispatch:** `NativeRuleEngine.emit("on_enemy_hq_damaged")`.
**Rules fired:** Cards with "when enemy HQ is damaged" / "when you damage HQ."

### 14. `on_destroy`
**Trigger point:** `NativeRuleEngine.emit_deaths_since()` after a `unit_died`
event is logged (native.py:144-154).
**Dispatch:** `NativeRuleEngine.execute(card_id, "on_destroy", ...)` per death.
**Rules fired:** The destroyed card's own "Destruction:" text.
**Guarantees:**
- The unit is dead (0 HP) but its card_id is still known.
- Continuous effects cleanup (`_cleanup_continuous_effects`) fires AFTER
  `on_destroy` handlers complete — the unit's cost modifiers persist during
  `on_destroy` and are reverted afterward.
- Death chaining uses a stable snapshot (`emit_deaths_since` iterates over deaths
  logged during the current batch; new deaths added by triggers are logged
  separately and do not re-trigger).
- No duplicate triggers per death.

### 15. `on_turn_start` / `on_friendly_turn_start`
**Trigger point:** `TurnManager.start_turn()` at the beginning of each player's turn.
**Dispatch:** `NativeRuleEngine.emit("on_turn_start")` then `emit("on_friendly_turn_start")`.
**Rules fired:** Cards with "at the start of your turn" / "start of turn."
**Guarantees:**
- Fires after turn counter advancement and temporary-effect cycle processing.
- Kredit refresh happens before rules execute.

### 16. `on_turn_end` / `on_friendly_turn_end`
**Trigger point:** `TurnManager.end_turn()` at the end of each player's turn.
**Dispatch:** `NativeRuleEngine.emit("on_turn_end")` then `emit("on_friendly_turn_end")`.
**Rules fired:** Cards with "at the end of your turn" / "end of turn."
**Guarantees:**
- Temporary-effect expiry (`_expiry_turn` check in cost_modifiers) fires BEFORE
  `on_turn_end` — expired modifiers are removed first, then end-of-turn rules
  execute with the post-expiry state.
- Fires before the turn passes to the opponent.

## Ordering Guarantees (per action)

### Play a unit: `validate -> pay -> UnitState created -> on_deploy -> on_friendly_card_played -> next action`
### Play an order: `validate -> pay -> on_play -> (countermeasure intercept) -> on_command_played -> on_friendly_card_played -> next action`
### Attack: `validate -> on_attack -> on_targeted_by_enemy_attack -> resolve combat -> on_damage_dealt -> on_combat_damage_dealt -> death check -> on_destroy (per death) -> on_survives_combat -> next action`
### Turn boundary: `temporary expiry -> on_turn_end -> on_friendly_turn_end -> pass -> on_turn_start -> on_friendly_turn_start -> kredit refresh`

## Event Dispatch Sites

| Event | Site | Module |
|---|---|---|
| on_card_drawn | `NativeRuleEngine.emit_draws_since` | native.py |
| on_play | `NativeRuleEngine.emit` + `EffectResolver.emit` | native.py + resolver.py |
| on_deploy | `NativeRuleEngine.emit` + `EffectResolver.emit` | native.py + resolver.py |
| on_friendly_card_played | `NativeRuleEngine.emit` | native.py |
| on_command_played | `NativeRuleEngine.emit` | native.py |
| on_targeted_by_enemy_effect | `NativeRuleEngine.emit` | native.py |
| on_attack | `NativeRuleEngine.emit` | native.py |
| on_targeted_by_enemy_attack | `NativeRuleEngine.emit` | native.py |
| on_damage_dealt | `NativeRuleEngine.emit` | native.py |
| on_damage | `NativeRuleEngine.emit_damage_since` | native.py |
| on_combat_damage_dealt | `NativeRuleEngine.emit` | native.py |
| on_survives_combat | `NativeRuleEngine.emit` | native.py |
| on_enemy_hq_damaged | `NativeRuleEngine.emit` | native.py |
| on_destroy | `NativeRuleEngine.emit_deaths_since` | native.py |
| on_turn_start / on_friendly_turn_start | `TurnManager` | core/turn.py |
| on_turn_end / on_friendly_turn_end | `TurnManager` | core/turn.py |

## Verifiability

This contract is verified by:
- Regression tests in `tests/test_native_rules.py` asserting specific event-log
  ordering for deploy/attack/destroy/end-turn.
- `tests/test_choose_one.py` asserting triggered card rules fire in order.
- `tests/test_cost_mechanics.py` asserting cost modifiers apply and clean up
  at the correct lifecycle points.
- The `git diff --check` + `unittest discover` gate before every milestone commit.
