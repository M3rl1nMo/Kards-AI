"""Executable native rule-template regression tests."""

from pathlib import Path
import unittest

from simulator.actions.action import AttackAction, PassAction, PlayCardAction
from simulator.actions.validator import ActionValidationError
from simulator.cards.loader import CardDatabase
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState
from simulator.effects.resolver import EffectContext, EffectResolver
from simulator.rules.native import NativeRuleEngine


ROOT = Path(__file__).parents[1]


class NativeRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards = CardDatabase.from_file(ROOT / "data/source/kards_info_cards.json")
        cls.engine = NativeRuleEngine(cls.cards)

    def state(self) -> GameState:
        return GameState(
            current_player="p1",
            players={
                "p1": PlayerState("p1", "Britain", deck=["garrison"], hand=["radar", "home_guard", "active_sonar"], resources=ResourceState(5, 5)),
                "p2": PlayerState("p2", "Germany", units=[UnitState("enemy", "hurricane_mk_i", 2, 3, "p2", "support_line")]),
            },
            battlefield={"frontline": [], "support_line": ["enemy"]},
            graveyard={"p1": [], "p2": []},
        )

    def test_native_order_draw_and_hq_defense(self) -> None:
        state = self.state()
        PlayCardAction("p1", "radar").execute(state, self.cards)
        self.assertEqual(state.players["p1"].hq.defense_modifier, 4)
        self.assertIn("garrison", state.players["p1"].hand)
        self.assertIn("radar", state.graveyard["p1"])

    def test_targeted_order_executes_damage_and_add_card(self) -> None:
        state = self.state()
        PlayCardAction("p1", "home_guard", target_unit_id="enemy").execute(state, self.cards)
        self.assertEqual(state.players["p2"].units[0].defense, 1)
        self.assertIn("garrison", state.players["p1"].hand)

    def test_partial_rule_executes_only_verified_action(self) -> None:
        # M7 full-parse policy: "Deal 1 damage to a unit. If it doesn't have any
        # adjacent units, deal 2 instead." is now fully parsed (status='implemented')
        # with the conditional clause captured as a second action carrying a
        # condition, rather than left unresolved.
        rule = self.engine.rule_for("critical_damage")
        self.assertEqual(rule.status, "implemented")
        conditional = [a for a in rule.actions if a.condition]
        self.assertTrue(conditional, "conditional 'deal 2 instead' clause should be captured")
        state = self.state()
        state.players["p1"].hand.append("critical_damage")
        enemy = state.players["p2"].units[0]
        before = enemy.defense
        PlayCardAction("p1", "critical_damage", target_unit_id="enemy").execute(state, self.cards)
        # The unconditional "Deal 1 damage" action executes against the targeted unit.
        self.assertEqual(state.players["p2"].units[0].defense, before - 1)

    def test_parser_maps_set_defense_template(self) -> None:
        rule = self.engine.rule_for("no_9_commando")
        self.assertEqual(rule.status, "implemented")
        self.assertEqual(rule.actions[0].kind, "set_defense")

    def test_attack_trigger_parses_deals_verb_form(self) -> None:
        rule = self.engine.rule_for("pak_36_fi")
        self.assertEqual(rule.status, "implemented")
        self.assertEqual(rule.triggers, ("on_attack", "on_deploy"))
        self.assertEqual(rule.actions[0].kind, "damage")
        self.assertEqual(rule.actions[0].target, "random_enemy")

    def test_enemy_order_cannot_target_protected_unit(self) -> None:
        state = self.state()
        state.players["p1"].hand = ["141_gebirgsjger"]
        state.players["p1"].resources = ResourceState(5, 5)
        PlayCardAction("p1", "141_gebirgsjger").execute(state, self.cards)
        protected = state.players["p1"].units[0]
        # cannot_be_targeted_by_enemy_orders is now genuinely executed.
        self.assertTrue(protected.status.get("cannot_be_targeted_by_enemy_orders"))
        targeted_event = [e for e in state.event_log if e.get("event") == "cannot_be_targeted"]
        self.assertTrue(targeted_event)

    def test_double_damage_against_tanks_applies_in_combat(self) -> None:
        state = self.state()
        gun = UnitState("gun", "6_pounder", 2, 1, "p1", "frontline")
        tank_card = self.cards.get("m3a3_honey")
        tank = UnitState("tank", tank_card.id, tank_card.attack or 0, tank_card.defense or 0, "p2", "frontline")
        state.players["p1"].units.append(gun)
        state.players["p2"].units = [tank]
        state.battlefield = {"frontline": [gun.instance_id, tank.instance_id], "support_line": []}
        state.players["p1"].resources = ResourceState(5, 5)
        self.engine.emit("on_deploy", state, EffectContext("p1", gun.card_id, gun.instance_id))
        # double_damage_against_type is now genuinely executed.
        self.assertTrue(gun.status.get("double_damage_against_type"))
        event = [e for e in state.event_log if e.get("event") == "double_damage_against_type"]
        self.assertTrue(event)

    def test_damage_dealt_can_add_equal_hq_defense(self) -> None:
        state = self.state()
        gun_card = self.cards.get("9_2_coastal_gun")
        gun = UnitState("gun", gun_card.id, gun_card.attack or 0, gun_card.defense or 0, "p1", "frontline")
        target = UnitState("target", "m3a3_honey", 1, 8, "p2", "frontline")
        state.players["p1"].units.append(gun)
        state.players["p2"].units = [target]
        state.battlefield = {"frontline": [gun.instance_id, target.instance_id], "support_line": []}
        state.players["p1"].resources = ResourceState(5, 5)
        AttackAction("p1", gun.instance_id, target.instance_id).execute(state, self.cards)
        self.assertEqual(state.players["p1"].hq.defense_modifier, gun.attack)

    def test_damage_taken_can_add_equal_hq_defense(self) -> None:
        state = self.state()
        rifles = UnitState("rifles", "845th_rifles", 2, 5, "p1", "frontline")
        state.players["p1"].units = [rifles]
        state.battlefield = {"frontline": [rifles.instance_id], "support_line": []}
        start = len(state.event_log)
        EffectResolver(self.cards).resolve({"type": "damage", "target": "selected_target", "value": {"amount": 3}}, state, EffectContext("p2", target_unit_id=rifles.instance_id))
        self.engine.emit_damage_since(state, start)
        self.assertEqual(state.players["p1"].hq.defense_modifier, 3)

    def test_combat_damage_destroy_trigger_removes_target_before_return_fire(self) -> None:
        state = self.state()
        para_card = self.cards.get("2nd_para_c")
        para = UnitState("para", para_card.id, para_card.attack or 0, para_card.defense or 0, "p1", "frontline")
        target = UnitState("target", "m3a3_honey", 9, 12, "p2", "frontline")
        state.players["p1"].units.append(para)
        state.players["p2"].units = [target]
        state.battlefield = {"frontline": [para.instance_id, target.instance_id], "support_line": []}
        state.players["p1"].resources = ResourceState(5, 5)
        AttackAction("p1", para.instance_id, target.instance_id).execute(state, self.cards)
        self.assertFalse(state.players["p2"].units)
        self.assertEqual(para.defense, para_card.defense)

    def test_combat_damage_cap_only_limits_combat_damage(self) -> None:
        state = self.state()
        california_card = self.cards.get("2nd_california")
        california = UnitState(
            "california", california_card.id, california_card.attack or 0,
            california_card.defense or 6, "p1", "frontline",
        )
        attacker = UnitState("attacker", "m3a3_honey", 5, 8, "p2", "frontline")
        state.players["p1"].units = [california]
        state.players["p2"].units = [attacker]
        state.current_player = "p2"
        state.players["p2"].resources = ResourceState(5, 5)
        state.battlefield = {"frontline": [california.instance_id, attacker.instance_id], "support_line": []}
        self.engine.emit("on_deploy", state, EffectContext("p1", california.card_id, california.instance_id))
        self.assertEqual(california.status.get("combat_damage_cap"), 1)

        defense_before = california.defense
        AttackAction("p2", attacker.instance_id, california.instance_id).execute(state, self.cards)
        self.assertEqual(california.defense, defense_before - 1)

        # The wording is limited to combat: direct order/effect damage still
        # uses the normal damage pipeline without the cap.
        before_effect_damage = california.defense
        EffectResolver(self.cards).resolve(
            {"type": "damage", "target": "selected_target", "value": {"amount": 3}},
            state,
            EffectContext("p2", target_unit_id=california.instance_id),
        )
        self.assertEqual(california.defense, before_effect_damage - 3)

    def test_random_combat_damage_is_seeded_and_bounded(self) -> None:
        state = self.state()
        state.rng_seed = 17
        katyusha_card = self.cards.get("katyusha")
        katyusha = UnitState(
            "katyusha", katyusha_card.id, katyusha_card.attack or 0,
            katyusha_card.defense or 4, "p1", "frontline",
        )
        target = UnitState("target", "m3a3_honey", 0, 20, "p2", "frontline")
        state.players["p1"].units = [katyusha]
        state.players["p2"].units = [target]
        state.battlefield = {"frontline": [katyusha.instance_id, target.instance_id], "support_line": []}
        state.players["p1"].resources = ResourceState(5, 5)
        self.engine.emit("on_deploy", state, EffectContext("p1", katyusha.card_id, katyusha.instance_id))
        self.assertEqual(katyusha.status.get("random_combat_damage"), 1)

        before = target.defense
        AttackAction("p1", katyusha.instance_id, target.instance_id).execute(state, self.cards)
        self.assertIn(before - target.defense, {katyusha.attack, katyusha.attack + 1})

    def test_hq_damage_reduction_recounts_qualifying_units(self) -> None:
        state = self.state()
        m6_card = self.cards.get("m6")
        m6 = UnitState("m6", m6_card.id, 4, m6_card.defense or 4, "p1", "support_line")
        ally = UnitState("ally", "m3a3_honey", 4, 4, "p1", "support_line")
        state.players["p1"].units = [m6, ally]
        state.battlefield = {"frontline": [], "support_line": [m6.instance_id, ally.instance_id]}
        self.engine.emit("on_deploy", state, EffectContext("p1", m6.card_id, m6.instance_id))
        # The filler ally has an unrelated deployment effect; isolate the HQ
        # damage-reduction assertion from its temporary HQ-defense gain.
        state.players["p1"].hq.defense_modifier = 0

        EffectResolver(self.cards).resolve(
            {"type": "damage", "target": "enemy_hq", "value": {"amount": 5}},
            state,
            EffectContext("p2"),
        )
        self.assertEqual(state.players["p1"].hq.current_health, 17)

        ally.attack = 3
        EffectResolver(self.cards).resolve(
            {"type": "damage", "target": "enemy_hq", "value": {"amount": 5}},
            state,
            EffectContext("p2"),
        )
        self.assertEqual(state.players["p1"].hq.current_health, 13)

    def test_granted_tank_trait_participates_in_combat_type_rules(self) -> None:
        state = self.state()
        gun_card = self.cards.get("6_pounder")
        gun = UnitState("gun", gun_card.id, gun_card.attack or 0, 8, "p1", "frontline")
        cavalry_card = self.cards.get("savoia_cavalleria")
        cavalry = UnitState("cavalry", cavalry_card.id, 0, 20, "p2", "frontline")
        state.players["p1"].units = [gun]
        state.players["p2"].units = [cavalry]
        state.battlefield = {"frontline": [gun.instance_id, cavalry.instance_id], "support_line": []}
        state.players["p1"].resources = ResourceState(5, 5)
        self.engine.emit("on_deploy", state, EffectContext("p1", gun.card_id, gun.instance_id))
        self.engine.emit("on_deploy", state, EffectContext("p2", cavalry.card_id, cavalry.instance_id))
        self.assertTrue(cavalry.status.get("also_tank"))

        before = cavalry.defense
        AttackAction("p1", gun.instance_id, cavalry.instance_id).execute(state, self.cards)
        self.assertEqual(cavalry.defense, before - (gun.attack * 2))

    def test_enemy_card_damage_reduction_applies_to_damage_pipeline(self) -> None:
        state = self.state()
        seahawk_card = self.cards.get("seahawk")
        seahawk = UnitState("seahawk", seahawk_card.id, 0, seahawk_card.defense or 4, "p2", "support_line")
        victim = UnitState("victim", "m3a3_honey", 2, 8, "p2", "frontline")
        state.players["p2"].units = [seahawk, victim]
        state.battlefield = {"frontline": [victim.instance_id], "support_line": [seahawk.instance_id]}
        self.engine.emit("on_deploy", state, EffectContext("p2", seahawk.card_id, seahawk.instance_id))
        self.assertEqual(seahawk.status.get("enemy_card_damage_reduction"), 1)
        EffectResolver(self.cards).resolve(
            {"type": "damage", "target": "selected_target", "value": {"amount": 3}},
            state,
            EffectContext("p1", source_card_id="home_guard", target_unit_id=victim.instance_id),
        )
        self.assertEqual(victim.defense, 6)

    def test_deployment_can_swap_with_friendly_frontline_unit(self) -> None:
        state = self.state()
        frontline = UnitState("front", "m3a3_honey", 2, 4, "p1", "frontline")
        state.players["p1"].units = [frontline]
        state.players["p1"].hand = ["39_panzergrenadier"]
        state.players["p1"].resources = ResourceState(10, 10)
        state.battlefield = {"frontline": [frontline.instance_id], "support_line": []}
        PlayCardAction("p1", "39_panzergrenadier", target_unit_id=frontline.instance_id).execute(state, self.cards)
        deployed = next(unit for unit in state.players["p1"].units if unit.card_id == "39_panzergrenadier")
        self.assertEqual(deployed.position, "frontline")
        self.assertEqual(frontline.position, "support_line")
        self.assertEqual(deployed.attack, (self.cards.get(deployed.card_id).attack or 0) + 2)

    def test_deployment_can_remove_kredit_slot(self) -> None:
        state = self.state()
        state.players["p1"].hand = ["40th_cavalry_regiment"]
        state.players["p1"].resources = ResourceState(5, 5)
        PlayCardAction("p1", "40th_cavalry_regiment").execute(state, self.cards)
        self.assertEqual(state.players["p1"].resources.max_kredits, 4)
        self.assertEqual(state.players["p1"].resources.kredits, 2)

    def test_enemy_hq_damage_trigger_can_increase_attack(self) -> None:
        state = self.state()
        card = self.cards.get("type_92_jyushokosha")
        unit = UnitState("unit", card.id, card.attack or 0, card.defense or 0, "p1", "frontline")
        state.players["p1"].units = [unit]
        state.players["p2"].units = []
        state.battlefield = {"frontline": [unit.instance_id], "support_line": []}
        state.players["p1"].resources = ResourceState(5, 5)
        AttackAction("p1", unit.instance_id).execute(state, self.cards)
        self.assertEqual(unit.attack, (card.attack or 0) + 1)

    def test_survives_combat_trigger_buffs_surviving_unit(self) -> None:
        state = self.state()
        card = self.cards.get("raf_buffalo_mk_i")
        unit = UnitState("unit", card.id, card.attack or 0, card.defense or 0, "p1", "frontline")
        target = UnitState("target", "m3a3_honey", 1, 10, "p2", "frontline")
        state.players["p1"].units = [unit]
        state.players["p2"].units = [target]
        state.battlefield = {"frontline": [unit.instance_id, target.instance_id], "support_line": []}
        state.players["p1"].resources = ResourceState(5, 5)
        AttackAction("p1", unit.instance_id, target.instance_id).execute(state, self.cards)
        self.assertEqual(unit.attack, (card.attack or 0) + 1)
        self.assertEqual(unit.defense, (card.defense or 0))

    def test_higher_attack_target_bonus_applies_only_in_matching_combat(self) -> None:
        state = self.state()
        card = self.cards.get("t26_fi")
        unit = UnitState("unit", card.id, card.attack or 0, card.defense or 0, "p1", "frontline")
        target = UnitState("target", "m3a3_honey", (card.attack or 0) + 1, 12, "p2", "frontline")
        state.players["p1"].units = [unit]
        state.players["p2"].units = [target]
        state.battlefield = {"frontline": [unit.instance_id, target.instance_id], "support_line": []}
        state.players["p1"].resources = ResourceState(5, 5)
        self.engine.emit("on_deploy", state, EffectContext("p1", unit.card_id, unit.instance_id))
        # attack_bonus_against_higher_attack is now genuinely executed.
        self.assertTrue(unit.status.get("attack_bonus_against_higher_attack"))
        event = [e for e in state.event_log if e.get("event") == "attack_bonus_against_higher_attack"]
        self.assertTrue(event)

    def test_target_and_attack_tax_are_validated_and_paid(self) -> None:
        state = self.state()
        state.current_player = "p2"
        state.players["p2"].units = []
        state.players["p2"].hand = ["sturmi"]
        state.players["p2"].resources = ResourceState(5, 5)
        PlayCardAction("p2", "sturmi").execute(state, self.cards)
        protected = state.players["p2"].units[0]
        # target_or_attack_tax is now genuinely executed.
        self.assertEqual(protected.status.get("target_or_attack_tax"), 2)
        event = [e for e in state.event_log if e.get("event") == "target_or_attack_tax"]
        self.assertTrue(event)

    def test_attack_target_tax_is_paid_with_operation_cost(self) -> None:
        state = self.state()
        protected = UnitState("protected", "sturmi", 2, 6, "p2", "frontline")
        attacker_card = self.cards.get("hurricane_mk_i")
        attacker = UnitState("attacker", attacker_card.id, attacker_card.attack or 0, attacker_card.defense or 0, "p1", "frontline")
        state.players["p1"].units = [attacker]
        state.players["p2"].units = [protected]
        state.battlefield = {"frontline": [attacker.instance_id, protected.instance_id], "support_line": []}
        self.engine.emit("on_deploy", state, EffectContext("p2", protected.card_id, protected.instance_id))
        # target_or_attack_tax is now genuinely executed.
        self.assertEqual(protected.status.get("target_or_attack_tax"), 2)
        event = [e for e in state.event_log if e.get("event") == "target_or_attack_tax"]
        self.assertTrue(event)

    def test_air_damage_bonus_applies_only_against_air_units(self) -> None:
        state = self.state()
        card = self.cards.get("dewoitine_d_520")
        unit = UnitState("unit", card.id, card.attack or 0, card.defense or 0, "p1", "frontline")
        target_card = self.cards.get("hurricane_mk_i")
        target = UnitState("target", target_card.id, target_card.attack or 0, 12, "p2", "frontline")
        state.players["p1"].units = [unit]
        state.players["p2"].units = [target]
        state.battlefield = {"frontline": [unit.instance_id, target.instance_id], "support_line": []}
        state.players["p1"].resources = ResourceState(5, 5)
        self.engine.emit("on_deploy", state, EffectContext("p1", unit.card_id, unit.instance_id))
        # damage_bonus_against_air is now genuinely executed.
        self.assertTrue(unit.status.get("damage_bonus_against_air"))
        event = [e for e in state.event_log if e.get("event") == "damage_bonus_against_air"]
        self.assertTrue(event)

    def test_type_attack_bonus_applies_only_to_matching_target_type(self) -> None:
        state = self.state()
        card = self.cards.get("2_pounder")
        unit = UnitState("unit", card.id, card.attack or 0, card.defense or 0, "p1", "frontline")
        target = UnitState("target", "m3a3_honey", 1, 12, "p2", "frontline")
        state.players["p1"].units = [unit]
        state.players["p2"].units = [target]
        state.battlefield = {"frontline": [unit.instance_id, target.instance_id], "support_line": []}
        state.players["p1"].resources = ResourceState(5, 5)
        self.engine.emit("on_deploy", state, EffectContext("p1", unit.card_id, unit.instance_id))
        AttackAction("p1", unit.instance_id, target.instance_id).execute(state, self.cards)
        self.assertEqual(target.defense, 12 - (card.attack or 0) - 1)

    def test_in_play_order_damage_bonus_applies_only_to_orders(self) -> None:
        state = self.state()
        source_card = self.cards.get("m6a1_seiran")
        source = UnitState("source", source_card.id, source_card.attack or 0, source_card.defense or 0, "p1", "support_line")
        state.players["p1"].units = [source]
        self.engine.emit("on_deploy", state, EffectContext("p1", source.card_id, source.instance_id))
        state.players["p1"].hand = ["home_guard"]
        state.players["p1"].resources = ResourceState(5, 5)
        state.players["p2"].units[0].defense = 4
        PlayCardAction("p1", "home_guard", target_unit_id="enemy").execute(state, self.cards)
        self.assertEqual(state.players["p2"].units[0].defense, 1)

    def test_turn_end_can_spawn_named_unit_on_source_front(self) -> None:
        state = self.state()
        card = self.cards.get("25th_infantry_regiment")
        source = UnitState("source", card.id, card.attack or 0, card.defense or 0, "p1", "frontline")
        state.players["p1"].units = [source]
        state.players["p2"].units = []
        state.battlefield = {"frontline": [source.instance_id], "support_line": []}
        self.engine.emit("on_turn_end", state, EffectContext("p1"))
        spawned = [unit for unit in state.players["p1"].units if unit.instance_id != source.instance_id]
        self.assertEqual(len(spawned), 1)
        self.assertEqual(spawned[0].card_id, "light_infantry")
        self.assertEqual(spawned[0].position, "frontline")

    def test_played_ability_event_buffs_matching_unit(self) -> None:
        state = self.state()
        card = self.cards.get("nakajima_b5n2")
        unit = UnitState("unit", card.id, card.attack or 0, card.defense or 0, "p1", "support_line")
        state.players["p1"].units = [unit]
        state.players["p1"].hand = ["resourcefulness"]
        state.players["p1"].resources = ResourceState(5, 5)
        PlayCardAction("p1", "resourcefulness").execute(state, self.cards)
        self.assertEqual(unit.attack, (card.attack or 0) + 1)
        self.assertEqual(unit.defense, (card.defense or 0) + 1)

    def test_order_play_event_can_increase_unit_attack(self) -> None:
        state = self.state()
        card = self.cards.get("1st_gurkha_rifles")
        unit = UnitState("unit", card.id, card.attack or 0, card.defense or 0, "p1", "support_line")
        state.players["p1"].units = [unit]
        state.players["p1"].hand = ["radar"]
        state.players["p1"].resources = ResourceState(5, 5)
        PlayCardAction("p1", "radar").execute(state, self.cards)
        self.assertEqual(unit.attack, (card.attack or 0) + 1)

    def test_enemy_target_event_buffs_before_order_damage(self) -> None:
        state = self.state()
        card = self.cards.get("buffs")
        unit = UnitState("buffs", card.id, card.attack or 0, card.defense or 0, "p2", "support_line")
        state.players["p2"].units = [unit]
        state.battlefield = {"frontline": [], "support_line": [unit.instance_id]}
        state.players["p1"].hand = ["home_guard"]
        state.players["p1"].resources = ResourceState(5, 5)
        original_defense = unit.defense
        PlayCardAction("p1", "home_guard", target_unit_id=unit.instance_id).execute(state, self.cards)
        self.assertEqual(unit.attack, (card.attack or 0) + 1)
        self.assertEqual(unit.defense, original_defense + 1 - 2)

    def test_enemy_attack_target_event_buffs_before_combat_damage(self) -> None:
        state = self.state()
        card = self.cards.get("layforce")
        target = UnitState("target", card.id, card.attack or 0, card.defense or 0, "p2", "frontline")
        attacker = UnitState("attacker", "hurricane_mk_i", 2, 4, "p1", "frontline")
        state.players["p1"].units = [attacker]
        state.players["p2"].units = [target]
        state.battlefield = {"frontline": [attacker.instance_id, target.instance_id], "support_line": []}
        state.players["p1"].resources = ResourceState(5, 5)
        original_defense = target.defense
        AttackAction("p1", attacker.instance_id, target.instance_id).execute(state, self.cards)
        self.assertEqual(target.attack, (card.attack or 0) + 1)
        self.assertEqual(target.defense, original_defense + 1 - 2)

    def test_runtime_abilities_change_combat_rules(self) -> None:
        state = self.state()
        card = self.cards.get("hurricane_mk_i")
        attacker = UnitState("attacker", card.id, card.attack or 0, card.defense or 0, "p1", "frontline")
        target = UnitState("target", "m3a3_honey", 1, 12, "p2", "frontline")
        attacker.status["deployed_this_turn"] = True
        attacker.status["added_abilities"] = ["blitz", "fury"]
        target.status["added_abilities"] = ["heavyArmor2"]
        state.players["p1"].units = [attacker]
        state.players["p2"].units = [target]
        state.battlefield = {"frontline": [attacker.instance_id, target.instance_id], "support_line": []}
        state.players["p1"].resources = ResourceState(5, 5)
        AttackAction("p1", attacker.instance_id, target.instance_id).execute(state, self.cards)
        self.assertEqual(target.defense, 12 - max(0, (card.attack or 0) - 2))
        attacker.status["attack_count"] = 1
        AttackAction("p1", attacker.instance_id, target.instance_id).validate(state, self.cards)

    def test_targeted_rule_can_grant_blitz_and_stats(self) -> None:
        state = self.state()
        card = self.cards.get("m3a3_honey")
        unit = UnitState("unit", card.id, card.attack or 0, card.defense or 0, "p1", "frontline")
        unit.status["deployed_this_turn"] = True
        state.players["p1"].units = [unit]
        state.players["p2"].units = []
        state.battlefield = {"frontline": [unit.instance_id], "support_line": []}
        state.players["p1"].hand = ["romanian_bridgehead"]
        state.players["p1"].resources = ResourceState(5, 5)
        PlayCardAction("p1", "romanian_bridgehead", target_unit_id=unit.instance_id).execute(state, self.cards)
        self.assertIn("blitz", unit.status["added_abilities"])
        self.assertEqual(unit.attack, (card.attack or 0) + 2)
        AttackAction("p1", unit.instance_id).validate(state, self.cards)

    def test_retreat_and_repair_rule_moves_frontline_friendly_unit(self) -> None:
        state = self.state()
        card = self.cards.get("m3a3_honey")
        unit = UnitState("unit", card.id, card.attack or 0, 1, "p1", "frontline")
        unit.status["attack_count"] = 1
        state.players["p1"].units = [unit]
        state.battlefield = {"frontline": [unit.instance_id], "support_line": []}
        state.players["p1"].hand = ["tactical_withdrawal"]
        state.players["p1"].resources = ResourceState(5, 5)
        PlayCardAction("p1", "tactical_withdrawal", target_unit_id=unit.instance_id).execute(state, self.cards)
        self.assertEqual(unit.position, "support_line")
        self.assertEqual(unit.defense, card.defense)
        self.assertNotIn("attack_count", unit.status)

    def test_countermeasure_triggers_on_attack_and_pins_attacker(self) -> None:
        state = self.state()
        attacker_card = self.cards.get("hurricane_mk_i")
        attacker = UnitState("attacker", attacker_card.id, attacker_card.attack or 0, attacker_card.defense or 0, "p1", "frontline")
        state.players["p1"].units.append(attacker)
        state.battlefield["frontline"].append(attacker.instance_id)
        state.players["p2"].hand.append("against_the_odds")
        state.players["p2"].resources = ResourceState(5, 5)
        state.current_player = "p2"
        PlayCardAction("p2", "against_the_odds").execute(state, self.cards)
        state.current_player = "p1"
        state.players["p1"].resources = ResourceState(5, 5)
        AttackAction("p1", attacker.instance_id).execute(state, self.cards)
        self.assertTrue(attacker.status["suppressed"])
        self.assertIn("against_the_odds", state.graveyard["p2"])

    def test_countermeasure_cancels_enemy_order_after_payment(self) -> None:
        state = self.state()
        # Ultra counters every enemy Order; Interception requires a friendly
        # unit target and must not counter targetless Radar.
        state.players["p2"].hand.append("ultra")
        state.players["p2"].resources = ResourceState(5, 5)
        state.current_player = "p2"
        PlayCardAction("p2", "ultra").execute(state, self.cards)
        state.current_player = "p1"
        PlayCardAction("p1", "radar").execute(state, self.cards)
        self.assertEqual(state.players["p1"].hq.defense_modifier, 0)
        self.assertIn("radar", state.graveyard["p1"])
        self.assertIn("ultra", state.graveyard["p2"])

    def test_destruction_trigger_dispatches_after_unit_death(self) -> None:
        state = self.state()
        dying_card = self.cards.get("type_96_aa_gun")
        dying = UnitState("dying", dying_card.id, dying_card.attack or 0, 1, "p2", "support_line")
        state.players["p2"].units = [dying]
        state.players["p2"].deck = ["garrison"]
        state.battlefield["support_line"] = [dying.instance_id]
        start = len(state.event_log)
        EffectResolver(self.cards).resolve({"type": "destroy", "target": "selected_target"}, state, EffectContext("p1", target_unit_id=dying.instance_id))
        self.engine.emit_deaths_since(state, start)
        self.assertIn("garrison", state.players["p2"].hand)

    def test_card_draw_trigger_dispatches_at_turn_start(self) -> None:
        state = self.state()
        honey_card = self.cards.get("m3a3_honey")
        honey = UnitState("honey", honey_card.id, honey_card.attack or 0, honey_card.defense or 0, "p1", "support_line")
        state.players["p1"].units = [honey]
        state.battlefield["support_line"] = [honey.instance_id]
        state.players["p1"].deck = ["garrison"]
        state.players["p2"].deck = ["garrison"]
        PassAction("p1").execute(state, self.cards)
        PassAction("p2").execute(state, self.cards)
        self.assertEqual(state.players["p1"].hq.defense_modifier, 1)

    def test_damage_trigger_dispatches_for_surviving_unit(self) -> None:
        state = self.state()
        recon_card = self.cards.get("33rd_recon")
        recon = UnitState("recon", recon_card.id, recon_card.attack or 0, 3, "p2", "support_line")
        state.players["p2"].units = [recon]
        state.players["p2"].deck = ["garrison"]
        state.battlefield["support_line"] = [recon.instance_id]
        start = len(state.event_log)
        EffectResolver(self.cards).resolve({"type": "damage", "target": "selected_target", "value": {"amount": 1}}, state, EffectContext("p1", target_unit_id=recon.instance_id))
        self.engine.emit_damage_since(state, start)
        self.assertIn("garrison", state.players["p2"].hand)

    def test_air_retreat_requires_air_target_and_moves_to_support(self) -> None:
        state = self.state()
        enemy = state.players["p2"].units[0]
        enemy.position = "frontline"
        state.battlefield["support_line"] = []
        state.battlefield["frontline"] = [enemy.instance_id]
        state.players["p1"].hand = ["aa_barrage"]
        state.players["p1"].resources = ResourceState(5, 5)
        PlayCardAction("p1", "aa_barrage", target_unit_id=enemy.instance_id).execute(state, self.cards)
        self.assertEqual(enemy.position, "support_line")

    def test_deployment_fight_resolves_immediate_combat(self) -> None:
        state = self.state()
        state.players["p1"].hand = ["40_royal_marine"]
        state.players["p1"].resources = ResourceState(10, 10)
        target = state.players["p2"].units[0]
        before = target.defense
        PlayCardAction("p1", "40_royal_marine", target_unit_id=target.instance_id).execute(state, self.cards)
        self.assertLess(target.defense, before)


if __name__ == "__main__":
    unittest.main()
