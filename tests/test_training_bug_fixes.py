"""Behavioral regressions for BUG-03 through BUG-09."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from simulator.actions.action import MulliganAction, PassAction, PlayCardAction, _operation_cost
from simulator.cards.loader import CardDatabase
from simulator.core.game import Simulator, UnsupportedEffectError
from simulator.core.state import GameState, PlayerState, ResourceState, UnitState, GameStatus
from simulator.effects.resolver import EffectContext, EffectResolver
from simulator.testing import build_random_deck


class TrainingBugFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cards = CardDatabase.from_file(Path(__file__).parents[1] / 'data/source/kards_info_cards.json')

    def fixture(self):
        state = GameState('p1', {
            'p1': PlayerState('p1', 'Britain', hand=['dilemma'] * 2, resources=ResourceState(10, 10)),
            'p2': PlayerState('p2', 'Germany', deck=['1_infantry_regiment']),
        }, graveyard={'p1': [], 'p2': []}, rng_seed=1)
        return Simulator(self.cards, state=state, record_replay=False)

    def test_manual_and_automatic_mulligan_have_identical_first_turn(self):
        decks = [build_random_deck(self.cards, seed=i, main_nation=n)
                 for i, n in enumerate(('Britain', 'Germany'))]
        auto, manual = Simulator(self.cards), Simulator(self.cards)
        auto.reset(*decks, nations=('Britain', 'Germany'), seed=42)
        manual.reset(*decks, nations=('Britain', 'Germany'), seed=42, auto_mulligan=False)
        manual.step_fast(MulliganAction('p1'))
        self.assertEqual(manual.state.players['p1'].resources.kredits, 0)
        manual.step_fast(MulliganAction('p2'))
        self.assertEqual(auto.state.to_dict(), manual.state.to_dict())
        self.assertEqual(manual.state.current_player, 'p1')
        self.assertEqual(manual.state.players['p1'].resources.kredits, 1)
        self.assertEqual([len(p.hand) for p in manual.state.players.values()], [4, 5])
        self.assertEqual(sum(e['event'] == 'turn_started' for e in manual.state.event_log), 1)

    def test_terminal_reward_is_from_acting_player_perspective(self):
        for status, expected in ((GameStatus.PLAYER_ONE_WON, 100),
                                 (GameStatus.PLAYER_TWO_WON, -100), (GameStatus.DRAW, 0)):
            with self.subTest(status=status):
                env = self.fixture()
                def finish(action, state, cards):
                    state.game_status = status
                with patch.object(PassAction, 'execute', finish):
                    self.assertEqual(env.step_fast(PassAction('p1')), (expected, True))

    def test_high_attack_does_not_remove_other_cards_options(self):
        env = self.fixture()
        env.state.players['p1'].units = [UnitState('own', '1_infantry_regiment', 4, 2, 'p1')]
        spec = SimpleNamespace(needs_target=False, swap_hand_unit=False, convert_to=False,
                               shock_tactics=False, develop_options=('choice-a', 'choice-b'))
        with patch('simulator.core.game.engine_for', return_value=SimpleNamespace(action_spec_for=lambda cid: spec)), \
             patch('simulator.core.game._is_valid', return_value=True):
            options = {a.selected_option for a in env.get_available_actions() if isinstance(a, PlayCardAction)}
        self.assertEqual(options, {None, 'choice-a', 'choice-b'})

    def test_observation_includes_own_effects_and_hides_enemy_secrets(self):
        env = self.fixture()
        own = env.state.players['p1']
        own.active_countermeasures = [{'card_id': 'entanglement', 'activated_turn': 1}]
        own.units = [UnitState('own', '1_infantry_regiment', 4, 2, 'p1',
                               modifiers=[{'type': 'set_operation_cost', 'value': 3}],
                               status={'keywords': ['guard'], 'pinned': True})]
        enemy = env.state.players['p2']
        enemy.active_countermeasures = [{'card_id': 'secret'}]
        obs = env.get_observation('p1')
        self.assertEqual(obs['self']['active_countermeasures'], own.active_countermeasures)
        self.assertEqual(obs['self']['units'][0]['operation_cost'], 3)
        self.assertEqual(obs['self']['units'][0]['status']['keywords'], ['guard'])
        self.assertIsNone(obs['opponent']['active_countermeasures'])
        obs['self']['units'][0]['status']['keywords'].append('blitz')
        self.assertEqual(own.units[0].status['keywords'], ['guard'])

    def test_covert_ids_are_opaque_and_player_actions_map_back(self):
        env = self.fixture()
        target = UnitState('1_infantry_regiment-secret', '1_infantry_regiment', 4, 2, 'p2', status={'covert': True})
        env.state.players['p2'].units = [target]
        env.state.battlefield['support_line'] = [target.instance_id]
        obs = env.get_observation('p1')
        self.assertNotIn('1_infantry_regiment', str(obs))
        self.assertIsNone(obs['opponent']['units'][0]['card_id'])
        # Reveal for a legal targeted order, retaining its opaque identity.
        target.status.pop('covert')
        player_actions = env.get_player_actions()
        self.assertNotIn('1_infantry_regiment-secret', str(player_actions))
        action = next(a for a in player_actions if isinstance(a, PlayCardAction))
        env.step_player(action)
        self.assertEqual((target.attack, _operation_cost(target, self.cards.get(target.card_id))), (0, 4))

    def test_unsupported_effect_rejects_episode_with_reproduction_data(self):
        env = self.fixture()
        def unsupported(action, state, cards):
            state.event_log.append({'event': 'unsupported_effect', 'type': 'missing'})
        with patch.object(PassAction, 'execute', unsupported):
            with self.assertRaises(UnsupportedEffectError) as caught:
                env.step_fast(PassAction('p1'))
        self.assertEqual(caught.exception.report['seed'], 1)
        self.assertEqual(caught.exception.report['actions'][0]['type'], 'PassAction')
        with self.assertRaises(RuntimeError):
            env.step_fast(PassAction('p1'))
        env = self.fixture()
        env.strict_effects = False
        with patch.object(PassAction, 'execute', unsupported):
            env.step_fast(PassAction('p1'))

    def test_resolver_unit_type_condition_can_access_database(self):
        env = self.fixture()
        env.state.players['p1'].units = [UnitState('u', '1_infantry_regiment', 4, 2, 'p1')]
        resolver = EffectResolver(self.cards)
        self.assertTrue(resolver._conditions_met({'type': 'has_unit_type', 'unit_type': 'infantry'}, env.state, EffectContext('p1')))

    def test_attack_changes_after_swap_are_used_by_next_swap(self):
        for delta in (-1, 1):
            with self.subTest(delta=delta):
                env = self.fixture()
                u = UnitState('u', 'hurricane_mk_i', 4, 2, 'p2')
                env.state.players['p2'].units = [u]
                env.state.battlefield['support_line'] = ['u']
                a = PlayCardAction('p1', 'dilemma', target_unit_id='u')
                env.step_fast(a)
                EffectResolver(self.cards).resolve({'type': 'modify_attack', 'target': 'selected_target', 'value': {'amount': delta}}, env.state, EffectContext('p1', target_unit_id='u'))
                base_cost = self.cards.get(u.card_id).operationCost
                self.assertEqual(u.attack, base_cost + delta)
                env.step_fast(a)
                self.assertEqual(u.attack, 4)
                self.assertEqual(_operation_cost(u, self.cards.get(u.card_id)), base_cost + delta)

    def test_cost_increase_after_swap_affects_current_cost_and_next_swap(self):
        self.check_cost_change(1)

    def test_cost_decrease_after_swap_affects_current_cost_and_next_swap(self):
        self.check_cost_change(-1)

    def check_cost_change(self, delta):
        env = self.fixture()
        u = UnitState('u', '1_infantry_regiment', 4, 2, 'p2')
        env.state.players['p2'].units = [u]
        env.state.battlefield['support_line'] = ['u']
        a = PlayCardAction('p1', 'dilemma', target_unit_id='u')
        env.step_fast(a)
        EffectResolver(self.cards).resolve({'type': 'modify_operation_cost', 'target': 'selected_target', 'value': {'amount': delta}}, env.state, EffectContext('p1', target_unit_id='u'))
        current_cost = 4 + delta
        self.assertEqual(_operation_cost(u, self.cards.get(u.card_id)), current_cost)
        env.step_fast(a)
        self.assertEqual(u.attack, current_cost)
        # Existing deltas are not applied a second time to the swapped-in cost.
        self.assertEqual(_operation_cost(u, self.cards.get(u.card_id)), 0)
        EffectResolver(self.cards).resolve({'type': 'modify_operation_cost', 'target': 'selected_target', 'value': {'amount': 1}}, env.state, EffectContext('p1', target_unit_id='u'))
        self.assertEqual(_operation_cost(u, self.cards.get(u.card_id)), 1)

    def test_cost_modifiers_before_swap_survive_two_swaps(self):
        for delta in (-1, 1):
            with self.subTest(delta=delta):
                env = self.fixture()
                u = UnitState('u', 'hurricane_mk_i', 4, 2, 'p2', modifiers=[
                    {'type': 'modify_operation_cost', 'amount': delta, 'source_unit_id': 'aura'}])
                env.state.players['p2'].units = [u]
                env.state.battlefield['support_line'] = ['u']
                original_cost = _operation_cost(u, self.cards.get(u.card_id))
                a = PlayCardAction('p1', 'dilemma', target_unit_id='u')
                env.step_fast(a)
                self.assertEqual((u.attack, _operation_cost(u, self.cards.get(u.card_id))), (original_cost, 4))
                env.step_fast(a)
                self.assertEqual((u.attack, _operation_cost(u, self.cards.get(u.card_id))), (4, original_cost))
                self.assertTrue(any(m.get('source_unit_id') == 'aura' for m in u.modifiers))

    def test_latest_set_cost_keeps_independent_adjustments(self):
        u = UnitState('u', 'hurricane_mk_i', 4, 2, 'p2', modifiers=[
            {'type': 'set_operation_cost', 'value': 4},
            {'type': 'modify_operation_cost', 'amount': 1},
            {'type': 'set_operation_cost', 'value': 2},
        ])
        self.assertEqual(_operation_cost(u, self.cards.get(u.card_id)), 3)


if __name__ == '__main__':
    unittest.main()
