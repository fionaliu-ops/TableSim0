"""Behavioral checks for adapted algorithms, evidence boundaries and execution."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from baseline import InnerMonologuePlanner, SayPlanPlanner, LERaPlanner, PaperObservation
from baseline.paper_common import parse_plan
from baseline.paper_runner import run_paper_episode, observed_action_success
from baseline.sayplan import TabletopHierarchy, TabletopSceneSimulator
from high_level_planner.reasoning_graph.language_models.base import LanguageModelError
from high_level_planner.reasoning_graph.state import AtomicAction
from high_level_planner.reasoning_graph.transition import SceneGraphTransitionModel
from high_level_planner.scene_graph.graph import SceneGraph, SceneObject, Relation
from high_level_planner.scene_graph.attributes import ObjectAttributes


def scene():
    return SceneGraph(objects=[SceneObject(key, 'book', attributes=ObjectAttributes(
        xy=xy, footprint=[.06, .05], stackable=True, yaw_degrees=0.,
        extra={'geometry_frame': 'image_normalized'})) for key, xy in
        [('a', [.2, .3]), ('b', [.8, .3])]])


def action(source='a', xy=None):
    return {'action_type': 'move', 'source_object_id': source, 'target': 'table',
        'parameters': {'pose': {'frame': 'image_normalized', 'xy': xy or [.4, .7], 'yaw_degrees': 0.}}}


def plan(*actions, status=None):
    return json.dumps({'status': status or ('plan' if actions else 'done'), 'actions': list(actions)})


class ScriptedModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs.get('images')))
        if not self.replies: raise RuntimeError('Unexpected model call')
        value = self.replies.pop(0)
        if isinstance(value, Exception): raise value
        return [value]


class PaperBaselinesTest(unittest.TestCase):
    def setUp(self):
        self.scene = scene()
        self.rgb = np.full((24, 24, 3), 127, np.uint8)
        self.obs = PaperObservation(self.rgb, self.scene)
        self.catalog = {'a': 'book', 'b': 'book'}

    def planner(self, cls, replies, **kwargs):
        return cls(model=ScriptedModel(replies), object_catalog=self.catalog, **kwargs)

    def test_im_text_only_history_contains_scene_and_binary_success(self):
        planner = self.planner(InnerMonologuePlanner, [plan(action()), plan(action('b'))])
        first = planner.start(self.obs, 'tidy')
        planner.record_execution_result(first.next_subtask, success=False,
            details={'hidden_world_pose': [999, 998, 997], 'failure_category': 'grasp_occluded'})
        second = planner.step(self.obs)
        prompt, images = planner.model.calls[-1]
        self.assertIsNone(images)
        self.assertIn('"Success": false', prompt)
        self.assertIn('PreviousAction', prompt)
        self.assertNotIn('hidden_world_pose', prompt)
        self.assertNotIn('grasp_occluded', prompt)
        self.assertEqual(second.next_subtask.source_object_id, 'b')

    def test_im_rejects_sequence_and_requires_feedback(self):
        planner = self.planner(InnerMonologuePlanner, [plan(action(), action('b'))])
        with self.assertRaises(LanguageModelError): planner.start(self.obs, 'tidy')
        planner = self.planner(InnerMonologuePlanner, [plan(action())])
        planner.start(self.obs, 'tidy')
        with self.assertRaises(RuntimeError): planner.step(self.obs)

    def test_lera_success_consumes_queue_without_model_calls(self):
        planner = self.planner(LERaPlanner, [])
        first = planner.start(self.obs, 'tidy', initial_plan=[action(), action('b')])
        planner.record_execution_result(first.next_subtask, success=True)
        second = planner.step(PaperObservation(None, None))
        self.assertEqual(second.next_subtask.source_object_id, 'b')
        self.assertEqual(planner.model_calls, 0)

    def test_lera_three_stages_only_look_gets_rgb_and_no_controller_reason(self):
        planner = self.planner(LERaPlanner, ['A book blocks access.', 'Move the blocker first.', plan(action('b'), action())])
        first = planner.start(self.obs, 'tidy', initial_plan=[action()])
        planner.record_execution_result(first.next_subtask, success=False,
            details={'blocking_object_ids': ['SECRET_CONTROLLER_ID']})
        output = planner.step(self.obs)
        self.assertEqual([x['stage'] for x in planner.audit], ['look', 'explain', 'replan'])
        self.assertEqual([bool(images) for _, images in planner.model.calls], [True, False, False])
        self.assertNotIn('SECRET_CONTROLLER_ID', json.dumps(planner.audit))
        self.assertEqual(output.next_subtask.source_object_id, 'b')
        self.assertEqual(planner.replans, 1)

    def test_lera_initial_planner_is_separate_from_replanner(self):
        planner = self.planner(LERaPlanner, [plan(action())])
        output = planner.start(self.obs, 'tidy')
        self.assertEqual(planner.audit[0]['stage'], 'initial_plan')
        self.assertEqual(output.status, 'action')

    def test_invalid_plan_is_rejected_atomically(self):
        planner = self.planner(LERaPlanner, [plan(action(), action('invented'))])
        with self.assertRaises(LanguageModelError): planner.start(self.obs, 'tidy')
        self.assertEqual(planner.plan, [])
        self.assertIsNone(planner.pending)

    def test_shared_executor_decodes_all_planners_without_graph_pipeline(self):
        from script.dynamic_sim import request_action
        for cls in [InnerMonologuePlanner, LERaPlanner]:
            planner = self.planner(cls, [plan(action())])
            output = planner.start(self.obs, 'tidy')
            decoded = request_action(output.vla_request)
            self.assertEqual((decoded['kind'], decoded['source']), ('move', 'a'))
            self.assertEqual(decoded['parameters']['pose']['frame'], 'image_normalized')
            self.assertIsNone(output.to_dict()['reasoning_graph'])

    def test_model_completion_is_not_external_success(self):
        planner = self.planner(InnerMonologuePlanner, [plan()])
        result = run_paper_episode(planner, observe=lambda: self.obs,
            execute=lambda request: self.fail('No action expected'),
            evaluate=lambda observation: {'success': False}, instruction='tidy')
        self.assertEqual(result['status'], 'false_completion')
        self.assertFalse(result['external_evaluation']['success'])

    def test_runner_recovery_and_final_step_goal_evaluation(self):
        current = deepcopy(self.scene)
        attempts = []
        planner = self.planner(LERaPlanner, ['Book stayed still.', 'Retry.', plan(action())])
        def observe(): return PaperObservation(self.rgb, deepcopy(current))
        def execute(request):
            nonlocal current
            attempts.append(request)
            if len(attempts) == 1: return {'success': False}
            edge = request.metadata['reasoning_edge']
            current = SceneGraphTransitionModel().apply(current,
                AtomicAction(edge['action_type'], request.object_target['source'], request.object_target['target'], parameters=edge['parameters']))
            return {'success': True}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'episode.json'
            result = run_paper_episode(planner, observe=observe, execute=execute,
                evaluate=lambda o: {'success': o.scene.get_object('a').metadata.get('planning_pose', {}).get('xy') == [.4, .7]},
                instruction='tidy', initial_plan=[action()], max_steps=2, output_path=path)
            self.assertEqual(result['status'], 'external_success')
            self.assertEqual(result['metrics']['execution_failures'], 1)
            self.assertEqual(len(json.loads(path.read_text())['steps']), 2)

    def test_fatal_stops_before_replanner(self):
        planner = self.planner(LERaPlanner, [])
        first = planner.start(self.obs, 'tidy', initial_plan=[action()])
        planner.record_execution_result(first.next_subtask, success=False, fatal=True)
        self.assertEqual(planner.step(self.obs).reason, 'fatal_execution_failure')
        self.assertEqual(planner.model_calls, 0)

    def test_model_budget_is_hard_and_calls_are_audited_on_failure(self):
        planner = self.planner(LERaPlanner, ['description', 'correction'], max_model_calls=2)
        first = planner.start(self.obs, 'tidy', initial_plan=[action()])
        planner.record_execution_result(first.next_subtask, success=False)
        with self.assertRaises(LanguageModelError): planner.step(self.obs)
        self.assertEqual(planner.model_calls, 2)
        planner = self.planner(InnerMonologuePlanner, [RuntimeError('API unavailable')])
        with self.assertRaises(RuntimeError): planner.start(self.obs, 'tidy')
        self.assertIn('API unavailable', planner.audit[0]['error'])

    def test_action_checker_uses_tolerance_and_rejects_unchanged_scene(self):
        planner = self.planner(InnerMonologuePlanner, [plan(action())])
        output = planner.start(self.obs, 'tidy')
        self.assertFalse(observed_action_success(self.obs, self.obs, output.next_subtask, {'success': True}))
        current = deepcopy(self.scene)
        current.get_object('a').attributes.xy = [.405, .705]
        self.assertTrue(observed_action_success(self.obs, PaperObservation(self.rgb, current), output.next_subtask, {'success': True}))
        current.get_object('a').metadata['visual_identity_confirmed'] = False
        self.assertFalse(observed_action_success(self.obs, PaperObservation(self.rgb, current), output.next_subtask, {'success': True}))

    def test_message_checks_unknown_ids_nan_and_conflicting_sources(self):
        for item in [action('unknown'), {**action(), 'parameters': {'pose': {'frame': 'world', 'xy': [0,0]}}},
                     {**action(), 'parameters': {'source_object_ids': ['a', 'b']}}]:
            with self.assertRaises(ValueError): parse_plan(plan(item), self.catalog, 32)
        for value in ('NaN', '1e999'):
            with self.assertRaises(ValueError): parse_plan('{"status":"done","actions":[],"extra":'+value+'}', self.catalog, 32)


class SayPlanTest(unittest.TestCase):
    def test_expand_contract_preserves_hierarchy_and_hides_details(self):
        observed = scene()
        observed.relations = [Relation('b', 'a', 'on')]
        hierarchy = TabletopHierarchy(observed)
        self.assertEqual(hierarchy.visible_ids(), {'__table__'})
        hierarchy.operate('expand', '__table__')
        self.assertEqual(hierarchy.visible_ids(), {'__table__', 'a'})
        self.assertNotIn('pose', next(x for x in hierarchy.payload()['nodes'] if x['id']=='a'))
        hierarchy.operate('expand', 'a')
        self.assertIn('b', hierarchy.visible_ids())
        hierarchy.operate('expand', 'b')
        hierarchy.operate('contract', 'a')
        self.assertNotIn('b', hierarchy.visible_ids())
        hierarchy.operate('expand', 'a')
        self.assertNotIn('b', hierarchy.expanded)

    def test_simulator_rejects_blocked_action_then_rolls_out_reordered_plan(self):
        observed = scene()
        observed.relations = [Relation('b','a','blocks')]
        observed.metadata['graph_validation_ignore_physical_feasibility'] = True
        snapshot = deepcopy(observed.to_dict())
        simulator = TabletopSceneSimulator()
        a = AtomicAction.from_dict(action('a'))
        b = AtomicAction.from_dict(action('b', [.75,.75]))
        bad = simulator.check(observed, [a])
        self.assertFalse(bad['valid'])
        self.assertEqual(bad['failed_index'], 0)
        good = simulator.check(observed, [b,a])
        self.assertTrue(good['valid'])
        self.assertFalse(good['physical_execution_verified'])
        self.assertEqual(observed.to_dict(), snapshot)

    def test_simulator_rejects_incomplete_goal_and_keeps_unknown_physics(self):
        observed = scene()
        goal = deepcopy(observed)
        goal.get_object('a').attributes.xy = [.4,.7]
        simulator = TabletopSceneSimulator()
        self.assertFalse(simulator.check(observed, [], goal)['valid'])
        report = simulator.check(observed, [AtomicAction.from_dict(action())], goal)
        self.assertTrue(report['valid'])
        self.assertIsNone(report['trace'][0]['feasibility']['checks']['trajectory_valid'])

    def test_sayplan_feedback_repairs_plan_before_any_execution(self):
        observed = scene()
        observed.relations = [Relation('b','a','blocks')]
        replies = [json.dumps({'operation':'expand','node_id':'__table__'}),
                   json.dumps({'operation':'done'}), plan(action()), plan(action('b',[.75,.75]),action())]
        model = ScriptedModel(replies)
        planner = SayPlanPlanner(model=model, object_catalog={'a':'book','b':'book'})
        output = planner.start(PaperObservation(scene=observed), 'tidy')
        self.assertEqual(output.next_subtask.source_object_id, 'b')
        self.assertEqual(planner.execution_count, 0)
        self.assertEqual(planner.replans, 1)
        self.assertIn('source_not_accessible', model.calls[-1][0])
        self.assertTrue(all(images is None for _,images in model.calls))

    def test_sayplan_search_and_repair_limits_stop_without_fallback(self):
        model = ScriptedModel([json.dumps({'operation':'expand','node_id':'__table__'})])
        planner = SayPlanPlanner(model=model, object_catalog={'a':'book','b':'book'}, max_search_rounds=1)
        with self.assertRaises(RuntimeError): planner.start(PaperObservation(scene=scene()), 'tidy')
        observed = scene()
        observed.relations = [Relation('b','a','blocks')]
        model = ScriptedModel([json.dumps({'operation':'done'}),plan(action())])
        planner = SayPlanPlanner(model=model, object_catalog={'a':'book','b':'book'}, max_plan_repairs=1)
        self.assertEqual(planner.start(PaperObservation(scene=observed),'tidy').status,'blocked')


if __name__ == '__main__':
    unittest.main()
