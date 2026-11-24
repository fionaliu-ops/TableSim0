import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from high_level_planner.goal_generator import GoalGenerator
from high_level_planner.scene_graph.graph import Relation
from high_level_planner.tests.test_unique_target_goal import scene
from script.adaptive_goal import AdaptiveGoalRevision


class AdaptiveGoalTest(unittest.TestCase):
    def test_pickup_failure_does_not_revise_terminal_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            target=scene();owner=AdaptiveGoalRevision(Mock(),target,target,'old.png',directory)
            for phase in ('park_other_arm','grasp_approach','grasp_contact'):
                self.assertFalse(owner.should_revise({'feedback':{
                    'failure_category':'motion_execution_failed','phase':phase}}))
            self.assertTrue(owner.should_revise({'feedback':{
                'failure_category':'motion_execution_failed','phase':'carry_transfer'}}))

    def test_rejected_images_do_not_consume_active_revision_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            target=scene();owner=AdaptiveGoalRevision(Mock(),target,target,'old.png',directory)
            record={'feedback':{'failure_category':'motion_execution_failed'}}
            owner.history=[{'status':'rejected'}]*3
            self.assertTrue(owner.should_revise(record))
            owner.history=[{'status':'activated'}]*3
            self.assertFalse(owner.should_revise(record))

    def test_failed_temporary_buffer_keeps_terminal_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            target=scene();owner=AdaptiveGoalRevision(Mock(),target,target,'old.png',directory)
            record={'request':{'object_target':{'source':'mouse_a'}},'feedback':{
                'failure_category':'motion_execution_failed','requested_pose':{
                    'frame':'image_normalized','xy':[.8,.2],'support':'table'}}}
            self.assertFalse(owner.should_revise(record))
            record['feedback']['requested_pose']['xy']=[.3,.7]
            self.assertTrue(owner.should_revise(record))

    def test_occluder_at_its_final_pose_is_a_terminal_layout_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            target=scene();owner=AdaptiveGoalRevision(Mock(),target,target,'old.png',directory)
            record={'request':{'object_target':{'source':'mouse_a'}},'feedback':{
                'failure_category':'grasp_occluded','blocking_object_ids':['keyboard_a'],
                'blocking_visual_poses':{'keyboard_a':{'frame':'image_normalized','xy':[.7,.7]}}}}
            self.assertTrue(owner.should_revise(record))
            record['feedback']['blocking_visual_poses']['keyboard_a']['xy']=[.6,.4]
            self.assertFalse(owner.should_revise(record))

    def test_unchanged_blocker_repairs_order_after_one_layout_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            owner=AdaptiveGoalRevision(Mock(),scene(),scene(),'old.png',directory)
            request={'object_target':{'source':'item'}}
            feedback={'failure_category':'destination_occupied','blocking_object_ids':['support'],
                      'blocking_visual_poses':{'support':{'frame':'image_normalized','xy':[.4,.4]}}}
            owner.history.append({'status':'activated','failed_request':request,'reason':feedback,
                                  'blocking_visual_poses':feedback['blocking_visual_poses']})
            self.assertFalse(owner.should_revise({'request':request,'feedback':feedback}))
            moved={**feedback,'blocking_visual_poses':{'support':{'frame':'image_normalized','xy':[.6,.4]}}}
            self.assertTrue(owner.should_revise({'request':request,'feedback':moved}))

    def test_revision_composes_rotation_with_current_observed_reference(self):
        current, target = scene(), scene()
        current.objects[0].attributes.yaw_degrees = 30.
        target.objects[0].attributes.yaw_degrees = -10.
        current.objects[1].metadata['visual_yaw_observable'] = False
        target.objects[1].attributes.yaw_degrees = 0.
        AdaptiveGoalRevision.retain_rotation_reference(current,target)
        self.assertEqual(target.objects[0].attributes.yaw_degrees,20.)
        self.assertEqual(target.objects[0].metadata['planning_pose']['yaw_degrees'],20.)
        self.assertIsNone(target.objects[1].attributes.yaw_degrees)
        self.assertNotIn('yaw_degrees',target.objects[1].metadata['planning_pose'])

    def test_planner_revision_swaps_graph_and_subscription_only_after_success(self):
        from high_level_planner.planner.planner import HighLevelPlanner
        from high_level_planner.evidence import EvidenceStore
        from high_level_planner.reasoning_graph.graph import ReasoningGraph
        from high_level_planner.reasoning_graph.state import TaskState
        from high_level_planner.memory.graph_memory import GraphMemory
        for fail in (False, True):
            with self.subTest(model_failed=fail):
                current = scene(); new_target = scene()
                old = ReasoningGraph(); old.add_node(TaskState('S0','old',scene_graph=current))
                new = ReasoningGraph(start_id='R0'); new.add_node(TaskState('R0','new',scene_graph=current))
                ledger = EvidenceStore()
                ledger.execution_events.append({'event_id':'EX1','success':False,'details':{}})
                memory = GraphMemory(scene_graph=current,reasoning_graph=old,current_node_id='S0')
                updater = Mock(); updater.subscribe_updates.return_value = 'new_subscription'
                original_generator = object(); generator = Mock()
                planner = NS(evidence_store=ledger, memory=memory, reasoning_generator=Mock(),
                    _target_position_tolerance=.02, _target_scene_graph=current,
                    _target_reasoning_generator=original_generator,_target_update_subscription='old_subscription',
                    _apply_failed_execution_evidence=Mock(), scene_updater=updater,
                    graph_planner=NS(plan=Mock(return_value=NS(next_subtask=None))), vla_interface=Mock())
                with patch('high_level_planner.reasoning_graph.target_conditioned.generate_between_scenes_with_generator',
                           side_effect=RuntimeError('failed') if fail else None,
                           return_value=(new,generator)) as build:
                    if fail:
                        with self.assertRaises(RuntimeError):
                            HighLevelPlanner.revise_terminal_target(planner,current,new_target,'organize',revision={'revision':1})
                        self.assertIs(memory.reasoning_graph,old)
                        self.assertIs(planner._target_reasoning_generator,original_generator)
                        updater.unsubscribe_updates.assert_not_called()
                    else:
                        HighLevelPlanner.revise_terminal_target(planner,current,new_target,'organize',revision={'revision':1})
                        self.assertIs(memory.reasoning_graph,new)
                        self.assertEqual(memory.current_node_id,'R0')
                        updater.unsubscribe_updates.assert_called_once_with('old_subscription')
                        self.assertIs(planner._target_reasoning_generator,generator)
                self.assertEqual(ledger.execution_events[0]['event_id'],'EX1')

    def test_supplied_initial_image_binding_cannot_disable_regeneration(self):
        generator = GoalGenerator()
        generator.generate = Mock(side_effect=AssertionError('initial image reuse must not run'))
        with patch('high_level_planner.goal_generator.image_first.generate_image_first') as call:
            generator.revise('current.png', current_scene_graph=scene(),
                             instruction='blocked', previous_goal_image='old.png')
        self.assertEqual(call.call_args.kwargs['reference_images'], ['old.png'])
        self.assertNotIn('layout_image', call.call_args.kwargs)

    def test_feedback_triggers_revision_but_occlusion_first_replans_and_held_never_revises(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = AdaptiveGoalRevision(Mock(), scene(), scene(), 'old.png', directory)
            request = {'feedback':{'failure_category':'destination_occupied'}}
            self.assertTrue(owner.should_revise(request))
            self.assertFalse(owner.should_revise(request, held=True))
            self.assertFalse(owner.should_revise({'feedback':{'failure_category':'grasp_occluded'}}))

    def test_revised_image_drives_new_graph_and_keeps_observed_node(self):
        with tempfile.TemporaryDirectory() as directory:
            current = scene(); current.image_path = 'current.png'
            target = scene(); revised = scene()
            revised.objects[0].metadata['planning_pose']['xy'] = [.4,.4]
            prediction = NS(scene_graph=revised, ready_for_planning=True,
                            image=NS(image_path=Path('new.png')), bundle_path='bundle.json')
            generator = NS(revise=Mock(return_value=prediction))
            output = NS(reasoning_graph=Mock(start_id='R0',metadata={}))
            planner = NS(revise_terminal_target=Mock(return_value=output))
            owner = AdaptiveGoalRevision(generator, current, target, 'old.png', directory)
            record = {'request':{'source':'mouse_a'}, 'node_localization':{'selected_node_id':'S4'},
                      'feedback':{'failure_category':'destination_occupied','blocking_object_ids':['keyboard_a'],
                                  'world_pose':[1,2,3]}}
            actual, _ = owner.revise(current, target, record, 'organize', planner)
            self.assertEqual(actual.objects[0].metadata['planning_pose'],revised.objects[0].metadata['planning_pose'])
            self.assertIs(planner.revise_terminal_target.call_args.args[0], current)
            prompt = generator.revise.call_args.kwargs['instruction']
            self.assertIn('S4', prompt)
            self.assertNotIn('world_pose', prompt)
            self.assertEqual(owner.history[0]['status'], 'activated')
            self.assertFalse(owner.history[0]['physical_success_claimed'])
            self.assertEqual(owner.image, Path('new.png'))

    def test_invalid_revision_keeps_original_active_goal(self):
        for defect in ('inventory', 'support', 'unchanged', 'planning_error'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                current = scene(); current.image_path = 'current.png'
                target = scene(); revised = scene()
                if defect != 'unchanged':
                    revised.objects[0].metadata['planning_pose']['xy'] = [.4,.4]
                if defect == 'inventory': revised.objects.pop()
                if defect == 'support': revised.add_relation(Relation('mouse_a','keyboard_a','on'))
                generator = NS(revise=Mock(return_value=NS(scene_graph=revised, ready_for_planning=True,
                    image=NS(image_path=Path('new.png')),bundle_path='bundle.json')))
                planner = NS(revise_terminal_target=Mock(side_effect=RuntimeError('model unavailable')))
                owner = AdaptiveGoalRevision(generator,current,target,'old.png',directory)
                with self.assertRaises((ValueError, RuntimeError)):
                    owner.revise(current,target,{'request':{},'feedback':{}},'organize',planner)
                self.assertEqual(owner.image,Path('old.png'))
                self.assertEqual(owner.history[0]['status'],'rejected')
                if defect != 'planning_error': planner.revise_terminal_target.assert_not_called()


if __name__ == '__main__':
    unittest.main()
