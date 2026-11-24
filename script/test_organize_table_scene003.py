"""Regression checks for action grounding and observed-state task scoring."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import sapien
from envs.organize_table_scene003 import (
    DEFAULT_PLAN, EXPECTED_CHAIN, load_action_chain, organize_table_scene003, without_large_book,
)
from envs.scene_003 import scene_003


class PlanTests(unittest.TestCase):
    def test_recorded_plan(self):
        steps, digest = load_action_chain()
        self.assertEqual(len(steps), 9)
        self.assertEqual(len(digest), 64)
        self.assertEqual(tuple((s['action_type'], s['source_object_id'], s['target']) for s in steps), EXPECTED_CHAIN)

    def test_large_book_removal_updates_actions_and_support_without_mutating_source(self):
        original,_=load_action_chain()
        saved=deepcopy(original)
        adapted=without_large_book(original)
        self.assertEqual(original,saved)
        self.assertEqual([step['step'] for step in adapted],[1,2,5,6,7,8,9])
        self.assertFalse(any(step['source_object_id']=='book_sam3_2' or step['target']=='book_sam3_2' for step in adapted))
        cup=next(step for step in adapted if step['step']==7)
        self.assertEqual(cup['target'],'book_sam3')
        pose=cup['parameters'].get('pose') or cup['parameters']['relative_pose']
        self.assertEqual(pose['support'],'book_sam3')

    def test_rejects_wrong_identity_frame_and_nonfinite_coordinates(self):
        original = json.loads(DEFAULT_PLAN.read_text())
        for mutation in ('identity', 'frame', 'nan', 'sequence', 'support'):
            data = deepcopy(original)
            first = data['steps'][0]
            if mutation == 'identity': first['source_object_id'] = 'book_sam3'
            if mutation == 'frame': first['parameters']['pose']['frame'] = 'world'
            if mutation == 'nan': first['parameters']['pose']['xy'][0] = float('nan')
            if mutation == 'sequence': first['step'] = 2
            if mutation == 'support': first['parameters']['pose']['support'] = 'book_sam3'
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'plan.json'
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError): load_action_chain(path)


class GoalTests(unittest.TestCase):
    def setUp(self):
        t = self.task = organize_table_scene003()
        t.table_z_bias = 0
        t.OBJECT_SPECS = deepcopy(scene_003.OBJECT_SPECS)
        t.OBJECT_SPECS.pop('folder')
        t.action_chain, _ = load_action_chain()
        t.action_chain = without_large_book(t.action_chain)
        t.object_geometry = {}
        for name, spec in t.OBJECT_SPECS.items():
            if name in t.PROXY_SIZES:
                w, d, h = t.PROXY_SIZES[name]
                geo = dict(height=h, footprint=np.array([w, d]), bottom_offset=0.)
            else:
                geo = t._asset_geometry(spec)
            t.object_geometry[name] = geo
        t.initial_bottom_z = {name:t._table_z()+t.SURFACE_GAP for name in t.OBJECT_SPECS}
        t.target_bottom_z = dict(t.initial_bottom_z)
        t._final_targets = t._compile_targets()
        t.objects = {}
        for name, target in t._final_targets.items():
            base = None if name in t.PROXY_SIZES else t.RIGID_Y_UP_TO_Z_UP
            pose = sapien.Pose([*target['xy'],target['bottom']-t.object_geometry[name]['bottom_offset']],t._yaw_quat(target['yaw'],base))
            actor = SimpleNamespace(pose=pose,angle=0.)
            actor.get_pose = lambda actor=actor: actor.pose
            t.objects[name] = actor
        t.monitor_initial_pose = sapien.Pose([.025,.205,.741])
        t.monitor = SimpleNamespace(get_pose=lambda: t.monitor_initial_pose)
        t.arm_home = {'left':np.zeros(7),'right':np.zeros(7)}
        t._current_ee_pose = lambda arm: np.zeros(7)
        t.is_left_gripper_open = lambda: True
        t.is_right_gripper_open = lambda: True
        t._score_peak = t._reward_previous = 0
        t.physical_grasp_log = {}
        t.grasp_choices = {}
        t.failure_reason = None

    def test_target_state_scores_without_expert_history(self):
        self.assertNotIn('folder',self.task.objects)
        self.assertNotIn('folder',self.task._final_targets)
        self.assertTrue(self.task.check_success())
        self.assertEqual(self.task.get_score(),100)
        self.assertIs(type(self.task.get_score()), int)
        json.dumps(self.task.goal_checks())

    def test_compact_layout_rejects_old_front_edge_placement(self):
        t=self.task
        self.assertTrue(t.goal_checks()['compact_layout'])
        old={'book':[-.16815,-.1296],'keyboard':[.1007,-.108],
             'mouse':[.21185,-.0784],'phone':[.2052,-.2648],'stapler':[-.305,-.185]}
        for name,xy in old.items():
            actor=t.objects[name]
            actor.pose=sapien.Pose([*xy,actor.pose.p[2]],actor.pose.q)
        self.assertFalse(t.goal_checks()['compact_layout'])
        self.assertLess(t.layout_metrics()['minimum_table_edge_margin'],.10)

    def test_missing_support_fails_even_with_correct_xy(self):
        cup = self.task.objects['cup']
        cup.pose = sapien.Pose([*cup.pose.p[:2],cup.pose.p[2]+.04],cup.pose.q)
        self.assertFalse(self.task.check_success())
        self.assertEqual(self.task.get_score(),75)

    def test_wrong_orientation_fails(self):
        item = self.task.objects['keyboard']
        item.pose = sapien.Pose(item.pose.p,self.task._yaw_quat(90,self.task.RIGID_Y_UP_TO_Z_UP))
        self.assertFalse(self.task.check_success())

    def test_unclosed_props_fail(self):
        for name in ('stapler',):
            with self.subTest(name=name):
                self.task.objects[name].angle = .4
                self.assertFalse(self.task.check_success())
                self.task.objects[name].angle = 0

    def test_closed_gripper_and_arm_away_fail(self):
        self.task.is_left_gripper_open = lambda: False
        self.assertEqual(self.task.get_score(),0)
        self.task.is_left_gripper_open = lambda: True
        self.task._current_ee_pose = lambda arm: np.ones(7)
        self.assertFalse(self.task.check_success())

    def test_transition_reward_does_not_repeat(self):
        self.assertEqual(self.task.run_reward(),100)
        self.assertEqual(self.task.run_reward(),0)
        self.task.objects['stapler'].angle = .5
        self.assertEqual(self.task.run_reward(),0)
        self.task.objects['stapler'].angle = 0
        self.assertEqual(self.task.run_reward(),0)

    def test_cup_buffer_is_clear_of_future_book(self):
        t = self.task
        delta = np.abs(np.asarray(t.stage_targets[1]['xy'])-t._final_targets['book']['xy'])
        required = (t.object_geometry['cup']['footprint']+t.object_geometry['book']['footprint'])/2
        self.assertTrue(np.any(delta > required))

    def test_failed_pick_stops_chain_and_preserves_instruction_generation(self):
        from description.utils.generate_episode_instructions import (
            extract_episodes_from_scene_info, generate_episode_descriptions,
        )
        t = self.task
        t.info = {}
        t.plan_sha256 = load_action_chain()[1]
        t.plan_success = True
        calls = []
        def fail_pick(name, stage):
            calls.append(name)
            return False
        t._pick_and_place = fail_pick
        info = t.play_once()
        self.assertEqual(calls, ['cup'])
        self.assertFalse(t.plan_success)
        self.assertIsNone(t._attached_actor)
        self.assertEqual(info['info'], {})
        episodes = extract_episodes_from_scene_info({'episode_0': info, 'episode_1': info})
        descriptions = generate_episode_descriptions('organize_table_scene003', episodes, 1)
        self.assertEqual(len(descriptions), 2)
        self.assertTrue(all(d['seen'] and d['unseen'] for d in descriptions))


class PhysicalGraspTests(unittest.TestCase):
    def test_side_entry_frame_and_edge_contact(self):
        from envs.utils.scene003_physical import side_grasp_geometry
        for edge in ('front','back','left','right'):
            point, rotation = side_grasp_geometry([0,0,.75],[.19,.145],.016,.75,0.,edge)
            np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rotation),1.)
            self.assertLess(abs(rotation[2,0]),.25)  # approach mostly horizontal
            self.assertGreater(abs(rotation[2,1]),.95)  # jaws pinch top/bottom
            self.assertGreater(np.linalg.norm(point[:2]),.06)  # edge, not center
            self.assertAlmostEqual(point[2],.758)

    def test_no_attachment_or_pose_snap(self):
        task = organize_table_scene003()
        with self.assertRaises(RuntimeError): task._attach_actor('book','left')
        with self.assertRaises(RuntimeError): task._detach_actor_at('book',sapien.Pose())

    def test_book_on_table_without_overhang_is_rejected(self):
        task = organize_table_scene003()
        task.table_z_bias = 0
        task.object_geometry = {'book':dict(height=.016,footprint=np.array([.19,.145]))}
        task.objects = {'book':SimpleNamespace(get_pose=lambda:sapien.Pose([0,0,.741]))}
        task._bottom = lambda name:.741
        candidate, reason = task._side_candidate('book')
        self.assertIsNone(candidate)
        self.assertIn('edge-slide',reason)

    def test_front_overhang_does_not_allow_grasping_supported_back_edge(self):
        task = organize_table_scene003()
        task.table_z_bias = 0
        task.object_geometry = {'book':dict(height=.016,footprint=np.array([.19,.145]))}
        task.objects = {'book':SimpleNamespace(get_pose=lambda:sapien.Pose([0,-.31,.741]))}
        task._bottom = lambda name:.741
        candidate, _ = task._side_candidate('book',('back',))
        self.assertIsNone(candidate)
        candidate, _ = task._side_candidate('book',('front',))
        self.assertIsNotNone(candidate)

    def test_clearance_plan_avoids_occupied_buffer_and_does_not_move_actors(self):
        task = organize_table_scene003()
        poses={'folder':[0,.2,.74],'book':[0,-.1,.74],'phone':[-.2,-.1,.74]}
        task.objects={name:SimpleNamespace(get_pose=lambda p=p:sapien.Pose(p)) for name,p in poses.items()}
        task.object_geometry={name:dict(footprint=np.array([.08,.08])) for name in poses}
        plan,blocked=task._plan_edge_corridor_clearance(np.array([-.1,-.2]),np.array([.1,0.]))
        self.assertFalse(blocked)
        self.assertEqual(plan[0][0],'book')
        self.assertGreater(plan[0][1][0],.1)
        np.testing.assert_allclose(task.objects['book'].get_pose().p,poses['book'])

    def test_unsolvable_clearance_stops_before_executing_partial_plan(self):
        task=organize_table_scene003()
        task.physical_grasp_log={}
        task.objects={'book':SimpleNamespace(get_pose=lambda:sapien.Pose([0,0,.74]))}
        task.object_geometry={'book':dict(footprint=np.array([1.0,.6]))}
        task._push_to_xy=lambda *args: self.fail('Must plan the entire clearance before moving')
        self.assertFalse(task._clear_edge_corridor(np.array([-.5,-.3]),np.array([.5,.3]),'close'))
        self.assertIn('No unobstructed',task.failure_reason['reason'])

    def test_failed_physical_attempts_are_bounded_and_not_collected(self):
        from script.collect_data import run
        class FailedTask:
            attempts = 0
            plan_success = False
            info = {'execution':{'failure_reason':'no physical grasp'}}
            def setup_demo(self, **kwargs): self.attempts += 1
            def play_once(self): pass
            def close_env(self): pass
        with tempfile.TemporaryDirectory() as directory:
            task = FailedTask()
            args = dict(task_name='organize_table_scene003',save_path=directory,use_seed=False,
                        episode_num=2,max_seed_attempts=2,render_freq=0,collect_data=False)
            with self.assertRaisesRegex(RuntimeError,'0/2 successful'):
                run(task,args)
            self.assertEqual(task.attempts,2)
            self.assertEqual(len(list((Path(directory)/'failed_attempts').glob('*.json'))),2)
            self.assertFalse((Path(directory)/'data').exists())


if __name__ == '__main__':
    unittest.main()
