import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from script.symbolic_grasp_execution import SymbolicGraspExecutor


class SymbolicGraspTest(unittest.TestCase):
    def executor(self):
        executor = SymbolicGraspExecutor.__new__(SymbolicGraspExecutor)
        actor = Mock()
        actor.actor = actor
        executor.task = SimpleNamespace(objects={'item': actor}, _yaw_quat=Mock(return_value=[1, 0, 0, 0]))
        executor.env = SimpleNamespace(_clear_attachment_velocity=Mock())
        executor.tip = Mock(return_value=(np.array([.1, .2, .8]), np.eye(3)))
        executor.attachment = None
        return executor, actor

    def test_unattached_object_never_moves(self):
        executor, actor = self.executor()
        executor.sync_attachment()
        actor.set_pose.assert_not_called()
        executor.tip.assert_not_called()

    def test_payload_uses_measured_tip_and_stops_after_release(self):
        executor, actor = self.executor()
        executor.attachment = {'name': 'item', 'arm': 'left', 'offset': np.array([0, 0, -.04]),
                               'orientation': [1, 0, 0, 0], 'wrist_yaw': np.pi/2}
        executor.sync_attachment()
        np.testing.assert_allclose(actor.set_pose.call_args.args[0].p, [.1, .2, .76], atol=1e-7)
        executor.tip.return_value = (np.array([.3, .4, .9]), np.eye(3))
        executor.sync_attachment()
        np.testing.assert_allclose(actor.set_pose.call_args.args[0].p, [.3, .4, .86], atol=1e-7)
        executor.attachment = None
        executor.sync_attachment()
        self.assertEqual(actor.set_pose.call_count, 2)

    def test_occluder_blocks_grasp_but_partial_visibility_alone_does_not(self):
        from script.symbolic_grasp_execution import GraspBlockedError
        executor, actor = self.executor()
        actor.get_pose.return_value = SimpleNamespace(p=np.array([0., 0., .7]))
        obstacle = Mock()
        obstacle.get_pose.return_value = SimpleNamespace(p=np.array([0., 0., .73]))
        executor.task.objects['occluder'] = obstacle
        executor.task.object_geometry = {'item':{'height':.03,'footprint':[.2,.2]},
            'occluder':{'height':.1,'footprint':[.06,.06]}}
        executor.env._object_yaw = lambda name:0.
        executor.env.object_map = {'item_visual':'item','obstacle_visual':'occluder'}
        with self.assertRaises(GraspBlockedError) as caught:
            executor.check_grasp_access('item')
        self.assertEqual(caught.exception.details['blocking_object_ids'], ['obstacle_visual'])
        self.assertNotIn('world_pose', caught.exception.details)
        obstacle.get_pose.return_value.p[0] = .2
        executor.check_grasp_access('item')
        actor.set_pose.assert_not_called()

    def test_missed_grasp_never_attaches_or_transfers(self):
        executor, actor = self.executor()
        actor.get_pose.return_value = SimpleNamespace(p=np.array([0., 0., .7]), q=[1, 0, 0, 0])
        executor.task.object_geometry = {'item': {'height': .03}}
        executor.task._table_z = lambda: .7
        executor.env.held = None
        executor.env.motion_error_type = RuntimeError
        executor.env._arm = lambda name: 'left'
        executor.env._gripper = Mock()
        executor.home = {'right': [0]*6, 'left': [0]*6}
        entity = Mock(); entity.get_root_pose.return_value = SimpleNamespace(p=np.zeros(3))
        executor.models = {'left': (entity,), 'right': (entity,)}
        executor.arm_bases = {'left':np.zeros(3), 'right':np.ones(3)}
        executor.solve = Mock(return_value=np.zeros(6))
        executor.check_grasp_access = Mock()
        executor.drive = Mock()
        executor.move = Mock()
        with self.assertRaisesRegex(RuntimeError, 'Cannot attach'):
            executor.place({'target': 'table'}, 'item')
        self.assertIsNone(executor.attachment)
        self.assertIsNone(executor.env.held)
        self.assertEqual([c.args[2] for c in executor.move.call_args_list], ['grasp_approach', 'grasp_contact'])
        actor.set_pose.assert_not_called()


if __name__ == '__main__':
    unittest.main()
