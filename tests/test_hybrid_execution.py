import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from script.hybrid_execution import HybridExecutor, smooth_joint_path, supervise_action


class HybridExecutionTest(unittest.TestCase):
    def test_retries_exception_owned_by_runtime_adapter(self):
        class EntrypointMotionError(RuntimeError):
            pass
        with tempfile.TemporaryDirectory() as directory:
            executor = self.executor(directory)
            executor.env.motion_error_type = EntrypointMotionError
            executor.task.plan_success = True
            executor.env._graph_validation_motion = Mock(
                side_effect=[EntrypointMotionError('failed'), None])
            executor._move_waypoint('left', np.array([0., 0., 1., 1, 0, 0, 0]), 'test')
            self.assertEqual(executor.env._graph_validation_motion.call_count, 2)

    def test_drive_path_has_bounded_increments_and_zero_velocity_hold(self):
        result = smooth_joint_path({'position': [[.03, -.02], [.04, 0]]}, [0, 0],
                                   max_step=.002, settle_ticks=5)
        differences = np.diff(np.vstack(([0, 0], result['position'])), axis=0)
        self.assertLessEqual(np.max(np.abs(differences)), .002000001)
        np.testing.assert_allclose(result['position'][-5:], [[.04, 0]]*5)
        self.assertFalse(result['velocity'].any())

    def executor(self, directory):
        pose = SimpleNamespace(p=np.array([0., 0., .7]))
        actor = Mock(); actor.get_pose.return_value = pose
        task = SimpleNamespace(objects={'item': actor}, _attach_actor=Mock(),
                               _current_ee_pose=lambda arm: np.array([0., 0., 1., 1, 0, 0, 0]))
        env = SimpleNamespace(task=task, output_dir=Path(directory), step_index=0,
                              held=None, _gripper=Mock(), _clear_attachment_velocity=Mock())
        executor = HybridExecutor(env)
        executor.grasp_pose = Mock(return_value=(np.array([0., 0., 1., 1, 0, 0, 0]), np.eye(3)))
        executor.body = Mock(return_value=Mock())
        executor.move = Mock()
        return executor

    def test_verified_contact_lift_never_attaches(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = self.executor(directory)
            executor.eligible = Mock(return_value=(True, 'fits'))
            executor.contacts = Mock(return_value=True)
            def lift(arm, pose, label):
                if label == 'contact_test_lift':
                    executor.task.objects['item'].get_pose().p[2] += .025
            executor.move.side_effect = lift
            self.assertEqual(executor.pick('item', 'left', 'auto'), 'physical_contact')
            executor.task._attach_actor.assert_not_called()
            executor.body().set_kinematic.assert_called_once_with(False)

    def test_contact_failure_reacquires_before_following(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = self.executor(directory)
            executor.eligible = Mock(return_value=(True, 'fits'))
            executor.contacts = Mock(return_value=False)
            self.assertEqual(executor.pick('item', 'left', 'auto'), 'assisted_follow')
            labels = [call.args[2] for call in executor.move.call_args_list]
            self.assertEqual(labels[-2:], ['fallback_approach', 'fallback_grasp'])
            executor.task._attach_actor.assert_called_once_with('item', 'left')
            executor.task.objects['item'].set_pose.assert_not_called()

    def test_no_physical_claim_for_oversize_object(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = self.executor(directory)
            executor.eligible = Mock(return_value=(False, 'too_wide'))
            executor.contacts = Mock()
            self.assertEqual(executor.pick('item', 'right', 'auto'), 'assisted_follow')
            executor.contacts.assert_not_called()
            self.assertNotIn(False, [c.args[0] for c in executor.body().set_kinematic.call_args_list])

    def test_contact_only_failure_does_not_silently_attach(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = self.executor(directory)
            executor.eligible = Mock(return_value=(True, 'fits'))
            executor.contacts = Mock(return_value=False)
            with self.assertRaises(RuntimeError):
                executor.pick('item', 'left', 'contact')
            executor.task._attach_actor.assert_not_called()

    def test_supervisor_cannot_rewrite_high_level_target(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p/'decision_000.json').write_text(json.dumps({'execute': True, 'target': [1, 2]}))
            with self.assertRaises(ValueError):
                supervise_action(p, 0, {'target': [0, 0]}, 'camera.png')


if __name__ == '__main__':
    unittest.main()
