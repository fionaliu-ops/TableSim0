"""Contract checks for assisted grasping without target-pose teleportation."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from script.dynamic_sim import TableSimAdapter, DynamicSimError


class ContinuousFollowTest(unittest.TestCase):
    def test_release_preserves_actual_pose_instead_of_snapping_to_target(self):
        class Pose:
            def __init__(self, p, q=(1, 0, 0, 0)):
                self.p, self.q = np.array(p, dtype=float), np.array(q, dtype=float)
            def to_transformation_matrix(self):
                matrix = np.eye(4)
                matrix[:3, 3] = self.p
                return matrix
        actor = Mock()
        state = {'pose': Pose([0, 0, .7]), 'ee': np.array([0, 0, 1, 1, 0, 0, 0.])}
        actor.get_pose.side_effect = lambda: state['pose']
        task = SimpleNamespace(objects={'book': actor}, object_geometry={'book': {'height': .02}},
            _top_down_pose=lambda xy, top, arm, clearance: [*xy, top+clearance, 1, 0, 0, 0],
            _current_ee_pose=lambda arm: state['ee'].copy(),
            _matrix_pose=lambda matrix: Pose(matrix[:3, 3]), _detach_actor_at=Mock())
        env = TableSimAdapter(task, {}, '.')
        env._arm = lambda name: 'left'
        env._destination = lambda *args: (np.array([.2, .1]), .7)
        env._clear_attachment_velocity = Mock()
        env._gripper = Mock()
        def attach(*args):
            task._ee_to_actor = np.eye(4)
            task._ee_to_actor[:3, 3] = state['pose'].p-state['ee'][:3]
        task._attach_actor = attach
        def motion(arm, pose, label):
            state['ee'] = np.array(pose, dtype=float)
            if label == 'follow_lower':
                state['ee'][0] += .007  # Realistic residual tracking error.
            if env.held is not None:
                state['pose'] = Pose(state['ee'][:3]+task._ee_to_actor[:3, 3])
            return True
        env._graph_validation_motion = motion
        with patch.dict('sys.modules', {'sapien': SimpleNamespace(Pose=Pose)}):
            env._graph_validation_place({'kind': 'move', 'parameters': {}}, 'book')
        released = task._detach_actor_at.call_args.args[1]
        np.testing.assert_allclose(released.p, [.207, .1, .7])
        actor.set_pose.assert_not_called()
        actor.actor.set_pose.assert_not_called()
        self.assertIsNone(env.held)

    def test_failed_validation_step_never_rolls_back_objects_or_robot(self):
        actor = Mock()
        task = SimpleNamespace(objects={'book': actor})
        env = TableSimAdapter(task, {}, '.')
        env.graph_validation_mode = True
        env._name = lambda source: 'book'
        env._snapshot_robot_state = Mock()
        env._restore_robot_state = Mock()
        env._restore_failed_pick = Mock()
        def failed_place(*args):
            env.held = ('book', 'left')
            raise DynamicSimError('transfer failed')
        env._graph_validation_place = failed_place
        request = {'instruction': 'move book', 'object_target': {'source': 'book'},
                   'metadata': {'reasoning_edge': {'action_type': 'move'}}}
        with self.assertRaises(DynamicSimError):
            env.step(request)
        actor.set_pose.assert_not_called()
        actor.actor.set_pose.assert_not_called()
        env._restore_robot_state.assert_not_called()
        env._restore_failed_pick.assert_not_called()
        self.assertEqual(env.held, ('book', 'left'))

    def test_failed_motion_is_not_reported_as_reached(self):
        task = SimpleNamespace(plan_success=True, move=Mock(return_value=False),
                               move_to_pose=Mock(), _arm_tag=lambda arm: arm,
                               _current_ee_pose=lambda arm: [0,0,1,1,0,0,0])
        env = TableSimAdapter(task, {}, '.')
        env._capture_motion_frame = Mock()
        with self.assertRaises(DynamicSimError):
            env._graph_validation_motion('left',[0,0,1,1,0,0,0],'test')

    def test_reported_success_requires_measured_gripper_arrival(self):
        task = SimpleNamespace(plan_success=True, move=Mock(return_value=True),
                               move_to_pose=Mock(), _arm_tag=lambda arm: arm,
                               _current_ee_pose=lambda arm: [0,0,1,1,0,0,0])
        env = TableSimAdapter(task, {}, '.')
        env._capture_motion_frame = Mock()
        with self.assertRaises(DynamicSimError):
            env._graph_validation_motion('left',[.3,0,1,1,0,0,0],'test')

    def test_failed_grasp_approach_never_attaches_or_places_object(self):
        actor = Mock()
        actor.get_pose.return_value = SimpleNamespace(p=np.array([0.,0.,.7]))
        task = SimpleNamespace(objects={'book': actor}, object_geometry={'book':{'height':.02}},
                               _top_down_pose=lambda *a,**kw: [0,0,1,1,0,0,0],
                               _attach_actor=Mock(), _detach_actor_at=Mock())
        env = TableSimAdapter(task, {}, '.')
        env._arm = lambda name:'left'
        env._gripper = Mock()
        env._graph_validation_motion = Mock(side_effect=DynamicSimError('unreachable'))
        with patch.dict('sys.modules', {'sapien': SimpleNamespace()}):
            with self.assertRaises(DynamicSimError):
                env._graph_validation_place({'parameters':{}},'book')
        task._attach_actor.assert_not_called()
        task._detach_actor_at.assert_not_called()
        actor.set_pose.assert_not_called()
        self.assertIsNone(env.held)


if __name__ == '__main__':
    unittest.main()
