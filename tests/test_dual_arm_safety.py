"""Safety regressions: swept clearance and literal image goals."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from script.dual_arm_safety import DualArmSafety, ArmClearanceError, sphere_clearance


class DualArmSafetyTests(unittest.TestCase):
    def test_surface_distance_includes_both_radii(self):
        left = np.array([[0., 0., 0., .04]])
        right = np.array([[.12, 0., 0., .03]])
        self.assertAlmostEqual(sphere_clearance(left, right), .05)

    def test_swept_crossing_is_rejected_before_execution(self):
        guard = DualArmSafety.__new__(DualArmSafety)
        guard.minimum_clearance = .05
        guard.planning_margin = .025
        guard.indices = {'left': [0]}
        guard.robot = SimpleNamespace(left_entity=SimpleNamespace(get_qpos=lambda: np.array([-1.])))
        guard.stats = {'checked_trajectory_samples': 0, 'rejected_trajectories': 0}
        guard.save = lambda: None
        guard.spheres = lambda arm, q=None: np.array([[q[0] if q is not None else 0., 0., 0., .04]])
        # Both endpoints are safe, but the interpolated trajectory crosses right.
        with self.assertRaises(ArmClearanceError):
            guard.validate({'left_arm': {'position': np.array([[1.]])}})
        self.assertEqual(guard.stats['rejected_trajectories'], 1)

    def test_two_active_arms_are_rejected(self):
        guard = DualArmSafety.__new__(DualArmSafety)
        with self.assertRaises(ArmClearanceError):
            guard.validate({'left_arm': {}, 'right_arm': {}})

    def test_planning_boundary_allows_only_configured_numeric_band(self):
        guard = DualArmSafety.__new__(DualArmSafety)
        guard.minimum_clearance = .05
        guard.planning_margin = .085
        guard.planning_tolerance = .006
        guard.indices = {'right': [0]}
        guard.robot = SimpleNamespace(right_entity=SimpleNamespace(get_qpos=lambda: np.array([0.])))
        guard.stats = {'checked_trajectory_samples': 0, 'rejected_trajectories': 0}
        guard.save = lambda: None
        clearance = [.13475]
        guard.spheres = lambda arm, q=None: np.array([[
            0. if arm == 'right' else clearance[0] + .08, 0., 0., .04]])
        guard.validate({'right_arm': {'position': np.array([[0.]])}})
        clearance[0] = .1284
        with self.assertRaises(ArmClearanceError):
            guard.validate({'right_arm': {'position': np.array([[0.]])}})

    def test_runtime_tracking_hysteresis_is_small_and_bounded(self):
        guard = DualArmSafety.__new__(DualArmSafety)
        guard.minimum_clearance = .05
        guard.runtime_tolerance = .01
        guard.scene = None
        guard.stats = {'checked_control_steps': 0, 'minimum_observed_m': None,
                       'runtime_violations': 0}
        guard.save = lambda: None
        clearance = [.045]
        guard.spheres = lambda arm: np.array([[
            0. if arm == 'left' else clearance[0] + .08, 0., 0., .04]])
        guard.check_live()
        clearance[0] = .039
        with self.assertRaises(ArmClearanceError):
            guard.check_live()

    def test_goal_rejects_wrong_image(self):
        from script.scene003_image_goal import Scene003ImageGoal
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'wrong.png'
            path.write_bytes(b'wrong goal')
            with self.assertRaises(ValueError):
                Scene003ImageGoal(None, path)

    def test_image_goal_does_not_accept_region_only_success(self):
        from script.scene003_image_goal import Scene003ImageGoal, CENTERS
        goal = Scene003ImageGoal.__new__(Scene003ImageGoal)
        goal.targets = {'phone': np.array([.45, .42])}
        goal.k = np.eye(3); goal.extrinsic = np.eye(4); goal.size = np.ones(2)
        actor = SimpleNamespace(get_pose=lambda: SimpleNamespace(p=np.array([.45, .1, 1.])))
        goal.task = SimpleNamespace(objects={'phone': actor}, object_geometry={'phone': {'height': 0.}})
        env = SimpleNamespace(orientation_status=lambda _: {'phone': {'passed': True}})
        result = goal.evaluate(env)
        self.assertFalse(result['passed'])
        self.assertGreater(result['objects']['phone']['position_error_m'], .025)

class CollisionGeometryTests(unittest.TestCase):
    def test_height_slice_interpolates_edges_without_inside_vertices(self):
        from script.collision_geometry import height_band_bounds
        # A tapered solid: a low slice must not use its wide upper footprint.
        vertices = np.array([[x*r, y*r, z] for z,r in ((0., .1), (1., 1.))
                             for x in (-1.,1.) for y in (-1.,1.)])
        center, size = height_band_bounds(vertices, .2, .3)
        np.testing.assert_allclose(center, [0., 0.])
        np.testing.assert_allclose(size, [.74, .74])
        self.assertIsNone(height_band_bounds(vertices, 1.1, 2.))

class SafetyHookTests(unittest.TestCase):
    def test_release_hook_replacement_cannot_disable_monitor(self):
        from unittest.mock import patch, Mock
        from tests.test_dynamic_sim import Task
        from script.dynamic_sim import TableSimAdapter
        task = Task()
        task.robot = SimpleNamespace()
        task.setup_demo = lambda **kwargs: None
        task._sync_attached_actor = lambda: None
        task.take_dense_action = lambda control, save_freq: task._sync_attached_actor()
        guard = Mock()
        with tempfile.TemporaryDirectory() as directory, patch('script.dual_arm_safety.DualArmSafety', return_value=guard):
            env = TableSimAdapter(task, {}, directory)
            env.reset()
            before = guard.check_live.call_count
            replacement = Mock()
            task._sync_attached_actor = replacement
            task.take_dense_action({})
            replacement.assert_called_once()
            self.assertEqual(guard.check_live.call_count, before + 1)
            self.assertIs(task._sync_attached_actor, replacement)

class OrientationRegressionTests(unittest.TestCase):
    def test_correct_yaw_with_tilt_still_requires_wrist_correction(self):
        import math
        import transforms3d as t3d
        from tests.test_dynamic_sim import Task
        from script.dynamic_sim import TableSimAdapter
        task = Task()
        pose = task.objects['book'].pose
        matrix = np.eye(4)
        matrix[:3,:3] = t3d.euler.euler2mat(math.radians(110.), 0., 0.)
        matrix[:3,3] = pose.p
        pose.to_transformation_matrix = lambda: matrix.copy()
        task.RIGID_Y_UP_TO_Z_UP = [2**-.5, 2**-.5, 0., 0.]
        task._yaw_quat = lambda yaw, base: t3d.quaternions.qmult(
            t3d.quaternions.axangle2quat([0,0,1], math.radians(yaw)), base)
        task._ee_to_actor = np.eye(4)
        task.robot = SimpleNamespace(right_entity=SimpleNamespace(get_qpos=lambda: np.zeros(1)))
        commands = []
        with tempfile.TemporaryDirectory() as directory:
            env = TableSimAdapter(task, {}, directory)
            def plan(arm, target, qpos):
                commands.append(target)
                return {'status': 'Success', 'position': np.zeros((1,1))}, np.zeros(1)
            def execute(*args):
                matrix[:3,:3] = t3d.quaternions.quat2mat(commands[-1][3:])
                pose.q = np.asarray(commands[-1][3:])
            env._plan_arm_motion = plan
            env._execute_planned_motion = execute
            env._rotate_held_to_yaw('book', 'right', 0.)
            self.assertTrue(commands)
            self.assertAlmostEqual(matrix[2,1], 1.)

class MotionRecoveryTests(unittest.TestCase):
    def test_planning_failure_uses_prevalidated_route(self):
        from unittest.mock import Mock
        from tests.test_dynamic_sim import Task
        from script.dynamic_sim import TableSimAdapter
        task = Task()
        task.move = lambda _: False
        with tempfile.TemporaryDirectory() as directory:
            env = TableSimAdapter(task, {}, directory)
            env._route_above_other_arm = Mock(return_value=True)
            env._motion('right', [0., 0., 1.], 'home')
            env._route_above_other_arm.assert_called_once()

    def test_rejected_segment_executes_nothing_before_complete_fallback(self):
        from unittest.mock import Mock
        from tests.test_dynamic_sim import Task
        from script.dynamic_sim import TableSimAdapter
        task = Task()
        task.robot = SimpleNamespace(right_entity=SimpleNamespace(get_qpos=lambda: np.zeros(1)))
        task._current_ee_pose = lambda arm: [0., 0., .9, 1., 0., 0., 0.]
        env = TableSimAdapter(task, {}, '.')
        env._execute_planned_motion = Mock()
        planned = []
        def plan(arm, target, qpos):
            self.assertEqual(env._execute_planned_motion.call_count, 0)
            planned.append(target)
            # Reject each overhead route's second segment. The front route
            # must be planned completely before either segment is executed.
            if len(planned) <= 6 and len(planned) % 2 == 0:
                return {'status': 'InterArmClearanceRejected'}, None
            return {'status': 'Success'}, np.zeros(1)
        env._plan_arm_motion = plan
        self.assertTrue(env._route_above_other_arm('right', [.3, .1, .9, 1., 0., 0., 0.], 'approach'))
        self.assertEqual(len(planned), 8)
        np.testing.assert_allclose(planned[-2][:3], [.2, -.2, 1.])
        self.assertEqual(env._execute_planned_motion.call_count, 2)

    def test_runtime_clearance_violation_is_never_retried(self):
        from unittest.mock import Mock
        from tests.test_dynamic_sim import Task
        from script.dynamic_sim import TableSimAdapter
        task = Task()
        task.move = Mock(side_effect=ArmClearanceError('runtime violation'))
        env = TableSimAdapter(task, {}, '.')
        env._route_above_other_arm = Mock()
        with self.assertRaises(ArmClearanceError):
            env._motion('right', [0., 0., 1.], 'home')
        env._route_above_other_arm.assert_not_called()

class ArmAssignmentTests(unittest.TestCase):
    def test_arm_assignment_follows_live_workspace_after_handoff(self):
        from tests.test_dynamic_sim import Task, Actor
        from script.dynamic_sim import TableSimAdapter
        task = Task()
        task.objects['phone'] = Actor([.08, -.23, .74])
        task.OBJECT_SPECS['phone'] = {'arm': 'right'}
        task.object_geometry['phone'] = {'height': .02, 'footprint': np.array([.06, .03])}
        task.robot = SimpleNamespace()
        env = TableSimAdapter(task, {}, '.')
        self.assertEqual(env._arm('phone'), 'right')
        task.objects['phone'].pose.p[0] = -.03
        self.assertEqual(env._arm('phone'), 'left')


class ExternalModelRecoveryTests(unittest.TestCase):
    def test_held_object_buffer_is_not_overwritten_by_final_goal(self):
        from script.direct_dynamic_sim import constrained_scene003_action
        pose = SimpleNamespace(p=np.array([.16, .08, .74]))
        task = SimpleNamespace(objects={'mouse': SimpleNamespace(get_pose=lambda: pose)})
        env = SimpleNamespace(
            held=('mouse', 'right'),
            task=task,
            object_map={'mouse_id': 'mouse'},
            image_goal=SimpleNamespace(targets={'mouse': np.array([.16, .08])}),
            transform=np.array([[1., 0.], [0., 1.], [0., 0.]]),
        )
        proposed = {
            'action_type': 'move', 'source_object_id': 'mouse_id', 'target': 'table',
            'pose': {'frame': 'image_normalized', 'xy': [.82, .62], 'support': 'table'},
            'reason': 'recover to an empty buffer',
        }
        action, override = constrained_scene003_action(env, {'mouse': 'mouse_id'}, proposed)
        self.assertEqual(action['pose']['xy'], [.82, .62])
        self.assertEqual(action['reason'], 'recover to an empty buffer')
        self.assertTrue(override['temporary'])
        self.assertIn('recovery buffer', override['reason'])
