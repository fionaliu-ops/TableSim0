"""Contract tests for the simulator/planner boundary, without a GPU runtime."""
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from script.dynamic_sim import (DynamicSimError, TableSimAdapter,
                                fit_image_to_world, request_action,
                                separated_holding_pose_is_safe)


class Pose:
    def __init__(self, xyz):
        self.p = np.asarray(xyz, dtype=float)
        self.q = np.asarray([1, 0, 0, 0], dtype=float)


class Actor:
    def __init__(self, xyz):
        self.pose = Pose(xyz)

    def get_pose(self):
        return self.pose

    def set_pose(self, pose):
        self.pose = pose


class Task:
    def __init__(self):
        self.objects = {"book": Actor([-.2, 0, .75]), "cup": Actor([.1, .1, .75]),
                        "mouse": Actor([.2, -.1, .75])}
        self.OBJECT_SPECS = {name: {"arm": "left"} for name in self.objects}
        self.object_geometry = {name: {"height": .04} for name in self.objects}
        self.plan_success = True
        self._attached_actor = None
        self._attached_arm = None
        self._ee_to_actor = None
        self.commands = []
        self.scene = SimpleNamespace(step=lambda: None)

    def _arm_tag(self, arm):
        return arm

    def _top_down_pose(self, xy, top, arm, clearance=0):
        return [float(xy[0]), float(xy[1]), float(top + clearance)]

    def move_to_pose(self, arm, pose):
        return ("move", arm, pose)

    def close_gripper(self, arm, pos=0):
        return ("close", arm, pos)

    def open_gripper(self, arm):
        return ("open", arm)

    def move(self, command):
        self.commands.append(command)
        if command[0] == "move" and self._attached_actor is not None:
            self._attached_actor.pose.p[:2] = command[2][:2]
        return True

    def _attach_actor(self, name, arm):
        self._attached_actor = self.objects[name]
        self._attached_arm = arm

    def _table_z(self):
        return .74

    def get_obs(self):
        return {"observation": {"head_camera": {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}}}


def graph(points):
    return SimpleNamespace(objects=[SimpleNamespace(id=name + "_sam3", name=name,
        attributes=SimpleNamespace(xy=xy)) for name, xy in points.items()])


class DynamicSimTests(unittest.TestCase):
    def _side_grasp_env(self, directory):
        env = TableSimAdapter.__new__(TableSimAdapter)
        keyboard = Actor([0., 0., .75])
        mouse = Actor([.2, .1, .75])
        entity = SimpleNamespace(get_qpos=lambda: np.zeros(2))
        env.task = SimpleNamespace(
            objects={"keyboard": keyboard, "mouse": mouse},
            object_geometry={
                "keyboard": {"height": .02, "footprint": np.array([.18, .06])},
                "mouse": {"height": .04, "footprint": np.array([.06, .04])}},
            robot=SimpleNamespace(left_entity=entity, right_entity=entity))
        env.arm_safety = None
        env.output_dir = Path(directory)
        env.step_index = 0
        env._side_grasp_preflight = False
        return env

    def test_side_grasp_pure_ik_failure_does_not_invent_nearest_blocker(self):
        with TemporaryDirectory() as directory:
            env = self._side_grasp_env(directory)
            env._side_grasp_obstacle_clearance = lambda *args: (.2, "mouse")
            env._plan_arm_motion = lambda *args: ({"status": "IK_FAIL"}, None)
            with self.assertRaises(DynamicSimError) as raised:
                env._select_side_grasp("keyboard", "left", .02, 0.)
            self.assertEqual(raised.exception.blockers, ())

    def test_side_grasp_reports_measured_corridor_blocker(self):
        with TemporaryDirectory() as directory:
            env = self._side_grasp_env(directory)
            env._side_grasp_obstacle_clearance = lambda *args: (0., "mouse")
            env._plan_arm_motion = lambda *args: self.fail(
                "IK must not run for a geometrically blocked insertion")
            with self.assertRaises(DynamicSimError) as raised:
                env._select_side_grasp("keyboard", "left", .02, 0.)
            self.assertEqual(raised.exception.blockers, ("mouse",))

    def test_side_grasp_searches_another_wrist_pitch_after_ik_failure(self):
        with TemporaryDirectory() as directory:
            env = self._side_grasp_env(directory)
            env._side_grasp_obstacle_clearance = lambda *args: (.2, None)
            calls = {"count": 0}
            def plan(_arm, _pose, qpos):
                calls["count"] += 1
                if calls["count"] == 1:
                    return {"status": "IK_FAIL"}, None
                return {"status": "Success"}, np.asarray(qpos)
            env._plan_arm_motion = plan
            selected = env._select_side_grasp("keyboard", "left", .02, 0.)
            self.assertEqual(selected["pitch"], 18.)

    def test_park_arm_settles_velocity_even_when_pose_is_already_home(self):
        env = TableSimAdapter.__new__(TableSimAdapter)
        state = {"velocity": .2, "settles": 0}
        entity = SimpleNamespace(
            get_qpos=lambda: np.array([.1, .2]),
            get_qvel=lambda: np.array([state["velocity"], 0.]),
        )
        robot = SimpleNamespace(
            left_original_pose=np.array([0., 0., 1., 1., 0., 0., 0.]),
            left_entity=entity,
        )
        env.task = SimpleNamespace(
            robot=robot,
            _current_ee_pose=lambda arm: robot.left_original_pose.copy(),
        )
        env.held = None
        env.image_goal = None
        env.arm_safety = SimpleNamespace(indices={"left": [0, 1]})
        env._motion = lambda *args: self.fail(
            "an arm already at home should only need velocity settling")
        def execute(_arm, _plan, label):
            state["settles"] += 1
            state["velocity"] = 0.
            self.assertEqual(label, "left_park_settle")
        env._execute_planned_motion = execute
        env._park_arm("left")
        self.assertEqual(state["settles"], 1)

    def test_yaw_uses_live_pose_and_wraps_angle_error(self):
        task = Task()
        angle = np.deg2rad(-179.)
        matrix = np.eye(4)
        matrix[:2,:2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        task.objects['book'].pose.to_transformation_matrix = lambda: matrix
        task.OBJECT_SPECS['book']['initial_yaw'] = -10.
        env = TableSimAdapter(task, {}, '/tmp/yaw_contract')
        check = env.orientation_status({'book':179.})['book']
        self.assertAlmostEqual(check['actual_yaw_degrees'], -179.)
        self.assertAlmostEqual(check['error_degrees'], 2.)
        self.assertTrue(check['passed'])
        self.assertFalse(env.orientation_status({'book':90.})['book']['passed'])

    def test_request_reads_atomic_edge_not_language(self):
        request = {"instruction": "do something", "object_target": {"source": "cup_sam3", "target": "table"},
                   "metadata": {"reasoning_edge": {"action_type": "move", "parameters": {
                       "pose": {"frame": "image_normalized", "xy": [.5, .5]}}}}}
        self.assertEqual(request_action(request)["kind"], "move")
        request["metadata"]["reasoning_edge"]["action_type"] = "generic"
        with self.assertRaises(DynamicSimError):
            request_action(request)

    def test_affine_grounding_and_inconsistent_scene_rejected(self):
        task = Task()
        image = {"book": [0, 0], "cup": [1, 0], "mouse": [0, 1]}
        mapping = {name + "_sam3": name for name in image}
        transform = fit_image_to_world(graph(image), mapping, task)
        np.testing.assert_allclose(np.r_[.5, .5, 1] @ transform, [.15, .0], atol=1e-12)
        with self.assertRaises(DynamicSimError):
            fit_image_to_world(graph({"book": [0, 0], "cup": [1, 0], "mouse": [2, 0]}), mapping, task)

    def test_request_drives_pick_follow_and_release(self):
        from tempfile import TemporaryDirectory
        task = Task()
        # A scene may expose legacy helpers with hardcoded destinations.
        # Executing a generated request must never call those helpers.
        def legacy_trajectory(*args):
            raise AssertionError("Generated destination was bypassed")
        task._pick_and_place = legacy_trajectory
        with TemporaryDirectory() as directory:
            env = TableSimAdapter(task, {}, directory)
            env.object_map = {name + "_sam3": name for name in task.objects}
            env.transform = fit_image_to_world(graph({"book": [0, 0], "cup": [1, 0], "mouse": [0, 1]}),
                                               env.object_map, task)
            request = {"instruction": "move cup", "object_target": {"source": "cup_sam3", "target": "table"},
                       "metadata": {"reasoning_edge": {"action_type": "move", "parameters": {
                           "pose": {"frame": "image_normalized", "xy": [.5, .5]}}}}}
            result = env.step(request)
            self.assertEqual(result["action"]["kind"], "move")
            self.assertIsNone(env.held)
            self.assertIsNone(task._attached_actor)
            self.assertTrue(any(c[0] == "close" for c in task.commands))
            self.assertTrue(any(c[0] == "open" for c in task.commands))
            np.testing.assert_allclose(task.objects["cup"].pose.p[:2], [.15, .0], atol=1e-12)

    def test_nonstack_placement_and_transfer_reject_nearby_objects(self):
        task = Task()
        task.object_geometry = {name: {"height": .04, "footprint": [.04, .04]}
                                for name in task.objects}
        env = TableSimAdapter(task, {}, ".")
        self.assertFalse(env._clear_table_placement("cup", task.objects["mouse"].get_pose().p[:2]))
        self.assertTrue(env._clear_table_placement("cup", [0, .3]))
        # An object may leave the footprint of the support it was picked
        # from, but it may not enter a different obstacle later in the sweep.
        self.assertTrue(env._clear_transfer_corridor("cup", [-.2, 0], [.2, 0], .76))
        self.assertFalse(env._clear_transfer_corridor("cup", [-.2, 0], [.2, -.1], .76))
        self.assertTrue(env._clear_transfer_corridor("cup", [-.2, 0], [.2, 0], .85))
        self.assertFalse(TableSimAdapter._rectangles_overlap(
            [.1, -.08], [.12, .04], 0, [.1, .04], [.07, .07], 0, .01))
        self.assertTrue(TableSimAdapter._rectangles_overlap(
            [.1, -.02], [.12, .04], 0, [.1, .04], [.07, .07], 0, .01))

    def test_transfer_search_tries_ik_feasible_route_after_direct_failure(self):
        task = Task()
        task.object_geometry["cup"] = {"height": .04, "footprint": [.05, .05]}
        env = TableSimAdapter(task, {}, ".")
        env.transform = np.array([[1., 0.], [0., 1.], [0., 0.]])
        env._clear_transfer_corridor = lambda *args, **kwargs: True

        def plan(_arm, pose, qpos):
            point = np.asarray(pose[:2])
            # The direct goal is unreachable from the initial arm branch.  A
            # live intermediate waypoint changes the branch and makes it
            # reachable, which a geometry-only shortest path would miss.
            if np.allclose(point, [1., 0.]) and qpos[0] < .5:
                return {"status": "IK failed"}, None
            next_qpos = np.asarray(qpos).copy()
            if point[1] > .01:
                next_qpos[0] = 1.
            return {"status": "Success", "position": np.array([next_qpos])}, next_qpos

        env._plan_arm_motion = plan
        plans, final_qpos = env._plan_transfer_waypoints(
            "cup", "left", np.array([0., 0., .75]), np.array([0., 0., .9]),
            np.array([1., 0.]), .75, .08, np.array([0.]))
        self.assertGreaterEqual(len(plans), 2)
        self.assertEqual(final_qpos[0], 1.)

    def test_static_book_stack_rejected_before_kinematic_pick(self):
        task = Task()
        static_type = type("PhysxRigidStaticComponent", (), {})
        task.objects["book"].actor = SimpleNamespace(get_components=lambda: [static_type()])
        env = TableSimAdapter(task, {}, ".")
        with self.assertRaisesRegex(DynamicSimError, "static actor"):
            env._place({"kind": "stack", "target": "cup_sam3", "parameters": {}}, "book")
        self.assertEqual(task.commands, [])

    def test_live_runner_rejects_cached_graph_before_observation(self):
        from script.dynamic_sim import run_closed_loop
        with self.assertRaisesRegex(DynamicSimError, "cached"):
            run_closed_loop(None, None, "organize", initial_graph=object())

    def test_perturbation_changes_simulation_state(self):
        task = Task()
        env = TableSimAdapter(task, {}, ".")
        env.object_map = {"cup_sam3": "cup"}
        result = env.inject_event({"type": "move_object", "object_id": "cup_sam3",
                                   "delta_xy": [.08, -.04]})
        np.testing.assert_allclose(result["new_pose"][:2], [.18, .06])
        with self.assertRaises(DynamicSimError):
            env.inject_event({"type": "identity_swap", "object_id": "cup_sam3"})

    def test_separated_holding_accepts_live_safe_pose_without_point_chasing(self):
        held=np.array([.20,-.20,.82])
        support=np.array([-.10,.05,.74])
        self.assertTrue(separated_holding_pose_is_safe(
            held,support,.20,.11,.09))
        self.assertFalse(separated_holding_pose_is_safe(
            held,support,.20,.08,.09))
        self.assertFalse(separated_holding_pose_is_safe(
            np.array([-.02,.02,.82]),support,.20,.11,.09))


if __name__ == "__main__":
    unittest.main()
