"""Arrange a compact desk inspired by ``scenes_constrained_v2/scene_000.png``.

Ordered task:

1. move the cup from the top of the book to clear space on the left;
2. place the keyboard in front of the monitor and the mouse beside it;
3. place the notebook squarely on the book.

Every manipulated object is a dynamic rigid body.  No object pose is changed
during the demonstration: motion is produced only by two-finger contact,
friction, gravity, and the robot trajectory.  A stage is rejected unless both
finger links contact the object and the object is measurably lifted before the
transfer begins.
"""

from __future__ import annotations

import numpy as np
import transforms3d as t3d

from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from ._base_task import Base_Task
from .clean_the_desk_009 import clean_the_desk_009
from .utils import create_actor


class arrange_scene_000(clean_the_desk_009):
    STACK_GAP = 0.006
    SURFACE_GAP = 0.004

    # Coordinates are compressed into ALOHA's reliable bimanual workspace.
    # Positive Y is the rear of the table; the robot and camera are at -Y.
    OBJECT_SPECS = {
        "book": {
            "source": "043_book",
            "model_id": 1,
            "scale": [0.60, 0.60, 0.60],
            "initial_xy": [-0.10, 0.035],
            "initial_yaw": 0.0,
            "target_xy": [-0.10, 0.035],
            "target_yaw": 0.0,
            "arm": "left",
        },
        "cup": {
            "source": "021_cup",
            "model_id": 3,
            "scale": [0.80, 0.80, 0.80],
            "initial_xy": [-0.10, 0.035],
            "initial_yaw": 20.0,
            "target_xy": [-0.200, -0.170],
            "target_yaw": 20.0,
            "arm": "left",
            "grasp_yaw_offset": 0.0,
        },
        "keyboard": {
            "source": "116_keyboard",
            "model_id": 3,
            "scale": [0.75, 0.75, 0.75],
            "initial_xy": [0.190, -0.175],
            "initial_yaw": 0.0,
            "target_xy": [0.000, -0.080],
            "target_yaw": 0.0,
            "arm": "right",
            "grasp_yaw_offset": 90.0,
        },
        "mouse": {
            "source": "047_mouse",
            "model_id": 0,
            "scale": [0.90, 0.90, 0.90],
            "initial_xy": [0.285, -0.040],
            "initial_yaw": 0.0,
            "target_xy": [0.125, -0.080],
            "target_yaw": 0.0,
            "arm": "right",
            "grasp_yaw_offset": 0.0,
        },
        "notebook": {
            "source": "092_notebook",
            "model_id": 2,
            "scale": [0.65, 0.65, 0.65],
            "initial_xy": [-0.290, -0.010],
            "initial_yaw": 0.0,
            "target_xy": [-0.10, 0.035],
            "target_yaw": 0.0,
            "arm": "left",
            "grasp_yaw_offset": 90.0,
        },
    }

    MONITOR_SPEC = {
        "source": "097_screen",
        "model_id": 3,
        "scale": [0.85, 0.85, 0.85],
        "initial_xy": [0.000, 0.190],
        "initial_yaw": 0.0,
    }

    ORDERED_STAGES = (
        "move_cup_left",
        "place_keyboard_front",
        "place_mouse_near_keyboard",
        "stack_notebook_on_book",
    )

    def _run_move(self, stage_name, *actions):
        success = self.move(*actions) and self.plan_success
        if not success:
            print(f"[arrange_scene_000] planning failed at {stage_name}")
        return success

    def take_dense_action(self, control_seq, save_freq=-1):
        """Use the normal physics controller without kinematic attachment."""
        return Base_Task.take_dense_action(self, control_seq, save_freq=save_freq)

    def load_actors(self):
        all_specs = list(self.OBJECT_SPECS.values()) + [self.MONITOR_SPEC]
        self.object_geometry_by_spec = {
            id(spec): self._asset_geometry(spec) for spec in all_specs
        }
        self.object_geometry = {
            name: self.object_geometry_by_spec[id(spec)]
            for name, spec in self.OBJECT_SPECS.items()
        }
        self.monitor_geometry = self.object_geometry_by_spec[id(self.MONITOR_SPEC)]

        table_z = self._table_z()
        book_height = self.object_geometry["book"]["height"]

        self.initial_bottom_z = {
            name: table_z + self.SURFACE_GAP for name in self.OBJECT_SPECS
        }
        self.initial_bottom_z["book"] = table_z
        self.initial_bottom_z["cup"] = (
            table_z + book_height + self.STACK_GAP
        )

        self.target_bottom_z = {
            name: table_z + self.SURFACE_GAP for name in self.OBJECT_SPECS
        }
        self.target_bottom_z["book"] = table_z
        self.target_bottom_z["notebook"] = (
            table_z + book_height + self.STACK_GAP
        )

        self.objects = {}
        for name, spec in self.OBJECT_SPECS.items():
            pose = self._rigid_pose(
                spec,
                spec["initial_xy"],
                spec["initial_yaw"],
                self.initial_bottom_z[name],
            )
            actor = create_actor(
                self,
                pose=pose,
                modelname=spec["source"],
                model_id=spec["model_id"],
                scale_multiplier=spec["scale"],
                convex=True,
                is_static=name == "book",
            )
            if actor is None:
                raise RuntimeError(
                    f"Failed to load {spec['source']}/base{spec['model_id']}"
                )
            actor.set_name(f"scene000_{name}")
            self.objects[name] = actor

        monitor_pose = self._rigid_pose(
            self.MONITOR_SPEC,
            self.MONITOR_SPEC["initial_xy"],
            self.MONITOR_SPEC["initial_yaw"],
            table_z,
        )
        self.monitor = create_actor(
            self,
            pose=monitor_pose,
            modelname=self.MONITOR_SPEC["source"],
            model_id=self.MONITOR_SPEC["model_id"],
            scale_multiplier=self.MONITOR_SPEC["scale"],
            convex=True,
            is_static=True,
        )
        if self.monitor is None:
            raise RuntimeError("Failed to load 097_screen/base3")
        self.monitor.set_name("scene000_monitor")

        # Dynamic actors settle onto their supports before the base stability
        # checker records the initial state.
        self.delay(1, save_freq=None)

    @staticmethod
    def _pose_with_world_yaw(base_quat, yaw_degrees):
        yaw = np.deg2rad(float(yaw_degrees))
        yaw_quat = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
        return t3d.quaternions.qmult(yaw_quat, base_quat).tolist()

    def _physical_top_down_pose(self, xy, contact_z, yaw_degrees, clearance):
        quat = self._pose_with_world_yaw(
            GRASP_DIRECTION_DIC["top_down"], yaw_degrees
        )
        rotation = t3d.quaternions.quat2mat(quat)
        contact = np.asarray([xy[0], xy[1], contact_z], dtype=np.float64)
        position = contact + rotation @ np.asarray(
            [-(0.12 + clearance), 0.0, 0.0], dtype=np.float64
        )
        return position.tolist() + quat

    def _gripper_link_names(self, arm_name):
        joints = (
            self.robot.left_gripper
            if arm_name == "left"
            else self.robot.right_gripper
        )
        return {joint[0].child_link.get_name() for joint in joints}

    def _finger_contacts(self, name, arm_name):
        actor_name = self.objects[name].actor.get_name()
        expected_links = self._gripper_link_names(arm_name)
        touching_links = set()
        for contact in self.scene.get_contacts():
            first = contact.bodies[0].entity.get_name()
            second = contact.bodies[1].entity.get_name()
            if first == actor_name and second in expected_links:
                touching_links.add(second)
            elif second == actor_name and first in expected_links:
                touching_links.add(first)
        return touching_links, expected_links

    def _grasp_contact_z(self, name):
        bottom = self._bottom_z(name)
        height = self.object_geometry[name]["height"]
        support_top = (
            self._bottom_z("book") + self.object_geometry["book"]["height"]
            if name == "cup"
            else self._table_z()
        )
        # ALOHA's finger meshes extend about 36 mm below the requested TCP
        # contact point.  This height lets them straddle thin objects without
        # entering the table/book collision mesh.
        return max(support_top + 0.040, bottom + height - 0.018)

    def _target_ee_pose(self, name, arm_name):
        ee_matrix = self._pose_matrix(self._current_ee_pose(arm_name))
        actor_matrix = self.objects[name].get_pose().to_transformation_matrix()
        # Preserve the physically held orientation.  Correcting a small
        # contact-induced tilt by rotating the tool can twist an object out of
        # the fingers; the scene starts these planar objects already aligned.
        target_actor_matrix = actor_matrix.copy()
        target_actor_matrix[:3, 3] = self._target_pose(name).p
        target_ee_matrix = target_actor_matrix @ np.linalg.inv(actor_matrix) @ ee_matrix
        return self._matrix_pose(target_ee_matrix)

    def _pick_and_place(self, name, stage_name):
        spec = self.OBJECT_SPECS[name]
        arm_name = spec["arm"]
        arm = self._arm_tag(arm_name)
        actor = self.objects[name]
        source_xy = actor.get_pose().p[:2]
        grasp_yaw = spec["initial_yaw"] + spec["grasp_yaw_offset"]
        contact_z = self._grasp_contact_z(name)
        source_pre = self._physical_top_down_pose(
            source_xy, contact_z, grasp_yaw, clearance=0.075
        )
        source_grasp = self._physical_top_down_pose(
            source_xy, contact_z, grasp_yaw, clearance=0.0
        )

        if not self._run_move(f"{stage_name}/open", self.open_gripper(arm)):
            return False
        if not self._run_move(
            f"{stage_name}/approach", self.move_to_pose(arm, source_pre)
        ):
            return False
        if not self._run_move(
            f"{stage_name}/descend", self.move_to_pose(arm, source_grasp)
        ):
            return False
        before_lift_z = float(actor.get_pose().p[2])
        if not self._run_move(
            f"{stage_name}/grasp", self.close_gripper(arm, pos=0.0)
        ):
            return False

        touching_links, expected_links = self._finger_contacts(name, arm_name)
        two_finger_contact = touching_links == expected_links
        if not two_finger_contact:
            self.plan_success = False
            print(
                f"[arrange_scene_000] rejected {stage_name}: "
                f"finger contacts={sorted(touching_links)}, "
                f"expected={sorted(expected_links)}"
            )
            return False

        if not self._run_move(
            f"{stage_name}/lift", self.move_to_pose(arm, source_pre)
        ):
            return False

        after_lift_z = float(actor.get_pose().p[2])
        touching_after_lift, _ = self._finger_contacts(name, arm_name)
        lifted = after_lift_z >= before_lift_z + 0.045
        retained = touching_after_lift == expected_links
        self.physical_grasp_log[name] = {
            "two_finger_contact": bool(two_finger_contact),
            "retained_after_lift": bool(retained),
            "lift_delta": round(after_lift_z - before_lift_z, 5),
        }
        if not (lifted and retained):
            self.plan_success = False
            print(
                f"[arrange_scene_000] rejected {stage_name}: "
                f"lift_delta={after_lift_z - before_lift_z:.4f}, "
                f"contacts_after_lift={sorted(touching_after_lift)}"
            )
            return False

        target_ee = self._target_ee_pose(name, arm_name)
        target_pre = target_ee.p.tolist() + np.asarray(target_ee.q).tolist()
        target_pre[2] += 0.055
        target_place = target_ee.p.tolist() + np.asarray(target_ee.q).tolist()
        print(
            f"[arrange_scene_000] {stage_name} physical transfer: "
            f"pre={np.round(target_pre, 4).tolist()}, "
            f"place={np.round(target_place, 4).tolist()}"
        )
        transfer_start = np.asarray(
            self._current_ee_pose(arm_name), dtype=np.float64
        )
        transfer_end = np.asarray(target_pre, dtype=np.float64)
        transfer_distance = np.linalg.norm(transfer_end[:3] - transfer_start[:3])
        waypoint_count = max(1, int(np.ceil(transfer_distance / 0.045)))
        for waypoint_index in range(1, waypoint_count + 1):
            alpha = waypoint_index / waypoint_count
            waypoint = transfer_start.copy()
            waypoint[:3] = (
                (1.0 - alpha) * transfer_start[:3]
                + alpha * transfer_end[:3]
            )
            if not self._run_move(
                f"{stage_name}/transfer_{waypoint_index}",
                self.move_to_pose(arm, waypoint.tolist()),
            ):
                return False
            touching_during_transfer, _ = self._finger_contacts(name, arm_name)
            if touching_during_transfer != expected_links:
                self.plan_success = False
                self.physical_grasp_log[name]["retained_during_transfer"] = False
                print(
                    f"[arrange_scene_000] rejected {stage_name}: object slipped "
                    f"at transfer waypoint {waypoint_index}/{waypoint_count}; "
                    f"contacts={sorted(touching_during_transfer)}"
                )
                return False
        self.physical_grasp_log[name]["retained_during_transfer"] = True
        if not self._run_move(
            f"{stage_name}/place", self.move_to_pose(arm, target_place)
        ):
            return False
        if not self._run_move(
            f"{stage_name}/release", self.open_gripper(arm)
        ):
            return False
        self.delay(1)
        retreat_pose = np.asarray(self._current_ee_pose(arm_name), dtype=np.float64)
        retreat_pose[2] += 0.060
        if not self._run_move(
            f"{stage_name}/retreat", self.move_to_pose(arm, retreat_pose.tolist())
        ):
            return False

        if not self._near_target(name, xy_tolerance=0.030, z_tolerance=0.025):
            self.plan_success = False
            print(
                f"[arrange_scene_000] rejected {stage_name}: object missed target; "
                f"actual={np.round(actor.get_pose().p, 4).tolist()}, "
                f"expected={np.round(self._target_pose(name).p, 4).tolist()}"
            )
            return False

        self.completed_stages.append(stage_name)
        return True

    def play_once(self):
        self.physical_grasp_log = {}
        self.info["info"] = {
            "{A}": "097_screen/base3",
            "{B}": "021_cup/base3",
            "{C}": "043_book/base1",
            "{D}": "116_keyboard/base3",
            "{E}": "047_mouse/base0",
            "{F}": "092_notebook/base2",
        }

        ordered_objects = (
            ("cup", "move_cup_left"),
            ("keyboard", "place_keyboard_front"),
            ("mouse", "place_mouse_near_keyboard"),
            ("notebook", "stack_notebook_on_book"),
        )
        for name, stage_name in ordered_objects:
            if not self._pick_and_place(name, stage_name):
                return self.info

        if not self._run_move(
            "return_to_origin_left",
            self.back_to_origin(self._arm_tag("left")),
        ):
            return self.info
        if not self._run_move(
            "return_to_origin_right",
            self.back_to_origin(self._arm_tag("right")),
        ):
            return self.info
        self.delay(2)
        return self.info

    @staticmethod
    def _xy_separated(
        pose_a,
        footprint_a,
        pose_b,
        footprint_b,
        gap=0.008,
    ):
        delta = np.abs(np.asarray(pose_a[:2]) - np.asarray(pose_b[:2]))
        required = (
            np.asarray(footprint_a) / 2.0
            + np.asarray(footprint_b) / 2.0
            + gap
        )
        return bool(delta[0] >= required[0] or delta[1] >= required[1])

    def _bottom_z(self, name):
        spec = self.OBJECT_SPECS[name]
        geometry = self.object_geometry[name]
        return float(
            self.objects[name].get_pose().p[2] + geometry["bottom_offset"]
        )

    def _no_target_interpenetration(self):
        cup = self.objects["cup"].get_pose().p
        book = self.objects["book"].get_pose().p
        keyboard = self.objects["keyboard"].get_pose().p
        mouse = self.objects["mouse"].get_pose().p
        monitor = self.monitor.get_pose().p

        cup_clear_of_book = self._xy_separated(
            cup,
            self.object_geometry["cup"]["footprint"],
            book,
            self.object_geometry["book"]["footprint"],
        )
        mouse_clear_of_keyboard = self._xy_separated(
            mouse,
            self.object_geometry["mouse"]["footprint"],
            keyboard,
            self.object_geometry["keyboard"]["footprint"],
        )
        keyboard_clear_of_monitor = self._xy_separated(
            keyboard,
            self.object_geometry["keyboard"]["footprint"],
            monitor,
            self.monitor_geometry["footprint"],
            gap=0.015,
        )

        book_top = self._bottom_z("book") + self.object_geometry["book"]["height"]
        notebook_gap = self._bottom_z("notebook") - book_top
        notebook_above_book = -0.001 <= notebook_gap <= self.STACK_GAP + 0.010
        return {
            "cup_clear_of_book": cup_clear_of_book,
            "mouse_clear_of_keyboard": mouse_clear_of_keyboard,
            "keyboard_clear_of_monitor": keyboard_clear_of_monitor,
            "notebook_above_book_without_penetration": notebook_above_book,
            "notebook_gap": round(float(notebook_gap), 5),
        }

    def _all_objects_reachable(self):
        x_low, x_high = self.WORKSPACE_X_LIMIT
        y_low, y_high = self.WORKSPACE_Y_LIMIT
        for actor in self.objects.values():
            x, y = actor.get_pose().p[:2]
            if not (x_low <= x <= x_high and y_low <= y <= y_high):
                return False
        monitor_x, monitor_y = self.MONITOR_SPEC["initial_xy"]
        return x_low <= monitor_x <= x_high and y_low <= monitor_y <= y_high

    def check_success(self):
        required_targets = ("cup", "keyboard", "mouse", "notebook", "book")
        objects_at_targets = {
            name: self._near_target(name, xy_tolerance=0.030, z_tolerance=0.025)
            for name in required_targets
        }
        stages_in_order = self.completed_stages == list(self.ORDERED_STAGES)

        keyboard_xy = self.objects["keyboard"].get_pose().p[:2]
        mouse_xy = self.objects["mouse"].get_pose().p[:2]
        monitor_xy = self.monitor.get_pose().p[:2]
        keyboard_in_front = bool(
            keyboard_xy[1] < monitor_xy[1] - 0.12
            and abs(float(keyboard_xy[0] - monitor_xy[0])) <= 0.035
        )
        mouse_near_keyboard = bool(
            np.linalg.norm(mouse_xy - keyboard_xy) <= 0.17
        )
        stack_xy_aligned = bool(
            np.linalg.norm(
                self.objects["notebook"].get_pose().p[:2]
                - self.objects["book"].get_pose().p[:2]
            )
            <= 0.014
        )
        clearances = self._no_target_interpenetration()
        all_reachable = self._all_objects_reachable()
        grippers_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        physical_grasps = {
            name: self.physical_grasp_log.get(name, {})
            for name in ("cup", "keyboard", "mouse", "notebook")
        }
        all_physical_grasps = all(
            log.get("two_finger_contact")
            and log.get("retained_after_lift")
            and log.get("retained_during_transfer")
            and log.get("lift_delta", 0.0) >= 0.045
            for log in physical_grasps.values()
        )

        result = {
            "objects_at_targets": objects_at_targets,
            "stages_in_order": stages_in_order,
            "keyboard_in_front_of_monitor": keyboard_in_front,
            "mouse_near_keyboard": mouse_near_keyboard,
            "book_notebook_xy_aligned": stack_xy_aligned,
            "clearances": clearances,
            "all_reachable": all_reachable,
            "grippers_open": grippers_open,
            "physical_grasps": physical_grasps,
        }
        print("[arrange_scene_000] success_check", result)
        return bool(
            all(objects_at_targets.values())
            and stages_in_order
            and keyboard_in_front
            and mouse_near_keyboard
            and stack_xy_aligned
            and all(
                value
                for key, value in clearances.items()
                if key != "notebook_gap"
            )
            and all_reachable
            and grippers_open
            and all_physical_grasps
        )
