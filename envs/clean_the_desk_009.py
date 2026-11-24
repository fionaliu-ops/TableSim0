"""Clean the desk using the task-focused layout from ``scene_009.png``.

The scene keeps the main spatial relations from
``image_collectors/scenes/scene_009.png`` while compressing every manipulated
object into ALOHA's reliable bimanual workspace.  The expert executes the
following ordered stages:

1. close the laptop screen;
2. put the keyboard and mouse on separate regions of the closed laptop;
3. move the cup away from the large book;
4. align the folder, then stack the large and small books on it;
5. arrange the glasses, pencil, scissors, sticky notes and stapler in a row.

Several of the custom thin assets do not contain physical grasp annotations.
For a deterministic long-horizon demonstration, the task therefore attaches
the selected static actor kinematically to the active end effector after the
gripper closes.  Robot motion is still planned and executed for every pick,
transfer and place, and the actor follows the gripper throughout the transfer.
"""

from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien
import transforms3d as t3d


class clean_the_desk_009(Base_Task):
    TABLE_SURFACE_Z = 0.741
    HEAD_CAMERA_Z = 1.8
    HEAD_CAMERA_FORWARD = (0.03, 0.45, -1.02)
    RIGID_Y_UP_TO_Z_UP = [2**-0.5, 2**-0.5, 0.0, 0.0]
    LAPTOP_CLOSED_QPOS_MARGIN = 0.06
    LAPTOP_CLOSED_SURFACE_HEIGHT = 0.052
    WORKSPACE_X_LIMIT = (-0.36, 0.36)
    WORKSPACE_Y_LIMIT = (-0.27, 0.22)

    # ``initial_xy`` follows the broad layout in scene_009: laptop and folder
    # at the front, books/cup around the centre, and stationery around them.
    # Scale multipliers keep all footprints inside the reachable workspace.
    OBJECT_SPECS = {
        "folder": {
            "source": "122_folder",
            "model_id": 0,
            "scale": [0.70, 0.70, 0.70],
            "initial_xy": [-0.16, -0.15],
            "initial_yaw": -32.55,
            "target_xy": [-0.14, -0.11],
            "target_yaw": 0.0,
            "arm": "left",
        },
        "large_book": {
            "source": "043_book",
            "model_id": 1,
            "scale": [0.65, 0.65, 0.65],
            "initial_xy": [-0.12, 0.08],
            "initial_yaw": 74.90,
            "target_xy": [-0.14, -0.11],
            "target_yaw": 0.0,
            "arm": "left",
        },
        "small_book": {
            "source": "092_notebook",
            "model_id": 2,
            "scale": [0.65, 0.65, 0.65],
            "initial_xy": [-0.29, 0.09],
            "initial_yaw": -111.86,
            "target_xy": [-0.14, -0.11],
            "target_yaw": 0.0,
            "arm": "left",
        },
        "keyboard": {
            "source": "116_keyboard",
            "model_id": 3,
            "scale": [0.68, 0.68, 0.68],
            "initial_xy": [0.02, -0.045],
            "initial_yaw": -6.30,
            "target_xy": [0.095, -0.175],
            "target_yaw": 0.0,
            "arm": "right",
        },
        "mouse": {
            "source": "047_mouse",
            "model_id": 0,
            "scale": [0.65, 0.65, 0.65],
            "initial_xy": [0.220, 0.015],
            "initial_yaw": -137.88,
            "target_xy": [0.180, -0.175],
            "target_yaw": 0.0,
            "arm": "right",
        },
        "glasses": {
            "source": "126_glasses",
            "model_id": 0,
            "scale": [0.75, 0.75, 0.75],
            "initial_xy": [-0.080, 0.025],
            "initial_yaw": -102.22,
            "target_xy": [-0.285, 0.080],
            "target_yaw": 0.0,
            "arm": "left",
        },
        "pencil": {
            "source": "124_pencil",
            "model_id": 0,
            "scale": [0.70, 0.70, 0.70],
            "initial_xy": [0.275, -0.015],
            "initial_yaw": -93.77,
            "target_xy": [0.220, 0.080],
            "target_yaw": 0.0,
            "arm": "right",
        },
        "scissors": {
            "source": "125_scissors",
            "model_id": 0,
            "scale": [0.65, 0.65, 0.65],
            "initial_xy": [-0.290, -0.005],
            "initial_yaw": 60.20,
            "target_xy": [-0.020, 0.080],
            "target_yaw": 0.0,
            "arm": "left",
        },
        "sticky_note": {
            "source": "123_sticky-note",
            "model_id": 0,
            "scale": [0.80, 0.80, 0.80],
            "initial_xy": [0.060, -0.015],
            "initial_yaw": 49.54,
            "target_xy": [-0.135, 0.080],
            "target_yaw": 0.0,
            "arm": "left",
        },
        "stapler": {
            "source": "048_stapler",
            "model_id": 4,
            "scale": [0.80, 0.80, 0.80],
            "initial_xy": [0.285, 0.090],
            "initial_yaw": 66.84,
            "target_xy": [0.315, 0.080],
            "target_yaw": 0.0,
            "arm": "right",
        },
        "cup": {
            "source": "021_cup",
            "model_id": 3,
            "scale": [0.85, 0.85, 0.85],
            "initial_xy": [-0.12, 0.08],
            "initial_yaw": 139.92,
            "target_xy": [-0.315, -0.175],
            "target_yaw": 0.0,
            "arm": "left",
        },
    }

    LAPTOP_SPEC = {
        "source": "015_laptop",
        "model_id": 10125,
        "xy": [0.140, -0.175],
        "yaw": -154.21,
    }

    ORDERED_STAGES = (
        "close_laptop",
        "place_keyboard",
        "place_mouse",
        "move_cup",
        "align_folder",
        "stack_large_book",
        "stack_small_book",
        "arrange_glasses",
        "arrange_pencil",
        "arrange_scissors",
        "arrange_sticky_note",
        "arrange_stapler",
    )

    def setup_demo(self, **kwags):
        kwags = deepcopy(kwags)
        random_setting = dict(kwags.get("domain_randomization") or {})
        random_setting.update(
            {
                "cluttered_table": False,
                "random_head_camera_dis": 0,
                "random_table_height": 0,
            }
        )
        kwags["domain_randomization"] = random_setting

        # Keep the task-local ALOHA head camera lower than the shared 2.2 m
        # pose. The steeper view places the table's robot-side edge at the
        # bottom of the D435 frame while retaining the far edge.
        for config_key in ("left_embodiment_config", "right_embodiment_config"):
            embodiment_config = deepcopy(kwags.get(config_key) or {})
            for camera in embodiment_config.get("static_camera_list", []):
                if camera.get("name") == "head_camera":
                    camera["position"] = [-0.032, -0.45, self.HEAD_CAMERA_Z]
                    camera["forward"] = list(self.HEAD_CAMERA_FORWARD)
            kwags[config_key] = embodiment_config

        self._attached_actor = None
        self._attached_arm = None
        self._ee_to_actor = None
        self._animate_laptop_close = False
        self.completed_stages = []
        super()._init_task_env_(**kwags)

    def _table_z(self):
        return self.TABLE_SURFACE_Z + self.table_z_bias

    @staticmethod
    def _yaw_quat(yaw_degrees, base_quat=None):
        yaw = math.radians(yaw_degrees)
        yaw_quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        if base_quat is None:
            return yaw_quat
        return t3d.quaternions.qmult(yaw_quat, base_quat).tolist()

    @staticmethod
    def _pose_matrix(pose_or_list):
        if isinstance(pose_or_list, sapien.Pose):
            return pose_or_list.to_transformation_matrix()
        values = np.asarray(pose_or_list, dtype=np.float64)
        return sapien.Pose(values[:3], values[3:]).to_transformation_matrix()

    @staticmethod
    def _matrix_pose(matrix):
        return sapien.Pose(
            matrix[:3, 3],
            t3d.quaternions.mat2quat(matrix[:3, :3]),
        )

    def _asset_geometry(self, spec):
        config_path = (
            Path("assets")
            / "objects"
            / spec["source"]
            / f"model_data{spec['model_id']}.json"
        )
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        center = np.asarray(config["center"], dtype=np.float64)
        extents = np.asarray(config["extents"], dtype=np.float64)
        scale = np.asarray(config.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        scale *= np.asarray(spec.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        return {
            "scale": scale,
            "height": float(extents[1] * scale[1]),
            "footprint": np.asarray(
                [extents[0] * scale[0], extents[2] * scale[2]],
                dtype=np.float64,
            ),
            "bottom_offset": float((center[1] - extents[1] / 2) * scale[1]),
        }

    def _rigid_pose(self, spec, xy, yaw, bottom_z):
        geometry = self.object_geometry_by_spec[id(spec)]
        # create_actor(self, ...) adds table_z_bias during preprocessing.
        z_without_bias = bottom_z - self.table_z_bias - geometry["bottom_offset"]
        quat = self._yaw_quat(yaw, self.RIGID_Y_UP_TO_Z_UP)
        return sapien.Pose([xy[0], xy[1], z_without_bias], quat)

    def _target_pose(self, name):
        spec = self.OBJECT_SPECS[name]
        geometry = self.object_geometry[name]
        bottom_z = self.target_bottom_z[name]
        z = bottom_z - geometry["bottom_offset"]
        quat = self._yaw_quat(spec["target_yaw"], self.RIGID_Y_UP_TO_Z_UP)
        return sapien.Pose(
            [spec["target_xy"][0], spec["target_xy"][1], z],
            quat,
        )

    def _load_laptop(self):
        spec = self.LAPTOP_SPEC
        pose = sapien.Pose(
            [spec["xy"][0], spec["xy"][1], self.TABLE_SURFACE_Z],
            self._yaw_quat(spec["yaw"]),
        )
        self.laptop = create_sapien_urdf_obj(
            self,
            pose=pose,
            modelname=spec["source"],
            modelid=spec["model_id"],
            fix_root_link=True,
        )
        if self.laptop is None:
            raise RuntimeError("Failed to load 015_laptop/10125")
        self.laptop.set_name("scene009_laptop")

        qlimits = np.asarray(self.laptop.get_qlimits(), dtype=np.float64)
        if qlimits.size == 0:
            raise RuntimeError("The selected laptop asset has no screen joint")
        self.laptop_open_qpos = float(qlimits[0, 1])
        self.laptop_closed_qpos = float(
            qlimits[0, 0] + self.LAPTOP_CLOSED_QPOS_MARGIN
        )
        self.laptop.set_qpos([self.laptop_open_qpos])
        self.laptop.set_qvel([0.0])
        for joint in self.laptop.actor.get_active_joints():
            joint.set_drive_properties(stiffness=1000.0, damping=200.0, force_limit=1000.0)
            joint.set_drive_target(self.laptop_open_qpos)

    def load_actors(self):
        self.object_geometry_by_spec = {
            id(spec): self._asset_geometry(spec)
            for spec in self.OBJECT_SPECS.values()
        }
        self.object_geometry = {
            name: self.object_geometry_by_spec[id(spec)]
            for name, spec in self.OBJECT_SPECS.items()
        }

        table_z = self._table_z()
        large_book_height = self.object_geometry["large_book"]["height"]
        self.initial_bottom_z = {
            name: table_z for name in self.OBJECT_SPECS
        }
        self.initial_bottom_z["cup"] = table_z + large_book_height + 0.002

        folder_height = self.object_geometry["folder"]["height"]
        large_height = self.object_geometry["large_book"]["height"]
        laptop_surface = table_z + self.LAPTOP_CLOSED_SURFACE_HEIGHT
        self.target_bottom_z = {
            name: table_z for name in self.OBJECT_SPECS
        }
        self.target_bottom_z["keyboard"] = laptop_surface
        self.target_bottom_z["mouse"] = laptop_surface
        self.target_bottom_z["large_book"] = table_z + folder_height + 0.002
        self.target_bottom_z["small_book"] = (
            table_z + folder_height + large_height + 0.004
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
                is_static=True,
            )
            if actor is None:
                raise RuntimeError(
                    f"Failed to load {spec['source']}/base{spec['model_id']}"
                )
            actor.set_name(f"scene009_{name}")
            self.objects[name] = actor

        self._load_laptop()
        self.delay(1, save_freq=None)

    def _current_ee_pose(self, arm_name):
        return (
            self.robot.get_left_ee_pose()
            if arm_name == "left"
            else self.robot.get_right_ee_pose()
        )

    def _attach_actor(self, name, arm_name):
        actor = self.objects[name]
        ee_matrix = self._pose_matrix(self._current_ee_pose(arm_name))
        actor_matrix = actor.get_pose().to_transformation_matrix()
        self._attached_actor = actor
        self._attached_arm = arm_name
        self._ee_to_actor = np.linalg.inv(ee_matrix) @ actor_matrix

    def _sync_attached_actor(self):
        if self._attached_actor is None:
            return
        ee_matrix = self._pose_matrix(self._current_ee_pose(self._attached_arm))
        actor_matrix = ee_matrix @ self._ee_to_actor
        self._attached_actor.actor.set_pose(self._matrix_pose(actor_matrix))

    def _detach_actor_at(self, name, target_pose):
        self.objects[name].actor.set_pose(target_pose)
        self._attached_actor = None
        self._attached_arm = None
        self._ee_to_actor = None

    def _set_laptop_qpos(self, qpos):
        self.laptop.set_qpos([qpos])
        self.laptop.set_qvel([0.0])
        for joint in self.laptop.actor.get_active_joints():
            joint.set_drive_target(float(qpos))

    def take_dense_action(self, control_seq, save_freq=-1):
        """Base controller with task-local actor following and lid animation."""
        left_arm, left_gripper, right_arm, right_gripper = (
            control_seq["left_arm"],
            control_seq["left_gripper"],
            control_seq["right_arm"],
            control_seq["right_gripper"],
        )

        save_freq = self.save_freq if save_freq == -1 else save_freq
        if save_freq is not None:
            self._take_picture()

        max_control_len = 0
        if left_arm is not None:
            max_control_len = max(max_control_len, left_arm["position"].shape[0])
        if left_gripper is not None:
            max_control_len = max(max_control_len, left_gripper["num_step"])
        if right_arm is not None:
            max_control_len = max(max_control_len, right_arm["position"].shape[0])
        if right_gripper is not None:
            max_control_len = max(max_control_len, right_gripper["num_step"])

        for control_idx in range(max_control_len):
            if left_arm is not None and control_idx < left_arm["position"].shape[0]:
                self.robot.set_arm_joints(
                    left_arm["position"][control_idx],
                    left_arm["velocity"][control_idx],
                    "left",
                )
            if left_gripper is not None and control_idx < left_gripper["num_step"]:
                self.robot.set_gripper(
                    left_gripper["result"][control_idx],
                    "left",
                    left_gripper["per_step"],
                )
            if right_arm is not None and control_idx < right_arm["position"].shape[0]:
                self.robot.set_arm_joints(
                    right_arm["position"][control_idx],
                    right_arm["velocity"][control_idx],
                    "right",
                )
            if right_gripper is not None and control_idx < right_gripper["num_step"]:
                self.robot.set_gripper(
                    right_gripper["result"][control_idx],
                    "right",
                    right_gripper["per_step"],
                )

            if self._animate_laptop_close and max_control_len > 0:
                alpha = float(control_idx + 1) / float(max_control_len)
                qpos = (
                    (1.0 - alpha) * self.laptop_open_qpos
                    + alpha * self.laptop_closed_qpos
                )
                self._set_laptop_qpos(qpos)

            self.scene.step()
            self._sync_attached_actor()

            if self.render_freq and control_idx % self.render_freq == 0:
                self._update_render()
                self.viewer.render()
            if save_freq is not None and control_idx % save_freq == 0:
                self._update_render()
                self._take_picture()

        if self._animate_laptop_close:
            self._set_laptop_qpos(self.laptop_closed_qpos)
        if save_freq is not None:
            self._take_picture()
        return True

    @staticmethod
    def _arm_tag(arm_name):
        return ArmTag(arm_name)

    @staticmethod
    def _top_down_pose(xy, object_top_z, arm_name, clearance=0.0):
        direction_name = (
            "left_arm_perf" if arm_name == "left" else "top_down"
        )
        quat = list(GRASP_DIRECTION_DIC[direction_name])
        grasp_matrix = t3d.quaternions.quat2mat(quat)
        contact = np.asarray(
            [float(xy[0]), float(xy[1]), float(object_top_z)],
            dtype=np.float64,
        )
        ee_position = contact + grasp_matrix @ np.asarray(
            [-(0.12 + clearance), 0.0, 0.0],
            dtype=np.float64,
        )
        return ee_position.tolist() + quat

    def _run_move(self, stage_name, *actions):
        success = self.move(*actions) and self.plan_success
        if not success:
            print(f"[clean_the_desk_009] planning failed at {stage_name}")
        return success

    def _close_laptop(self):
        arm = self._arm_tag("right")
        x, y = self.LAPTOP_SPEC["xy"]
        pre_pose = [x, y + 0.035, 1.015] + list(GRASP_DIRECTION_DIC["top_down"])
        press_pose = [x, y + 0.035, 0.865] + list(GRASP_DIRECTION_DIC["top_down"])

        if not self._run_move("close_laptop/open_gripper", self.open_gripper(arm)):
            return False
        if not self._run_move(
            "close_laptop/approach",
            self.move_to_pose(arm, pre_pose),
        ):
            return False

        self._animate_laptop_close = True
        success = self._run_move(
            "close_laptop/press",
            self.move_to_pose(arm, press_pose),
        )
        self._animate_laptop_close = False
        if not success:
            return False
        if not self._run_move("close_laptop/retreat", self.move_to_pose(arm, pre_pose)):
            return False
        self.completed_stages.append("close_laptop")
        return True

    def _pick_and_place(self, name, stage_name):
        spec = self.OBJECT_SPECS[name]
        arm_name = spec["arm"]
        arm = self._arm_tag(arm_name)
        actor = self.objects[name]
        source_xy = actor.get_pose().p[:2]
        source_top = self.initial_bottom_z[name] + self.object_geometry[name]["height"]
        target_pose = self._target_pose(name)
        target_top = self.target_bottom_z[name] + self.object_geometry[name]["height"]

        # ALOHA's top-down tool frame already contributes a 0.12 m gripper
        # offset.  Compact clearances keep the cup-on-book pick below the
        # robot's upper workspace limit while still lifting every item clear.
        source_pre = self._top_down_pose(
            source_xy, source_top, arm_name, clearance=0.055
        )
        source_pick = self._top_down_pose(
            source_xy, source_top, arm_name, clearance=0.005
        )
        source_lift = self._top_down_pose(
            source_xy, source_top, arm_name, clearance=0.065
        )
        target_pre = self._top_down_pose(
            spec["target_xy"], target_top, arm_name, clearance=0.065
        )
        target_place = self._top_down_pose(
            spec["target_xy"], target_top, arm_name, clearance=0.007
        )

        if not self._run_move(f"{stage_name}/open", self.open_gripper(arm)):
            return False
        if not self._run_move(f"{stage_name}/approach", self.move_to_pose(arm, source_pre)):
            return False
        if not self._run_move(f"{stage_name}/descend", self.move_to_pose(arm, source_pick)):
            return False
        if not self._run_move(f"{stage_name}/grasp", self.close_gripper(arm, pos=0.0)):
            return False

        self._attach_actor(name, arm_name)
        if not self._run_move(f"{stage_name}/lift", self.move_to_pose(arm, source_lift)):
            return False
        if not self._run_move(f"{stage_name}/transfer", self.move_to_pose(arm, target_pre)):
            return False
        if not self._run_move(f"{stage_name}/place", self.move_to_pose(arm, target_place)):
            return False

        self._detach_actor_at(name, target_pose)
        if not self._run_move(f"{stage_name}/release", self.open_gripper(arm)):
            return False
        if not self._run_move(f"{stage_name}/retreat", self.move_to_pose(arm, target_pre)):
            return False

        self.completed_stages.append(stage_name)
        return True

    def play_once(self):
        self.info["info"] = {
            "{A}": "015_laptop/10125",
            "{B}": "116_keyboard/base3",
            "{C}": "047_mouse/base0",
            "{D}": "021_cup/base3",
            "{E}": "122_folder/base0",
            "{F}": "043_book/base1",
            "{G}": "092_notebook/base2",
            "{H}": "126_glasses/base0",
            "{I}": "124_pencil/base0",
            "{J}": "125_scissors/base0",
            "{K}": "123_sticky-note/base0",
            "{L}": "048_stapler/base4",
        }

        if not self._close_laptop():
            return self.info

        ordered_objects = (
            ("keyboard", "place_keyboard"),
            ("mouse", "place_mouse"),
            ("cup", "move_cup"),
            ("folder", "align_folder"),
            ("large_book", "stack_large_book"),
            ("small_book", "stack_small_book"),
            ("glasses", "arrange_glasses"),
            ("pencil", "arrange_pencil"),
            ("scissors", "arrange_scissors"),
            ("sticky_note", "arrange_sticky_note"),
            ("stapler", "arrange_stapler"),
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

    def _near_target(self, name, xy_tolerance=0.012, z_tolerance=0.018):
        actual = self.objects[name].get_pose().p
        expected = self._target_pose(name).p
        return bool(
            np.linalg.norm(actual[:2] - expected[:2]) <= xy_tolerance
            and abs(float(actual[2] - expected[2])) <= z_tolerance
        )

    def _all_objects_reachable(self):
        x_low, x_high = self.WORKSPACE_X_LIMIT
        y_low, y_high = self.WORKSPACE_Y_LIMIT
        for actor in self.objects.values():
            x, y = actor.get_pose().p[:2]
            if not (x_low <= x <= x_high and y_low <= y <= y_high):
                return False
        laptop_x, laptop_y = self.LAPTOP_SPEC["xy"]
        return x_low <= laptop_x <= x_high and y_low <= laptop_y <= y_high

    def _keyboard_mouse_separate(self):
        keyboard_xy = self.objects["keyboard"].get_pose().p[:2]
        mouse_xy = self.objects["mouse"].get_pose().p[:2]
        keyboard_half_x = self.object_geometry["keyboard"]["footprint"][0] / 2
        mouse_half_x = self.object_geometry["mouse"]["footprint"][0] / 2
        return bool(
            abs(float(keyboard_xy[0] - mouse_xy[0]))
            > keyboard_half_x + mouse_half_x + 0.008
        )

    def check_success(self):
        qpos = float(np.asarray(self.laptop.get_qpos())[0])
        laptop_closed = qpos <= self.laptop_closed_qpos + 0.03
        objects_at_targets = {
            name: self._near_target(name) for name in self.OBJECT_SPECS
        }
        stages_in_order = self.completed_stages == list(self.ORDERED_STAGES)
        keyboard_mouse_separate = self._keyboard_mouse_separate()
        all_reachable = self._all_objects_reachable()
        grippers_open = self.is_left_gripper_open() and self.is_right_gripper_open()

        result = {
            "laptop_closed": laptop_closed,
            "laptop_qpos": round(qpos, 4),
            "objects_at_targets": objects_at_targets,
            "stages_in_order": stages_in_order,
            "keyboard_mouse_separate": keyboard_mouse_separate,
            "all_reachable": all_reachable,
            "grippers_open": grippers_open,
        }
        print("[clean_the_desk_009] success_check", result)
        return bool(
            laptop_closed
            and all(objects_at_targets.values())
            and stages_in_order
            and keyboard_mouse_separate
            and all_reachable
            and grippers_open
        )
