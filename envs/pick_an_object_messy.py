"""Pick a lightweight object from a fixed messy-desk scene.

Scene source:
    image_collectors/scenes/manifest.json -> scene_009.png

The task keeps the manifest scene layout deterministic, then asks the left arm
to pick the mouse and drop it in a clear tabletop region. Non-target objects
are loaded as static clutter so the task remains focused on grasping one object
instead of recovering from random pile dynamics.
"""

from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *

import math
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien
import transforms3d as t3d


class pick_an_object_messy(Base_Task):
    TABLE_SURFACE_Z = 0.741
    RIGID_Y_UP_TO_Z_UP = [2**-0.5, 2**-0.5, 0.0, 0.0]
    TARGET_LABEL = "mouse"
    TARGET_ASSET = "047_mouse"
    TARGET_MODEL_ID = 2

    # This is scene_009 from image_collectors/scenes/manifest.json.
    SCENE_OBJECTS = [
        {
            "label": "notebook",
            "source": "092_notebook",
            "kind": "rigid",
            "model_id": "2",
            "xy": [0.11008, 0.13772],
            "z_offset": 0.0,
            "yaw_degrees": -111.86,
        },
        {
            "label": "pen",
            "source": "010_pen",
            "kind": "sapien_urdf",
            "model_id": "101685",
            "xy": [0.12031, 0.14308],
            "z_offset": 0.02633,
            "yaw_degrees": -169.54,
        },
        {
            "label": "mouse",
            "source": "047_mouse",
            "kind": "rigid",
            "model_id": "2",
            "xy": [-0.29208, -0.10648],
            "z_offset": 0.0,
            "yaw_degrees": -137.88,
        },
        {
            "label": "cup",
            "source": "021_cup",
            "kind": "rigid",
            "model_id": "3",
            "xy": [0.07588, 0.08259],
            "z_offset": 0.0,
            "yaw_degrees": 139.92,
        },
        {
            "label": "keyboard",
            "source": "116_keyboard",
            "kind": "rigid",
            "model_id": "3",
            "xy": [0.05464, -0.12235],
            "z_offset": 0.0,
            "yaw_degrees": -6.3,
        },
        {
            "label": "folder",
            "source": "122_folder",
            "kind": "rigid",
            "model_id": "0",
            "xy": [0.26594, -0.266],
            "z_offset": 0.0,
            "yaw_degrees": -32.55,
        },
        {
            "label": "glasses",
            "source": "126_glasses",
            "kind": "rigid",
            "model_id": "0",
            "xy": [0.1728, -0.05111],
            "z_offset": 0.0,
            "yaw_degrees": -102.22,
        },
        {
            "label": "stapler",
            "source": "048_stapler",
            "kind": "rigid",
            "model_id": "4",
            "xy": [0.40243, -0.06781],
            "z_offset": 0.0,
            "yaw_degrees": 66.84,
        },
        {
            "label": "phone",
            "source": "077_phone",
            "kind": "rigid",
            "model_id": "0",
            "xy": [-0.51494, -0.20859],
            "z_offset": 0.0,
            "yaw_degrees": -139.29,
        },
        {
            "label": "book",
            "source": "043_book",
            "kind": "rigid",
            "model_id": "0",
            "xy": [0.17681, 0.23878],
            "z_offset": 0.0,
            "yaw_degrees": 74.9,
        },
        {
            "label": "lamp",
            "source": "127_lamp",
            "kind": "rigid",
            "model_id": "0",
            "xy": [0.4268, 0.06848],
            "z_offset": 0.0,
            "yaw_degrees": -96.19,
        },
        {
            "label": "scissors",
            "source": "125_scissors",
            "kind": "rigid",
            "model_id": "0",
            "xy": [-0.26631, 0.02364],
            "z_offset": 0.0,
            "yaw_degrees": 60.2,
        },
        {
            "label": "pencil",
            "source": "124_pencil",
            "kind": "rigid",
            "model_id": "0",
            "xy": [0.47443, -0.22705],
            "z_offset": 0.0,
            "yaw_degrees": -93.77,
        },
        {
            "label": "laptop",
            "source": "015_laptop",
            "kind": "sapien_urdf",
            "model_id": "10125",
            "xy": [-0.18773, -0.23462],
            "z_offset": 0.0,
            "yaw_degrees": -154.21,
        },
        {
            "label": "sticky note",
            "source": "123_sticky-note",
            "kind": "rigid",
            "model_id": "0",
            "xy": [-0.00263, -0.22012],
            "z_offset": 0.0,
            "yaw_degrees": 49.54,
        },
        {
            "label": "marker",
            "source": "058_markpen",
            "kind": "rigid",
            "model_id": "4",
            "xy": [0.0581, 0.33019],
            "z_offset": 0.0,
            "yaw_degrees": 91.28,
        },
        {
            "label": "calendar",
            "source": "133_calendar",
            "kind": "rigid",
            "model_id": "0",
            "xy": [-0.46696, 0.06963],
            "z_offset": 0.0,
            "yaw_degrees": 1.92,
        },
    ]

    def setup_demo(self, **kwags):
        kwags = deepcopy(kwags)
        random_setting = dict(kwags.get("domain_randomization") or {})
        random_setting["cluttered_table"] = False
        kwags["domain_randomization"] = random_setting
        super()._init_task_env_(**kwags)

    def _table_z(self):
        return self.TABLE_SURFACE_Z + self.table_z_bias

    def _yaw_quat(self, yaw_degrees, base_quat=None):
        yaw = math.radians(yaw_degrees)
        yaw_quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        if base_quat is None:
            return yaw_quat
        return t3d.quaternions.qmult(yaw_quat, base_quat).tolist()

    def _rigid_bottom_offset(self, source, model_id):
        config_path = Path("assets") / "objects" / source / f"model_data{model_id}.json"
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        center = np.array(config["center"], dtype=np.float64)
        extents = np.array(config["extents"], dtype=np.float64)
        scale = np.array(config.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        return float((center[1] - extents[1] / 2) * scale[1])

    def _scene_pose(self, item):
        x, y = item["xy"]
        if item["kind"] == "rigid":
            bottom_offset = self._rigid_bottom_offset(item["source"], int(item["model_id"]))
            z = self.TABLE_SURFACE_Z + item["z_offset"] - bottom_offset
            quat = self._yaw_quat(item["yaw_degrees"], self.RIGID_Y_UP_TO_Z_UP)
        else:
            z = self.TABLE_SURFACE_Z + item["z_offset"]
            quat = self._yaw_quat(item["yaw_degrees"])
        return sapien.Pose([x, y, z], quat)

    def _load_scene_item(self, item, is_target=False):
        pose = self._scene_pose(item)
        if item["kind"] == "rigid":
            actor = create_actor(
                self,
                pose=pose,
                modelname=item["source"],
                model_id=int(item["model_id"]),
                convex=True,
                is_static=not is_target,
            )
        elif item["kind"] == "sapien_urdf":
            actor = create_sapien_urdf_obj(
                self,
                pose=pose,
                modelname=item["source"],
                modelid=int(item["model_id"]),
                fix_root_link=True,
            )
        else:
            raise ValueError(f"Unsupported scene object kind: {item['kind']}")

        if actor is None:
            raise RuntimeError(f"Failed to load scene object: {item}")
        actor.set_name("target_mouse" if is_target else f"clutter_{item['label']}_{item['source']}")
        return actor

    def _target_item(self):
        for item in self.SCENE_OBJECTS:
            if item["label"] == self.TARGET_LABEL and item["source"] == self.TARGET_ASSET:
                return item
        raise RuntimeError("Target mouse item is missing from SCENE_OBJECTS")

    def load_actors(self):
        self.scene_items = {}
        target_item = self._target_item()
        self.target_actor = None

        for item in self.SCENE_OBJECTS:
            is_target = item is target_item
            actor = self._load_scene_item(item, is_target=is_target)
            self.scene_items[item["label"]] = actor
            if is_target:
                self.target_actor = actor
            else:
                self.add_prohibit_area(actor, padding=0.01)

        self.initial_target_pose = self.target_actor.get_pose()
        target_data = self.target_actor.config
        scale = np.array(target_data.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        extents = np.array(target_data.get("extents", [0.08, 0.05, 0.13]), dtype=np.float64)
        # local Y is vertical for numbered GLB assets after the Y-up to Z-up rotation.
        self.target_height = float(extents[1] * scale[1])
        self.target_top_z = self._table_z() + self.target_height + 0.006
        # Let the static clutter and target mouse contact the tabletop before
        # planning the pick. The base class runs its own stability check after this.
        self.delay(1)

    def _top_down_grasp_actions(self, arm_tag, pre_grasp_dis=0.08):
        quat = GRASP_DIRECTION_DIC["top_down"]
        grasp_matrix = t3d.quaternions.quat2mat(quat)

        target_xy = np.array(self.initial_target_pose.p[:2], dtype=np.float64)
        contact_point = np.array(
            [
                target_xy[0],
                target_xy[1],
                self.target_top_z,
            ],
            dtype=np.float64,
        )
        grasp_position = contact_point + grasp_matrix @ np.array([-0.12, 0.0, 0.0])
        pre_grasp_position = grasp_position + grasp_matrix @ np.array([-pre_grasp_dis, 0.0, 0.0])

        pre_grasp_pose = pre_grasp_position.tolist() + quat
        grasp_pose = grasp_position.tolist() + quat
        return arm_tag, [
            Action(arm_tag, "move", target_pose=pre_grasp_pose),
            Action(
                arm_tag,
                "move",
                target_pose=grasp_pose,
                constraint_pose=[1, 1, 1, 0, 0, 0],
            ),
            Action(arm_tag, "close", target_gripper_pos=0.0),
        ]

    def _run_stage(self, stage_name, *actions):
        success = self.move(*actions) and self.plan_success
        if not success:
            print(f"[pick_an_object_messy] failed at {stage_name}")
        return success

    def play_once(self):
        arm_tag = ArmTag("left")
        self.info["info"] = {
            "{A}": f"{self.TARGET_ASSET}/base{self.TARGET_MODEL_ID}",
            "{a}": str(arm_tag),
        }

        if not self._run_stage("open_left_gripper", self.open_gripper(arm_tag, pos=1.0)):
            return self.info
        if not self._run_stage(
            "grasp_mouse",
            self.grasp_actor(
                self.target_actor,
                arm_tag=arm_tag,
                pre_grasp_dis=0.08,
                grasp_dis=0.0,
                gripper_pos=0.0,
                contact_point_id=[0, 1],
            ),
        ):
            return self.info
        if not self._run_stage("lift_mouse", self.move_by_displacement(arm_tag, z=0.12)):
            return self.info
        self.delay(2)
        return self.info

    def check_success(self):
        target_pose = self.target_actor.get_pose().p
        initial_pose = self.initial_target_pose.p
        xy_from_start = np.linalg.norm(target_pose[:2] - initial_pose[:2])
        lifted = target_pose[2] > initial_pose[2] + 0.045

        print(
            "[pick_an_object_messy] success_check",
            {
                "target": np.round(target_pose, 4).tolist(),
                "initial": np.round(initial_pose, 4).tolist(),
                "xy_from_start": round(float(xy_from_start), 4),
                "lifted": bool(lifted),
                "left_closed": self.is_left_gripper_close(),
            },
        )

        return lifted and self.is_left_gripper_close()
