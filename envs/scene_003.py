"""Deterministic SAPIEN reconstruction of ``scene_003.png``.

The layout preserves the visible composition of the reference image:

* a monitor at the rear centre;
* a keyboard in front of it, with a mouse on the rear-right;
* a phone and stapler at the front-right and front-left respectively;
* a folder, book, and cup stack on the left side of the desk.

Run ``python -m envs.scene_003`` to initialize the full TableSim environment
and save the head-camera render to ``envs/scene_003_simulation.png``.
"""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import transforms3d as t3d

from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .clean_the_desk_009 import clean_the_desk_009
from .utils import ArmTag, create_actor


class scene_003(clean_the_desk_009):
    """Build scene 003 and organize its electronic objects by desk region."""

    SURFACE_GAP = 0.004
    STACK_GAP = 0.006
    REGION_X = {
        "left": (-0.36, -0.12),
        "middle": (-0.12, 0.12),
        "right": (0.12, 0.36),
    }
    TARGETS = {
        "cup": [0.155, 0.080],
        "phone": [0.000, -0.180],
        "keyboard": [0.100, 0.070],
        "mouse": [0.075, 0.015],
    }

    OBJECT_SPECS = {
        "folder": {
            "source": "122_folder",
            "model_id": 0,
            "scale": [0.72, 0.72, 0.72],
            "initial_xy": [-0.155, 0.025],
            "initial_yaw": -8.0,
        },
        "book": {
            "source": "043_book",
            "model_id": 1,
            "scale": [0.64, 0.64, 0.64],
            "initial_xy": [-0.065, 0.005],
            "initial_yaw": -10.0,
        },
        "cup": {
            "source": "021_cup",
            "model_id": 3,
            "scale": [0.82, 0.82, 0.82],
            "initial_xy": [-0.065, 0.005],
            "initial_yaw": 20.0,
            "target_xy": TARGETS["cup"],
            "target_yaw": 20.0,
            "arm": "left",
        },
        "keyboard": {
            "source": "116_keyboard",
            "model_id": 3,
            "scale": [0.72, 0.72, 0.72],
            "initial_xy": [0.100, -0.080],
            "initial_yaw": -12.0,
            "target_xy": TARGETS["keyboard"],
            "target_yaw": -12.0,
            "arm": "right",
        },
        "mouse": {
            "source": "047_mouse",
            "model_id": 0,
            "scale": [0.82, 0.82, 0.82],
            "initial_xy": [0.265, 0.045],
            "initial_yaw": -8.0,
            "target_xy": TARGETS["mouse"],
            "target_yaw": -8.0,
            "arm": "right",
        },
        "phone": {
            "source": "077_phone",
            "model_id": 0,
            "scale": [0.78, 0.78, 0.78],
            "initial_xy": [0.235, -0.245],
            "initial_yaw": -8.0,
            "target_xy": TARGETS["phone"],
            "target_yaw": -8.0,
            "arm": "right",
        },
        "stapler": {
            "source": "048_stapler",
            "model_id": 4,
            "scale": [0.78, 0.78, 0.78],
            "initial_xy": [-0.305, -0.185],
            "initial_yaw": -70.0,
        },
    }

    MONITOR_SPEC = {
        "source": "097_screen",
        "model_id": 3,
        "scale": [0.84, 0.84, 0.84],
        "initial_xy": [0.025, 0.205],
        "initial_yaw": 0.0,
    }

    def setup_demo(self, **kwargs):
        settings = deepcopy(kwargs)
        randomization = dict(settings.get("domain_randomization") or {})
        randomization.update(
            {
                "cluttered_table": False,
                "random_background": False,
                "random_head_camera_dis": 0,
                "random_table_height": 0,
                "random_light": False,
            }
        )
        settings["domain_randomization"] = randomization
        super().setup_demo(**settings)

    def load_actors(self):
        specs = list(self.OBJECT_SPECS.values()) + [self.MONITOR_SPEC]
        self.object_geometry_by_spec = {
            id(spec): self._asset_geometry(spec) for spec in specs
        }
        geometry = self.object_geometry_by_spec
        self.object_geometry = {
            name: geometry[id(spec)] for name, spec in self.OBJECT_SPECS.items()
        }
        table_z = self._table_z()

        bottom_z = {
            name: table_z + self.SURFACE_GAP for name in self.OBJECT_SPECS
        }
        bottom_z["folder"] = table_z
        bottom_z["book"] = (
            table_z + geometry[id(self.OBJECT_SPECS["folder"])]["height"]
            + self.STACK_GAP
        )
        bottom_z["cup"] = (
            bottom_z["book"]
            + geometry[id(self.OBJECT_SPECS["book"])]["height"]
            + self.STACK_GAP
        )
        self.initial_bottom_z = dict(bottom_z)
        self.target_bottom_z = {
            name: table_z + self.SURFACE_GAP for name in self.OBJECT_SPECS
        }

        self.objects = {}
        for name, spec in self.OBJECT_SPECS.items():
            pose = self._rigid_pose(
                spec,
                spec["initial_xy"],
                spec["initial_yaw"],
                bottom_z[name],
            )
            actor = create_actor(
                self,
                pose=pose,
                modelname=spec["source"],
                model_id=spec["model_id"],
                scale_multiplier=spec["scale"],
                convex=True,
                # The folder is the fixed support surface.  The user permits
                # the book parameters to change so it can be manipulated by
                # the gripper; keep every other asset parameter unchanged.
                is_static=name == "folder",
            )
            if actor is None:
                raise RuntimeError(
                    f"Failed to load {spec['source']}/base{spec['model_id']}"
                )
            actor.set_name(f"scene003_{name}")
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
        self.monitor.set_name("scene003_monitor")

        # Let the movable actors settle before Base_Task checks stability.
        self.delay(1, save_freq=None)

    @classmethod
    def region_for_x(cls, x):
        for region, (lower, upper) in cls.REGION_X.items():
            if lower <= float(x) <= upper:
                return region
        return "outside"

    def _top_down_push_pose(self, xy, contact_z, clearance=0.0):
        quat = GRASP_DIRECTION_DIC["top_down"]
        rotation = t3d.quaternions.quat2mat(quat)
        contact = np.asarray([xy[0], xy[1], contact_z], dtype=np.float64)
        position = contact + rotation @ np.asarray(
            [-(0.12 + clearance), 0.0, 0.0], dtype=np.float64
        )
        return position.tolist() + list(quat)

    def _push_keyboard_to_upper_middle(self):
        keyboard = self.objects["keyboard"]
        pose = keyboard.get_pose().p
        geometry = self.object_geometry["keyboard"]
        half_depth = float(geometry["footprint"][1]) / 2.0
        contact_z = self._table_z() + max(0.018, geometry["height"] * 0.55)
        start_xy = [float(pose[0]), float(pose[1] - half_depth - 0.012)]
        end_xy = [self.TARGETS["keyboard"][0], self.TARGETS["keyboard"][1] - half_depth - 0.012]
        arm = ArmTag("right")

        if not self._run_move(
            "push_keyboard/approach",
            self.move_to_pose(
                arm,
                self._top_down_push_pose(start_xy, contact_z, clearance=0.070),
            ),
        ):
            return False
        if not self._run_move(
            "push_keyboard/close", self.close_gripper(arm, pos=0.0)
        ):
            return False
        if not self._run_move(
            "push_keyboard/contact",
            self.move_to_pose(arm, self._top_down_push_pose(start_xy, contact_z)),
        ):
            return False

        ee_matrix = self._pose_matrix(self._current_ee_pose("right"))
        actor_matrix = keyboard.get_pose().to_transformation_matrix()
        self._attached_actor = keyboard
        self._attached_arm = "right"
        self._ee_to_actor = np.linalg.inv(ee_matrix) @ actor_matrix
        push_waypoints = np.linspace(start_xy, end_xy, num=6)[1:]
        push_succeeded = True
        for index, waypoint in enumerate(push_waypoints, start=1):
            if not self._run_move(
                f"push_keyboard/push_{index}",
                self.move_to_pose(
                    arm, self._top_down_push_pose(waypoint, contact_z)
                ),
            ):
                push_succeeded = False
                break
        self._attached_actor = None
        self._attached_arm = None
        self._ee_to_actor = None
        if not push_succeeded:
            return False
        if not self._run_move(
            "push_keyboard/retreat",
            self.move_to_pose(
                arm,
                self._top_down_push_pose(end_xy, contact_z, clearance=0.080),
            ),
        ):
            return False
        if not self._run_move(
            "push_keyboard/open", self.open_gripper(arm, pos=1.0)
        ):
            return False
        self.completed_stages.append("push_keyboard_to_upper_middle")
        return True

    def _return_arm_to_origin(self, arm_name, completed_stage):
        arm = ArmTag(arm_name)
        if not self._run_move(
            f"{completed_stage}/return_to_origin",
            self.back_to_origin(arm_tag=arm),
        ):
            return False
        self.completed_stages.append(f"{completed_stage}_arm_reset")
        return True

    def play_once(self):
        self.completed_stages = []
        self.info["info"] = {
            "{A}": "021_cup/base3",
            "{B}": "077_phone/base0",
            "{C}": "116_keyboard/base3",
            "{D}": "047_mouse/base0",
            "{a}": "left arm",
            "{b}": "right arm",
        }
        if not self._pick_and_place("cup", "place_cup_in_upper_right"):
            return self.info
        if not self._return_arm_to_origin("left", "place_cup_in_upper_right"):
            return self.info
        if not self._pick_and_place("phone", "place_phone_in_middle"):
            return self.info
        if not self._return_arm_to_origin("right", "place_phone_in_middle"):
            return self.info
        if not self._push_keyboard_to_upper_middle():
            return self.info
        if not self._return_arm_to_origin("right", "push_keyboard_to_upper_middle"):
            return self.info
        if not self._pick_and_place("mouse", "place_mouse_near_keyboard"):
            return self.info
        if not self._return_arm_to_origin("right", "place_mouse_near_keyboard"):
            return self.info
        return self.info

    def check_success(self):
        cup = self.objects["cup"].get_pose().p
        phone = self.objects["phone"].get_pose().p
        keyboard = self.objects["keyboard"].get_pose().p
        mouse = self.objects["mouse"].get_pose().p

        cup_ok = cup[0] >= 0.100 and cup[1] >= 0.070
        phone_ok = self.region_for_x(phone[0]) == "middle"
        keyboard_ok = (
            self.region_for_x(keyboard[0]) == "middle" and keyboard[1] >= 0.030
        )
        mouse_ok = abs(float(mouse[0])) <= 0.180
        mouse_near_keyboard = np.linalg.norm(mouse[:2] - keyboard[:2]) <= 0.180
        grippers_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        self.eval_success = bool(
            cup_ok
            and phone_ok
            and keyboard_ok
            and mouse_ok
            and mouse_near_keyboard
            and grippers_open
        )
        if not self.eval_success:
            print(
                "[scene_003] goal check:",
                f"cup={cup[:2].round(3).tolist()} right_upper={cup_ok};",
                f"phone={phone[:2].round(3).tolist()} middle={phone_ok};",
                f"keyboard={keyboard[:2].round(3).tolist()} upper_middle={keyboard_ok};",
                f"mouse={mouse[:2].round(3).tolist()} middle={mouse_ok}",
                f"near_keyboard={mouse_near_keyboard};",
                f"grippers_open={grippers_open}",
            )
        return self.eval_success

    def get_instruction(self):
        return (
            "Move the cup to the upper-right, move the phone to the middle, "
            "push the keyboard toward the monitor, and place the mouse near it."
        )


def render_scene(output_path: Path, config_name: str = "scene_003") -> Path:
    """Initialize the task and save one deterministic head-camera RGB frame."""

    project_root = Path(__file__).resolve().parents[1]
    script_dir = project_root / "script"
    sys.path.insert(0, str(script_dir))
    os.chdir(project_root)
    os.environ.setdefault("SAPIEN_RENDERER", "opengl")
    os.environ.setdefault("SAPIEN_NO_DISPLAY", "1")

    import collect_data

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save_initialized_scene(task, task_args):
        task_args["need_plan"] = False
        task_args["save_data"] = False
        task_args["render_freq"] = 0
        task_args["camera"]["collect_wrist_camera"] = False
        task_args.pop("seed", None)
        task.setup_demo(now_ep_num=0, seed=0, **task_args)
        try:
            task.scene.update_render()
            for camera in task.cameras.static_camera_list:
                camera.take_picture()
            rgb = task.cameras.get_rgb()["head_camera"]["rgb"]
            imageio.imwrite(output_path, rgb)
            print(f"Saved scene_003 simulation image: {output_path}")
        finally:
            task.close_env(clear_cache=True)

    collect_data.run = save_initialized_scene
    collect_data.main("scene_003", config_name)
    return output_path


def main() -> int:
    default_output = Path(__file__).with_name("scene_003_simulation.png")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--config", default="scene_003")
    args = parser.parse_args()
    render_scene(args.output, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
