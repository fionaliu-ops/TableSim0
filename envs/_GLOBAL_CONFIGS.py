"""Project-wide paths and motion-planning constants.

All paths are derived from this file so TableSim can be mounted at any host
location while still running at ``/workspace/TableSim`` in Docker.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Keep these values as strings with a trailing separator for compatibility
# with existing RoboTwin-style scripts that concatenate paths directly.
ROOT_PATH = str(PROJECT_ROOT) + os.sep
ASSETS_PATH = str(PROJECT_ROOT / "assets") + os.sep
EMBODIMENTS_PATH = str(PROJECT_ROOT / "assets" / "embodiments") + os.sep
TEXTURES_PATH = str(PROJECT_ROOT / "assets" / "background_texture") + os.sep
CONFIGS_PATH = str(PROJECT_ROOT / "task_config") + os.sep
SCRIPT_PATH = str(PROJECT_ROOT / "script") + os.sep
DESCRIPTION_PATH = str(PROJECT_ROOT / "description") + os.sep


GRASP_DIRECTION_DIC = {
    "left": [0, 0, 0, -1],
    "front_left": [-0.383, 0, 0, -0.924],
    "front": [-0.707, 0, 0, -0.707],
    "front_right": [-0.924, 0, 0, -0.383],
    "right": [-1, 0, 0, 0],
    "top_down": [-0.5, 0.5, -0.5, -0.5],
    "down_right": [-0.707, 0, -0.707, 0],
    "down_left": [0, 0.707, 0, -0.707],
    "top_down_little_left": [-0.353523, 0.61239, -0.353524, -0.61239],
    "top_down_little_right": [-0.61239, 0.353523, -0.61239, -0.353524],
    "left_arm_perf": [-0.853532, 0.146484, -0.353542, -0.3536],
    "right_arm_perf": [-0.353518, 0.353564, -0.14642, -0.853568],
}

WORLD_DIRECTION_DIC = {
    "left": [0, -0.707, 0, 0.707],
    "front": [0.5, -0.5, 0.5, 0.5],
    "right": [0.707, 0, 0.707, 0],
    "top_down": [0, 0.707, -0.707, 0],
}

ROTATE_NUM = 10
