#!/usr/bin/env python3
"""Check the TableSim Docker, renderer, task loader, and optional full scene."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("SAPIEN_RENDERER", "opengl")
os.environ.setdefault("SAPIEN_NO_DISPLAY", "1")

import sapien
import torch
import warp

import collect_data
from test_render import Sapien_TEST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="stack_plate", help="envs task name")
    parser.add_argument(
        "--config", default="stack_plate", help="task_config YAML name"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="also initialize robot, cameras, objects, and motion planners",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SAPIEN: {sapien.__version__}")
    print(f"Warp: {warp.__version__}")
    print(f"warp.torch available: {hasattr(warp, 'torch')}")

    if not args.setup:
        Sapien_TEST()

    def check_task(task, task_args):
        if not args.setup:
            print(
                "Task pipeline OK:",
                type(task).__name__,
                task_args["left_robot_file"],
            )
            return

        task_args["need_plan"] = True
        task_args["save_data"] = False
        task.setup_demo(now_ep_num=0, seed=0, **task_args)
        try:
            print(
                "Full simulation setup OK:",
                type(task).__name__,
                f"actors={len(task.scene.get_all_actors())}",
            )
        finally:
            task.close_env(clear_cache=True)

    collect_data.run = check_task
    collect_data.main(args.task, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
