#!/usr/bin/env python3
"""Validate the procedural Messy Desk asset batch.

Examples:

    python script/check_messy_desk_assets.py
    python script/check_messy_desk_assets.py --sapien
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
OBJECTS_DIR = ROOT / "assets" / "objects"
ASSET_NAMES = [
    "121_paper",
    "122_folder",
    "123_sticky-note",
    "124_pencil",
    "125_scissors",
    "126_glasses",
    "127_lamp",
    "128_paper-clip",
    "129_sticker",
    "130_headphone",
    "131_printer",
    "132_tablet",
    "133_calendar",
    "134_rubber-band",
    "135_magazine",
]
REQUIRED_MODEL_KEYS = {
    "center",
    "extents",
    "scale",
    "transform_matrix",
    "target_pose",
    "contact_points_pose",
    "functional_matrix",
    "orientation_point",
    "contact_points_group",
    "contact_points_mask",
    "stable",
}


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    if not isinstance(loaded, trimesh.Scene) or not loaded.geometry:
        raise ValueError(f"{path} does not contain a non-empty GLB scene")
    return trimesh.util.concatenate(tuple(loaded.dump()))


def validate_files() -> None:
    for name in ASSET_NAMES:
        directory = OBJECTS_DIR / name
        visual_path = directory / "visual" / "base0.glb"
        collision_path = directory / "collision" / "base0.glb"
        config_path = directory / "model_data0.json"
        points_path = directory / "points_info.json"
        for path in (visual_path, collision_path, config_path, points_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"missing or empty asset file: {path}")

        visual = load_mesh(visual_path)
        collision = load_mesh(collision_path)
        if not np.isfinite(visual.vertices).all() or not np.isfinite(collision.vertices).all():
            raise ValueError(f"{name} contains non-finite mesh vertices")
        if not collision.is_watertight:
            raise ValueError(f"{name} collision mesh is not watertight")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        missing_keys = REQUIRED_MODEL_KEYS - config.keys()
        if missing_keys:
            raise ValueError(f"{name} config is missing keys: {sorted(missing_keys)}")
        configured_extents = np.asarray(config["extents"], dtype=float)
        if configured_extents.shape != (3,) or np.any(configured_extents <= 0):
            raise ValueError(f"{name} has invalid configured extents")
        if not np.allclose(configured_extents, collision.extents, atol=1e-6):
            raise ValueError(
                f"{name} configured extents {configured_extents} "
                f"do not match collision extents {collision.extents}"
            )
        points = json.loads(points_path.read_text(encoding="utf-8"))
        if set(points) != {"contact_points", "functional_points"}:
            raise ValueError(f"{name} has invalid points_info.json keys")
        print(
            f"[files] {name}: visual={len(visual.faces)} faces, "
            f"collision={len(collision.faces)} faces"
        )


def validate_sapien() -> None:
    import sapien

    from envs.utils.create_actor import create_actor

    scene = sapien.Scene()
    scene.set_timestep(1 / 250)
    scene.add_ground(0)
    actors = []
    # This is the same Y-up GLB to Z-up world rotation used by
    # rand_create_cluttered_actor for numbered assets.
    object_rotation = [2**-0.5, 2**-0.5, 0.0, 0.0]
    for index, name in enumerate(ASSET_NAMES):
        wrapper = create_actor(
            scene,
            sapien.Pose([index * 0.5, 0.0, 0.03], object_rotation),
            name,
            convex=True,
            model_id=0,
        )
        if wrapper is None:
            raise RuntimeError(f"SAPIEN failed to create {name}")
        actors.append(wrapper)
        print(f"[sapien] loaded {name}")

    for _ in range(250):
        scene.step()
    for wrapper in actors:
        pose = wrapper.get_pose()
        if not np.isfinite(np.r_[pose.p, pose.q]).all():
            raise RuntimeError(f"{wrapper.get_name()} has a non-finite pose")
        if pose.p[2] < -0.1:
            raise RuntimeError(f"{wrapper.get_name()} fell through the ground: z={pose.p[2]}")
    print(f"[sapien] settled {len(actors)} dynamic actors successfully")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sapien",
        action="store_true",
        help="Also create and step all assets as dynamic SAPIEN actors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_files()
    if args.sapien:
        validate_sapien()
    print(f"PASS: validated {len(ASSET_NAMES)} Messy Desk assets")


if __name__ == "__main__":
    main()
