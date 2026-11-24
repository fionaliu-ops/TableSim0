#!/usr/bin/env python3
"""Render existing Messy Desk assets with the TableSim SAPIEN renderer.

Each source asset receives a directory under ``image_collectors/objects``.
Every model variant is rendered in multiple yaw poses. The resulting RGB
images therefore match the models and materials seen by the simulator.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = ROOT / "assets" / "objects"
OUTPUT_ROOT = ROOT / "image_collectors" / "objects"
MAP_PATH = ROOT / "image_collectors" / "object_asset_map.json"
OBJECTS_JSON = ROOT / "image_collectors" / "messy_desk.json"
MISSING_MD = ROOT / "image_collectors" / "messy_desk_missing_objects.md"


@dataclass(frozen=True)
class Variant:
    display_id: str
    loader_id: str | int


def load_specs() -> list[dict]:
    specs = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    required = {"output", "source", "kind", "objects"}
    outputs = set()
    for spec in specs:
        missing = required - spec.keys()
        if missing:
            raise ValueError(f"asset map entry is missing {sorted(missing)}: {spec}")
        if spec["output"] in outputs:
            raise ValueError(f"duplicate output directory: {spec['output']}")
        if spec["kind"] not in {"rigid", "sapien_urdf", "objaverse_urdf"}:
            raise ValueError(f"unsupported asset kind: {spec['kind']}")
        outputs.add(spec["output"])
    return specs


def validate_mapping(specs: list[dict]) -> None:
    objects = set(json.loads(OBJECTS_JSON.read_text(encoding="utf-8"))["objects"])
    markdown = MISSING_MD.read_text(encoding="utf-8")
    missing_match = re.search(
        r"## 仍明确缺失的 \d+ 类物品(?P<body>.*?)"
        r"## 只有近似资产的 \d+ 类物品",
        markdown,
        re.DOTALL,
    )
    approximate_match = re.search(
        r"## 只有近似资产的 \d+ 类物品(?P<body>.*?)"
        r"## 已确认存在的主要别名",
        markdown,
        re.DOTALL,
    )
    if missing_match is None or approximate_match is None:
        raise ValueError("could not parse missing/approximate sections from Markdown")
    missing_section = missing_match.group("body")
    approximate_section = approximate_match.group("body")
    excluded = set(re.findall(r"^- `([^`]+)`", missing_section, re.MULTILINE))
    excluded.update(
        re.findall(r"^\| `([^`]+)` \|", approximate_section, re.MULTILINE)
    )
    expected = objects - excluded
    mapped = {name for spec in specs for name in spec["objects"]}
    if mapped != expected:
        raise ValueError(
            "object-to-asset map does not match the current JSON classification; "
            f"unmapped={sorted(expected - mapped)}, unexpected={sorted(mapped - expected)}"
        )


def natural_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def discover_variants(spec: dict) -> list[Variant]:
    kind = spec["kind"]
    source = spec["source"]
    if kind == "rigid":
        visual_dir = ASSETS_ROOT / source / "visual"
        collision_dir = ASSETS_ROOT / source / "collision"
        variants = []
        for visual in sorted(visual_dir.glob("base*.glb"), key=lambda p: natural_key(p.name)):
            match = re.fullmatch(r"base(\d+)\.glb", visual.name)
            if not match:
                continue
            model_id = match.group(1)
            if not (collision_dir / f"base{model_id}.glb").is_file():
                continue
            if not (ASSETS_ROOT / source / f"model_data{model_id}.json").is_file():
                continue
            variants.append(Variant(model_id, int(model_id)))
        return variants

    if kind == "sapien_urdf":
        model_dirs = [
            directory
            for directory in (ASSETS_ROOT / source).iterdir()
            if directory.is_dir()
            and directory.name != "visual"
            and (directory / "mobility.urdf").is_file()
        ]
        model_dirs.sort(key=lambda p: natural_key(p.name))
        return [
            Variant(directory.name, index)
            for index, directory in enumerate(model_dirs)
        ]

    objaverse_list = json.loads(
        (ASSETS_ROOT / "objaverse" / "list.json").read_text(encoding="utf-8")
    )
    return [
        Variant(str(model_id), str(model_id))
        for model_id in objaverse_list["list_of_items"].get(source, [])
        if (ASSETS_ROOT / "objaverse" / source / str(model_id) / "model.urdf").is_file()
    ]


def add_scene_lighting(scene) -> None:
    scene.set_ambient_light([0.62, 0.62, 0.62])
    scene.add_directional_light([0.5, 0.35, -1.0], [1.35, 1.30, 1.25], shadow=True)
    scene.add_directional_light([-0.4, -0.2, -1.0], [0.45, 0.48, 0.55], shadow=False)


def load_variant(scene, spec: dict, variant: Variant):
    import sapien

    from envs.utils.create_actor import create_actor, create_sapien_urdf_obj

    if spec["kind"] == "rigid":
        rotation = [2**-0.5, 2**-0.5, 0.0, 0.0]
        pose_config = ASSETS_ROOT / spec["source"] / "scene_pose.json"
        if pose_config.is_file():
            rotation = json.loads(pose_config.read_text(encoding="utf-8"))["rotation_wxyz"]
            rotation = np.asarray(rotation, dtype=float)
            if (
                rotation.shape != (4,)
                or not np.isfinite(rotation).all()
                or not np.isclose(np.linalg.norm(rotation), 1.0)
            ):
                raise ValueError(f"invalid scene rotation in {pose_config}")
            rotation = rotation.tolist()
        return create_actor(
            scene,
            sapien.Pose([0.0, 0.0, 0.0], rotation),
            spec["source"],
            model_id=int(variant.loader_id),
            convex=True,
            is_static=True,
        )
    if spec["kind"] == "sapien_urdf":
        return create_sapien_urdf_obj(
            scene,
            sapien.Pose(),
            spec["source"],
            modelid=int(variant.loader_id),
            fix_root_link=True,
        )

    from envs.utils.rand_create_cluttered_actor import create_cluttered_urdf_obj

    return create_cluttered_urdf_obj(
        scene,
        sapien.Pose(),
        f"objects/objaverse/{spec['source']}/{variant.loader_id}",
        fix_root_link=True,
    )


def component_bounds(component) -> np.ndarray | None:
    try:
        bounds = np.asarray(component.get_global_aabb_fast(), dtype=float)
    except Exception:
        try:
            bounds = np.asarray(component.compute_global_aabb_tight(), dtype=float)
        except Exception:
            return None
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        return None
    return bounds


def actor_bounds(wrapper) -> np.ndarray:
    import sapien

    actor = wrapper.actor
    bounds = []
    if isinstance(actor, sapien.Entity):
        for component in actor.get_components():
            value = component_bounds(component)
            if value is not None:
                bounds.append(value)
    else:
        for link in actor.get_links():
            render_bounds = []
            for component in link.entity.get_components():
                if "RenderBody" not in type(component).__name__:
                    continue
                value = component_bounds(component)
                if value is not None:
                    render_bounds.append(value)
            if render_bounds:
                bounds.extend(render_bounds)
            else:
                value = component_bounds(link)
                if value is not None:
                    bounds.append(value)
    if not bounds:
        raise RuntimeError("could not obtain render or collision bounds for actor")
    stacked = np.asarray(bounds)
    return np.asarray([stacked[:, 0].min(axis=0), stacked[:, 1].max(axis=0)])


def get_pose(wrapper):
    return wrapper.actor.get_pose()


def set_pose(wrapper, pose) -> None:
    wrapper.actor.set_pose(pose)


def center_actor_on_ground(scene, wrapper):
    import sapien

    scene.update_render()
    bounds = actor_bounds(wrapper)
    center = bounds.mean(axis=0)
    base_pose = get_pose(wrapper)
    shift = np.asarray([-center[0], -center[1], 0.003 - bounds[0, 2]])
    centered_pose = sapien.Pose(base_pose.p + shift, base_pose.q)
    set_pose(wrapper, centered_pose)
    scene.update_render()
    return centered_pose, actor_bounds(wrapper)


def camera_pose(position: np.ndarray, target: np.ndarray):
    import sapien

    forward = target - position
    forward /= np.linalg.norm(forward)
    left = np.cross(np.asarray([0.0, 0.0, 1.0]), forward)
    if np.linalg.norm(left) < 1e-8:
        left = np.asarray([0.0, 1.0, 0.0])
    else:
        left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([forward, left, up], axis=1)
    matrix[:3, 3] = position
    return sapien.Pose(matrix)


def render_variant(
    spec: dict,
    variant: Variant,
    output_dir: Path,
    poses: int,
    resolution: int,
) -> list[dict]:
    import sapien

    scene = sapien.Scene()
    scene.set_timestep(1 / 250)
    add_scene_lighting(scene)
    wrapper = load_variant(scene, spec, variant)
    if wrapper is None:
        raise RuntimeError("asset loader returned None")
    centered_pose, bounds = center_actor_on_ground(scene, wrapper)

    extents = bounds[1] - bounds[0]
    if np.any(extents <= 0) or not np.isfinite(extents).all():
        raise RuntimeError(f"invalid actor bounds: {bounds.tolist()}")
    center = bounds.mean(axis=0)
    center[2] = bounds[0, 2] + extents[2] * 0.48
    max_extent = float(max(extents))
    fovy = math.radians(48)
    distance = max(0.25, max_extent / (2 * math.tan(fovy / 2)) * 1.75)
    direction = np.asarray([0.95, -1.15, 0.78], dtype=float)
    direction /= np.linalg.norm(direction)

    scene.add_ground(
        0.0,
        render_material=[0.31, 0.33, 0.36],
    )
    camera = scene.add_camera(
        "object_camera",
        resolution,
        resolution,
        fovy,
        max(0.002, distance / 1000),
        max(10.0, distance * 12),
    )
    camera.entity.set_pose(camera_pose(center + direction * distance, center))

    records = []
    for pose_index in range(poses):
        yaw = 2 * math.pi * pose_index / poses
        yaw_pose = sapien.Pose(q=[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        set_pose(wrapper, yaw_pose * centered_pose)
        scene.update_render()
        camera.take_picture()
        rgba = camera.get_picture("Color")
        rgb = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
        filename = f"model_{variant.display_id}_pose_{pose_index:02d}.png"
        path = output_dir / filename
        Image.fromarray(rgb).save(path, optimize=True)
        if path.stat().st_size == 0 or float(rgb.std()) < 1.0:
            raise RuntimeError(f"rendered image appears blank: {path}")
        records.append(
            {
                "file": filename,
                "model_id": variant.display_id,
                "pose_id": pose_index,
                "yaw_degrees": round(math.degrees(yaw), 2),
                "resolution": [resolution, resolution],
                "camera_distance": round(distance, 6),
                "actor_extents": [round(float(value), 6) for value in extents],
            }
        )
    scene.clear()
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=int, default=3, help="Yaw poses per model variant.")
    parser.add_argument("--resolution", type=int, default=512, help="Square image resolution.")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Render only this output folder; repeat to select multiple folders.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately after the first failed model variant.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poses < 1 or args.resolution < 64:
        raise ValueError("--poses must be >= 1 and --resolution must be >= 64")
    specs = load_specs()
    validate_mapping(specs)
    if args.only:
        selected = set(args.only)
        unknown = selected - {spec["output"] for spec in specs}
        if unknown:
            raise ValueError(f"unknown --only outputs: {sorted(unknown)}")
        specs = [spec for spec in specs if spec["output"] in selected]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "renderer": "TableSim SAPIEN",
        "poses_per_variant": args.poses,
        "resolution": [args.resolution, args.resolution],
        "assets": [],
        "failures": [],
    }
    total_images = 0
    for spec_index, spec in enumerate(specs, start=1):
        variants = discover_variants(spec)
        if not variants:
            manifest["failures"].append(
                {"output": spec["output"], "model_id": None, "error": "no variants found"}
            )
            if args.fail_fast:
                break
            continue
        output_dir = OUTPUT_ROOT / spec["output"]
        output_dir.mkdir(parents=True, exist_ok=True)
        asset_record = {
            "output": spec["output"],
            "source": spec["source"],
            "kind": spec["kind"],
            "objects": spec["objects"],
            "variants": [],
        }
        print(
            f"[{spec_index:02d}/{len(specs):02d}] {spec['output']}: "
            f"{len(variants)} variant(s)",
            flush=True,
        )
        for variant in variants:
            try:
                images = render_variant(
                    spec,
                    variant,
                    output_dir,
                    args.poses,
                    args.resolution,
                )
                asset_record["variants"].append(
                    {"model_id": variant.display_id, "images": images}
                )
                total_images += len(images)
                print(
                    f"  PASS model={variant.display_id} images={len(images)}",
                    flush=True,
                )
            except Exception as error:
                failure = {
                    "output": spec["output"],
                    "model_id": variant.display_id,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                manifest["failures"].append(failure)
                print(
                    f"  FAIL model={variant.display_id}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    break
        manifest["assets"].append(asset_record)
        if args.fail_fast and manifest["failures"]:
            break

    manifest["summary"] = {
        "asset_directories": len(manifest["assets"]),
        "model_variants": sum(
            len(asset["variants"]) for asset in manifest["assets"]
        ),
        "images": total_images,
        "failures": len(manifest["failures"]),
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False), flush=True)
    if manifest["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
