#!/usr/bin/env python3
"""Generate random messy desk scenes from the current TableSim assets.

The file name keeps the user's requested spelling: scence_generator.
Scenes are sampled from image_collectors/messy_desk.json, constrained to
objects that are mapped in image_collectors/object_asset_map.json, then
rendered through SAPIEN so the saved images match simulator appearance.
Categories use relative weights without replacement; relationships do not
inject categories or reposition objects. Placement is static and collision-safe,
not a gravity-drop simulation. Image-space occlusion is currently not measured.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "image_collectors" / "scenes"
MESSY_JSON = ROOT / "image_collectors" / "messy_desk.json"
ASSET_MAP_JSON = ROOT / "image_collectors" / "object_asset_map.json"

TABLE_LENGTH = 1.35
TABLE_WIDTH = 0.86
TABLE_SURFACE_Z = 0.74
TABLE_XLIM = (-TABLE_LENGTH / 2 + 0.06, TABLE_LENGTH / 2 - 0.06)
TABLE_YLIM = (-TABLE_WIDTH / 2 + 0.06, TABLE_WIDTH / 2 - 0.06)

# Large room-level assets do exist in assets/objects, but they are not useful
# as tabletop clutter. These guards also catch oversized or mis-scaled variants.
MAX_FOOTPRINT_XY = 0.68
MAX_RADIUS = 0.38
MAX_HEIGHT = 0.45
MIN_RADIUS = 0.018
DENSE_SAMPLE_PROBABILITY = 0.84
CLUTTER_CENTER = np.asarray([0.0, -0.015], dtype=float)
CLUTTER_STD = np.asarray([0.24, 0.145], dtype=float)

# The camera looks from negative Y toward the back wall at positive Y.  At most
# one large desktop device is used as the visual anchor of a scene.  Its
# canonical yaw=0 faces negative Y in the current TableSim assets.
REAR_CENTER_DEVICE_LABELS = frozenset({"laptop", "monitor", "printer"})
FRONT_FACING_CLOCK_LABELS = frozenset({"clock", "alarm clock"})
REAR_CENTER_XLIM = (-0.14, 0.14)
REAR_CENTER_YLIM = (0.20, 0.30)
SAMPLING_POLICY = "natural_weighted_v1"
FRONT_CLEARANCE = 0.075
FRONT_SECTOR_CENTERS_X = (-0.34, 0.0, 0.34)
FRONT_SECTOR_STD_X = 0.085
PLACEMENT_CLEARANCE = 0.004
VERTICAL_COLLISION_TOLERANCE = 0.001
TABLE_PENETRATION_TOLERANCE = 0.002
MAX_SUPPORT_GAP = 0.025
MAX_FLAT_PEN_HEIGHT = 0.04
MAX_FLAT_PEN_HEIGHT_RATIO = 0.5

EXCLUDED_DESK_LABELS = {
    "bookshelf",
    "cabinet",
    "drawer",
    "trash can",
}


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_messy_desk_objects import (  # noqa: E402
    Variant,
    actor_bounds,
    add_scene_lighting,
    camera_pose,
    center_actor_on_ground,
    discover_variants,
    load_specs,
    load_variant,
    set_pose,
)


@dataclass
class ObjectChoice:
    label: str
    spec: dict[str, Any]
    variant: Variant


@dataclass
class LoadedObject:
    label: str
    asset_output: str
    source: str
    kind: str
    variant_id: str
    wrapper: Any
    centered_pose: Any
    footprint: np.ndarray
    height: float
    radius: float
    xy: np.ndarray | None = None
    z_offset: float = 0.0
    yaw: float = 0.0
    relation: dict[str, Any] | None = None
    world_bounds: np.ndarray | None = None


def load_messy_config() -> dict[str, Any]:
    return json.loads(MESSY_JSON.read_text(encoding="utf-8"))


def build_label_specs(specs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    label_specs: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        for label in spec["objects"]:
            label_specs.setdefault(label, []).append(spec)
    return label_specs


def normalize_weights(weights: list[float]) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    values[~np.isfinite(values)] = 0
    values = np.maximum(values, 0)
    if float(values.sum()) <= 0:
        return np.full(len(values), 1.0 / len(values), dtype=float)
    return values / float(values.sum())


def weighted_choice(rng: np.random.Generator, items: list[Any], weights: list[float]) -> Any:
    probs = normalize_weights(weights)
    index = int(rng.choice(len(items), p=probs))
    return items[index]


def weighted_sample_without_replacement(
    rng: np.random.Generator,
    items: list[Any],
    weights: list[float],
    count: int,
) -> list[Any]:
    remaining = list(items)
    remaining_weights = list(weights)
    sampled: list[Any] = []
    while remaining and len(sampled) < count:
        item = weighted_choice(rng, remaining, remaining_weights)
        index = remaining.index(item)
        sampled.append(item)
        remaining.pop(index)
        remaining_weights.pop(index)
    return sampled


def append_unique(labels: list[str], label: str) -> None:
    if label not in labels:
        labels.append(label)


def relation_key(relation: dict[str, Any]) -> tuple[str, str, str]:
    return (
        relation["subject"],
        relation["predicate"].lower(),
        relation["object"],
    )


def relation_is_available(relation: dict[str, Any], available: set[str]) -> bool:
    return relation["subject"] in available and relation["object"] in available


def pair_is_available(pair: dict[str, Any], available: set[str]) -> bool:
    return all(name in available for name in pair["objects"])


def sample_scene_labels(
    rng: np.random.Generator,
    messy: dict[str, Any],
    available_labels: set[str],
    min_objects: int,
    max_objects: int,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]], tuple[str, str] | None]:
    """Sample categories by relative weight, without injecting relation endpoints.

    Weights are not independent per-scene occurrence probabilities: category
    count, available assets, and the single-rear-device constraint also matter.
    The empty relation outputs retain compatibility with existing debug tools.
    """
    object_freq = messy["objects"]
    pool = [
        label
        for label in sorted(available_labels)
        if math.isfinite(float(object_freq.get(label, 0.0)))
        and float(object_freq.get(label, 0.0)) > 0
    ]
    capacity = sum(label not in REAR_CENTER_DEVICE_LABELS for label in pool)
    capacity += int(any(label in REAR_CENTER_DEVICE_LABELS for label in pool))
    if capacity < min_objects:
        raise ValueError(
            f"only {capacity} positive-weight categories available under the layout "
            f"constraints, fewer than --min-objects={min_objects}"
        )
    target_count = int(rng.integers(min_objects, min(max_objects, capacity) + 1))
    labels: list[str] = []
    while len(labels) < target_count:
        label = weighted_choice(rng, pool, [float(object_freq[name]) for name in pool])
        labels.append(label)
        pool.remove(label)
        # Preserve the existing desk-size constraint, without a second 70%
        # gate that overrides the configured category weights.
        if label in REAR_CENTER_DEVICE_LABELS:
            pool = [name for name in pool if name not in REAR_CENTER_DEVICE_LABELS]
    return labels, [], [], None


def choose_object(
    rng: np.random.Generator,
    label: str,
    label_specs: dict[str, list[dict[str, Any]]],
    variant_cache: dict[str, list[Variant]],
) -> ObjectChoice | None:
    specs = label_specs.get(label, [])
    specs = [spec for spec in specs if variant_cache.get(spec["output"])]
    if not specs:
        return None
    spec = specs[int(rng.integers(0, len(specs)))]
    variants = variant_cache[spec["output"]]
    variant = variants[int(rng.integers(0, len(variants)))]
    return ObjectChoice(label=label, spec=spec, variant=variant)

def pen_is_upright(bounds: np.ndarray) -> bool:
    extents = bounds[1] - bounds[0]
    return float(extents[2]) > max(
        MAX_FLAT_PEN_HEIGHT,
        MAX_FLAT_PEN_HEIGHT_RATIO * float(max(extents[:2])),
    )

def lay_pen_flat(scene: Any, wrapper: Any) -> None:
    """Correct upright pen variants before measuring their placement footprint."""
    import sapien
    scene.update_render()
    initial_bounds = actor_bounds(wrapper)
    if not pen_is_upright(initial_bounds):
        return
    initial_pose = wrapper.actor.get_pose()
    quarter_turn = math.sqrt(0.5)
    rotations = (
        [quarter_turn, quarter_turn, 0.0, 0.0],
        [quarter_turn, -quarter_turn, 0.0, 0.0],
        [quarter_turn, 0.0, quarter_turn, 0.0],
        [quarter_turn, 0.0, -quarter_turn, 0.0],
    )
    best_pose = initial_pose
    best_height = float(initial_bounds[1, 2] - initial_bounds[0, 2])
    for quaternion in rotations:
        candidate_pose = sapien.Pose(q=quaternion) * initial_pose
        set_pose(wrapper, candidate_pose)
        scene.update_render()
        bounds = actor_bounds(wrapper)
        height = float(bounds[1, 2] - bounds[0, 2])
        if height < best_height:
            best_pose = candidate_pose
            best_height = height
    set_pose(wrapper, best_pose)
    scene.update_render()
    if pen_is_upright(actor_bounds(wrapper)):
        raise RuntimeError("pen remains upright after orientation correction")

def load_object(scene: Any, choice: ObjectChoice) -> LoadedObject:
    wrapper = load_variant(scene, choice.spec, choice.variant)
    if wrapper is None:
        raise RuntimeError("asset loader returned None")
    try:
        if choice.label == "pen":
            lay_pen_flat(scene, wrapper)
        centered_pose, bounds = center_actor_on_ground(scene, wrapper)
        extents = bounds[1] - bounds[0]
        if not np.isfinite(extents).all() or np.any(extents <= 0):
            raise RuntimeError(f"invalid bounds: {bounds.tolist()}")

        footprint = np.asarray([float(extents[0]), float(extents[1])], dtype=float)
        radius = max(MIN_RADIUS, float(np.linalg.norm(footprint) / 2.0))
        height = float(extents[2])
        if (
            float(footprint.max()) > MAX_FOOTPRINT_XY
            or radius > MAX_RADIUS
            or height > MAX_HEIGHT
        ):
            raise RuntimeError(
                "asset is too large for tabletop sampling: "
                f"footprint={footprint.round(4).tolist()}, height={height:.4f}"
            )

        return LoadedObject(
            label=choice.label,
            asset_output=choice.spec["output"],
            source=choice.spec["source"],
            kind=choice.spec["kind"],
            variant_id=choice.variant.display_id,
            wrapper=wrapper,
            centered_pose=centered_pose,
            footprint=footprint,
            height=height,
            radius=radius,
        )
    except Exception:
        # An actor rejected after loading still belongs to the SAPIEN scene.
        # Hide it before sampling continues, otherwise it can appear as a giant
        # unrecorded object even though it is listed under "skipped".
        import sapien

        set_pose(wrapper, sapien.Pose([100.0, 100.0, -100.0]))
        raise


def move_offscreen(obj: LoadedObject) -> None:
    import sapien

    set_pose(obj.wrapper, sapien.Pose([4.0, 4.0, -3.0]) * obj.centered_pose)


def yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def place_loaded_object(obj: LoadedObject, xy: np.ndarray, z_offset: float, yaw: float) -> None:
    import sapien

    obj.xy = np.asarray(xy, dtype=float)
    obj.z_offset = float(z_offset)
    obj.yaw = float(yaw)
    pose = sapien.Pose(
        [float(obj.xy[0]), float(obj.xy[1]), TABLE_SURFACE_Z + obj.z_offset],
        yaw_quat(obj.yaw),
    )
    set_pose(obj.wrapper, pose * obj.centered_pose)


def object_order(labels: list[str], relations: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    for relation in relations:
        append_unique(ordered, relation["object"])
        append_unique(ordered, relation["subject"])
    for label in labels:
        append_unique(ordered, label)
    return ordered


def relation_for_label(
    label: str,
    relations: list[dict[str, Any]],
    placed: dict[str, LoadedObject],
) -> dict[str, Any] | None:
    candidates = [
        relation
        for relation in relations
        if relation["subject"] == label and relation["object"] in placed
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: float(item.get("frequency", 0)), reverse=True)
    return candidates[0]


def projected_half_extent(obj: LoadedObject, axis: np.ndarray) -> float:
    """Projection radius of an object's yaw-rotated XY bounding rectangle."""

    local_x = np.asarray([math.cos(obj.yaw), math.sin(obj.yaw)], dtype=float)
    local_y = np.asarray([-math.sin(obj.yaw), math.cos(obj.yaw)], dtype=float)
    half = obj.footprint * 0.5
    return float(abs(np.dot(axis, local_x)) * half[0] + abs(np.dot(axis, local_y)) * half[1])


def rotated_half_extents(obj: LoadedObject) -> np.ndarray:
    return np.asarray(
        [
            projected_half_extent(obj, np.asarray([1.0, 0.0])),
            projected_half_extent(obj, np.asarray([0.0, 1.0])),
        ],
        dtype=float,
    )


def rectangles_overlap(
    obj: LoadedObject,
    xy: np.ndarray,
    other: LoadedObject,
    other_xy: np.ndarray,
    clearance: float = PLACEMENT_CLEARANCE,
) -> bool:
    """2-D separating-axis test for two yaw-rotated footprints."""

    axes = [
        np.asarray([math.cos(obj.yaw), math.sin(obj.yaw)], dtype=float),
        np.asarray([-math.sin(obj.yaw), math.cos(obj.yaw)], dtype=float),
        np.asarray([math.cos(other.yaw), math.sin(other.yaw)], dtype=float),
        np.asarray([-math.sin(other.yaw), math.cos(other.yaw)], dtype=float),
    ]
    delta = np.asarray(other_xy, dtype=float) - np.asarray(xy, dtype=float)
    for axis in axes:
        required = projected_half_extent(obj, axis) + projected_half_extent(other, axis) + clearance
        if abs(float(np.dot(delta, axis))) >= required:
            return False
    return True


def vertical_intervals_overlap(
    obj: LoadedObject,
    z_offset: float,
    other: LoadedObject,
) -> bool:
    lower = max(float(z_offset), float(other.z_offset))
    upper = min(float(z_offset + obj.height), float(other.z_offset + other.height))
    return upper - lower > VERTICAL_COLLISION_TOLERANCE


def fits_table(obj: LoadedObject, xy: np.ndarray) -> bool:
    half = rotated_half_extents(obj) + PLACEMENT_CLEARANCE
    return (
        TABLE_XLIM[0] + half[0] <= xy[0] <= TABLE_XLIM[1] - half[0]
        and TABLE_YLIM[0] + half[1] <= xy[1] <= TABLE_YLIM[1] - half[1]
    )


def collides(
    obj: LoadedObject,
    xy: np.ndarray,
    placed: dict[str, LoadedObject],
    allow_overlap_with: str | None = None,
    z_offset: float = 0.0,
) -> bool:
    for other_label, other in placed.items():
        if other is obj or other.xy is None:
            continue
        if other_label == allow_overlap_with:
            continue
        if not vertical_intervals_overlap(obj, z_offset, other):
            continue
        if rectangles_overlap(obj, xy, other, other.xy):
            return True
    return False


def clip_xy_for_table(obj: LoadedObject, xy: np.ndarray) -> np.ndarray:
    half = rotated_half_extents(obj) + PLACEMENT_CLEARANCE
    return np.asarray(
        [
            np.clip(xy[0], TABLE_XLIM[0] + half[0], TABLE_XLIM[1] - half[0]),
            np.clip(xy[1], TABLE_YLIM[0] + half[1], TABLE_YLIM[1] - half[1]),
        ],
        dtype=float,
    )


def random_xy(
    rng: np.random.Generator,
    obj: LoadedObject,
    *,
    front_y_max: float | None = None,
) -> np.ndarray | None:
    half = rotated_half_extents(obj) + PLACEMENT_CLEARANCE
    if TABLE_XLIM[0] + half[0] >= TABLE_XLIM[1] - half[0]:
        return None
    if TABLE_YLIM[0] + half[1] >= TABLE_YLIM[1] - half[1]:
        return None
    if front_y_max is not None:
        y_high = min(TABLE_YLIM[1] - half[1], front_y_max)
        y_low = TABLE_YLIM[0] + half[1]
        if y_low >= y_high:
            return None
        sector_x = FRONT_SECTOR_CENTERS_X[
            int(rng.integers(0, len(FRONT_SECTOR_CENTERS_X)))
        ]
        xy = np.asarray(
            [
                rng.normal(sector_x, FRONT_SECTOR_STD_X),
                rng.uniform(y_low, y_high),
            ],
            dtype=float,
        )
        xy = clip_xy_for_table(obj, xy)
        xy[1] = min(xy[1], y_high)
        return xy
    if rng.random() < DENSE_SAMPLE_PROBABILITY:
        xy = CLUTTER_CENTER + rng.normal(0.0, CLUTTER_STD, size=2)
        return clip_xy_for_table(obj, xy)
    return np.asarray(
        [
            rng.uniform(TABLE_XLIM[0] + half[0], TABLE_XLIM[1] - half[0]),
            rng.uniform(TABLE_YLIM[0] + half[1], TABLE_YLIM[1] - half[1]),
        ],
        dtype=float,
    )


def relation_candidate(
    rng: np.random.Generator,
    obj: LoadedObject,
    support: LoadedObject,
    predicate: str,
) -> tuple[np.ndarray, float, str | None]:
    assert support.xy is not None
    predicate = predicate.lower()
    z_offset = 0.0
    allow_overlap_with: str | None = None

    if predicate == "in front of":
        axis = np.asarray([0.0, 1.0])
        spacing = (
            projected_half_extent(obj, axis)
            + projected_half_extent(support, axis)
            + rng.uniform(PLACEMENT_CLEARANCE, 0.018)
        )
        xy = support.xy + np.asarray(
            [rng.uniform(-0.045, 0.045), -spacing],
            dtype=float,
        )
    elif predicate == "on":
        xy = support.xy + rng.normal(0.0, 0.010, size=2)
        z_offset = support.z_offset + support.height + 0.006
        allow_overlap_with = support.label
    elif predicate == "inside":
        xy = support.xy + rng.normal(0.0, 0.012, size=2)
        z_offset = support.z_offset + max(0.018, support.height * 0.28)
        allow_overlap_with = support.label
    else:
        directions = [
            np.asarray([1.0, 0.0]),
            np.asarray([-1.0, 0.0]),
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, -1.0]),
        ]
        direction = directions[int(rng.integers(0, len(directions)))]
        tangent = np.asarray([-direction[1], direction[0]])
        spacing = (
            projected_half_extent(obj, direction)
            + projected_half_extent(support, direction)
            + rng.uniform(PLACEMENT_CLEARANCE, 0.018)
        )
        xy = support.xy + direction * spacing + tangent * rng.uniform(-0.04, 0.04)

    return np.asarray(xy, dtype=float), z_offset, allow_overlap_with


def reapply_on_relations(
    rng: np.random.Generator,
    placed: dict[str, LoadedObject],
    relations: list[dict[str, Any]],
    *,
    front_y_max: float | None = None,
) -> list[dict[str, Any]]:
    applied = []
    for relation in relations:
        if relation["predicate"].lower() != "on":
            continue
        subject = placed.get(relation["subject"])
        support = placed.get(relation["object"])
        if subject is None or support is None or support.xy is None:
            continue
        z_offset = support.z_offset + support.height + 0.006
        for _ in range(60):
            xy = support.xy + rng.normal(
                0.0,
                min(0.018, max(0.006, support.radius * 0.08)),
                size=2,
            )
            xy = clip_xy_for_table(subject, xy)
            if front_y_max is not None:
                xy[1] = min(xy[1], front_y_max)
            if collides(
                subject,
                xy,
                placed,
                allow_overlap_with=support.label,
                z_offset=z_offset,
            ):
                continue
            place_loaded_object(subject, xy, z_offset, subject.yaw)
            subject.relation = relation
            applied.append(relation)
            break
    return applied


def choose_pose(
    rng: np.random.Generator,
    obj: LoadedObject,
    relation: dict[str, Any] | None,
    placed: dict[str, LoadedObject],
    *,
    front_y_max: float | None = None,
) -> tuple[np.ndarray, float, str | None] | None:
    if relation is not None:
        support = placed[relation["object"]]
        for _ in range(60):
            xy, z_offset, allow_overlap_with = relation_candidate(
                rng,
                obj,
                support,
                relation["predicate"],
            )
            if (
                (front_y_max is None or xy[1] <= front_y_max)
                and fits_table(obj, xy)
                and not collides(
                    obj,
                    xy,
                    placed,
                    allow_overlap_with,
                    z_offset=z_offset,
                )
            ):
                obj.relation = relation
                return xy, z_offset, allow_overlap_with

    for _ in range(120):
        xy = random_xy(rng, obj, front_y_max=front_y_max)
        if xy is None:
            return None
        if fits_table(obj, xy) and not collides(obj, xy, placed, z_offset=0.0):
            return xy, 0.0, None
    return None


def choose_rear_center_pose(
    rng: np.random.Generator,
    obj: LoadedObject,
    placed: dict[str, LoadedObject],
) -> np.ndarray | None:
    """Place a large desktop device at the centered rear anchor."""

    for _ in range(120):
        xy = np.asarray(
            [
                np.clip(rng.normal(0.0, 0.045), *REAR_CENTER_XLIM),
                rng.uniform(*REAR_CENTER_YLIM),
            ],
            dtype=float,
        )
        xy = clip_xy_for_table(obj, xy)
        if fits_table(obj, xy) and not collides(obj, xy, placed, z_offset=0.0):
            return xy
    return None


def add_table_scene(scene: Any, rng: np.random.Generator) -> None:
    import sapien

    from envs.utils.create_actor import create_box, create_table

    scene.default_physical_material = scene.create_physical_material(0.55, 0.45, 0.02)
    add_scene_lighting(scene)
    scene.add_point_light([0.0, -0.75, 1.65], [1.0, 0.96, 0.9], shadow=True)
    scene.add_ground(0.0, render_material=[0.42, 0.43, 0.45])
    create_table(
        scene,
        sapien.Pose([0.0, 0.0, TABLE_SURFACE_Z]),
        length=TABLE_LENGTH,
        width=TABLE_WIDTH,
        height=TABLE_SURFACE_Z,
        thickness=0.055,
        color=(0.68, 0.59, 0.48),
        name="desk",
        is_static=True,
    )
    wall_color = tuple(float(v) for v in rng.uniform(0.78, 0.88, size=3))
    create_box(
        scene,
        sapien.Pose([0.0, TABLE_WIDTH / 2 + 0.055, 0.85]),
        half_size=[1.2, 0.035, 0.85],
        color=wall_color,
        is_static=True,
        name="back_wall",
    )


def add_camera(scene: Any, rng: np.random.Generator, width: int, height: int) -> Any:
    target = np.asarray(
        [
            rng.uniform(-0.03, 0.03),
            rng.uniform(-0.03, 0.04),
            TABLE_SURFACE_Z + rng.uniform(0.08, 0.14),
        ],
        dtype=float,
    )
    position = np.asarray(
        [
            rng.uniform(-0.10, 0.10),
            rng.uniform(-1.12, -0.92),
            rng.uniform(1.22, 1.42),
        ],
        dtype=float,
    )
    camera = scene.add_camera(
        "messy_desk_camera",
        width,
        height,
        math.radians(54),
        0.01,
        10.0,
    )
    camera.entity.set_pose(camera_pose(position, target))
    return camera


def render_camera(camera: Any, output_path: Path) -> None:
    camera.take_picture()
    rgba = camera.get_picture("Color")
    rgb = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
    mean = float(rgb.mean())
    std = float(rgb.std())
    percentile_95 = float(np.percentile(rgb, 95))
    if std < 1.0 or mean < 80.0 or percentile_95 < 110.0:
        raise RuntimeError(
            "rendered image is blank or severely underexposed: "
            f"mean={mean:.2f}, std={std:.2f}, p95={percentile_95:.2f}"
        )
    Image.fromarray(rgb).save(output_path, optimize=True)
    if output_path.stat().st_size == 0:
        raise RuntimeError(f"rendered image file is empty: {output_path}")


def layout_region(obj: LoadedObject) -> str:
    if obj.label in REAR_CENTER_DEVICE_LABELS:
        return "rear_center"
    assert obj.xy is not None
    if obj.xy[0] < -0.12:
        return "left_front"
    if obj.xy[0] > 0.12:
        return "right_front"
    return "front_center"


def object_record(obj: LoadedObject) -> dict[str, Any]:
    assert obj.xy is not None
    record = {
        "label": obj.label,
        "asset": obj.asset_output,
        "source": obj.source,
        "kind": obj.kind,
        "model_id": obj.variant_id,
        "xy": [round(float(v), 5) for v in obj.xy],
        "z_offset": round(float(obj.z_offset), 5),
        "yaw_degrees": round(math.degrees(obj.yaw), 2),
        "layout_region": layout_region(obj),
        "footprint": [round(float(v), 5) for v in obj.footprint],
        "height": round(float(obj.height), 5),
        "relation": obj.relation,
    }
    if obj.world_bounds is not None:
        record["world_bounds"] = [
            [round(float(value), 6) for value in row]
            for row in obj.world_bounds
        ]
    return record


def validate_physical_layout(
    placed: dict[str, LoadedObject],
) -> dict[str, Any]:
    """Reject penetration, floating objects, and unintended 3-D overlaps.

    Scene objects are static for deterministic previews, so physics cannot
    separate a bad initial layout.  We therefore validate both final simulator
    bounds and the yaw-oriented placement volumes before rendering.
    """

    violations: list[str] = []
    table_clearances: list[float] = []
    allowed_overlap_pairs: set[frozenset[str]] = set()

    for obj in placed.values():
        try:
            bounds = np.asarray(actor_bounds(obj.wrapper), dtype=float)
        except Exception as error:
            violations.append(f"{obj.label}:bounds_unavailable:{error}")
            continue
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            violations.append(f"{obj.label}:invalid_world_bounds")
            continue
        obj.world_bounds = bounds
        bottom = float(bounds[0, 2])
        table_clearances.append(bottom - TABLE_SURFACE_Z)
        if bottom < TABLE_SURFACE_Z - TABLE_PENETRATION_TOLERANCE:
            violations.append(
                f"{obj.label}:table_penetration={TABLE_SURFACE_Z - bottom:.6f}"
            )

    # Bounds for every actor must be available before checking a stacked pair;
    # dictionary order is not part of the physical contract.
    for obj in placed.values():
        if obj.world_bounds is None:
            continue
        bottom = float(obj.world_bounds[0, 2])
        relation = obj.relation if isinstance(obj.relation, dict) else None
        predicate = str(relation.get("predicate", "")).lower() if relation else ""
        support_label = relation.get("object") if relation else None
        support = placed.get(support_label) if isinstance(support_label, str) else None
        if predicate in {"on", "inside"} and support is not None:
            allowed_overlap_pairs.add(frozenset((obj.label, support.label)))
        if predicate == "on":
            if support is None or support.world_bounds is None:
                violations.append(f"{obj.label}:missing_on_support")
            else:
                gap = bottom - float(support.world_bounds[1, 2])
                if gap < -TABLE_PENETRATION_TOLERANCE:
                    violations.append(f"{obj.label}:penetrates_support={-gap:.6f}")
                elif gap > MAX_SUPPORT_GAP:
                    violations.append(f"{obj.label}:floats_above_support={gap:.6f}")
        elif predicate != "inside" and bottom > TABLE_SURFACE_Z + MAX_SUPPORT_GAP:
            violations.append(f"{obj.label}:floats_above_table={bottom - TABLE_SURFACE_Z:.6f}")

        if obj.xy is None or not fits_table(obj, obj.xy):
            violations.append(f"{obj.label}:outside_table_bounds")

    labels = list(placed)
    overlapping_pairs: list[list[str]] = []
    for index, label in enumerate(labels):
        obj = placed[label]
        if obj.xy is None:
            continue
        for other_label in labels[index + 1 :]:
            other = placed[other_label]
            if other.xy is None:
                continue
            pair = frozenset((label, other_label))
            if pair in allowed_overlap_pairs:
                continue
            if vertical_intervals_overlap(obj, obj.z_offset, other) and rectangles_overlap(
                obj,
                obj.xy,
                other,
                other.xy,
                clearance=0.0,
            ):
                overlapping_pairs.append([label, other_label])
                violations.append(f"{label}:intersects:{other_label}")

    return {
        "valid": not violations,
        "violations": violations,
        "overlapping_pairs": overlapping_pairs,
        "minimum_table_clearance": (
            round(min(table_clearances), 6) if table_clearances else None
        ),
        "placement_clearance": PLACEMENT_CLEARANCE,
        "table_penetration_tolerance": TABLE_PENETRATION_TOLERANCE,
        "max_support_gap": MAX_SUPPORT_GAP,
    }


def _generate_scene_in_scene(
    scene: Any,
    scene_index: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
    messy: dict[str, Any],
    label_specs: dict[str, list[dict[str, Any]]],
    variant_cache: dict[str, list[Variant]],
) -> dict[str, Any]:
    scene.set_timestep(1 / 250)
    add_table_scene(scene, rng)

    available_labels = set(label_specs)
    labels, selected_relations, selected_co_pairs, _ = sample_scene_labels(
        rng,
        messy,
        available_labels,
        args.min_objects,
        args.max_objects,
    )
    order = [
        label for label in labels if label in REAR_CENTER_DEVICE_LABELS
    ] + [
        label for label in labels if label not in REAR_CENTER_DEVICE_LABELS
    ]

    loaded_by_label: dict[str, LoadedObject] = {}
    placed: dict[str, LoadedObject] = {}
    skipped: list[dict[str, str]] = []

    for label in order:
        choice = choose_object(rng, label, label_specs, variant_cache)
        if choice is None:
            skipped.append({"label": label, "reason": "no mapped asset variant"})
            continue
        try:
            obj = load_object(scene, choice)
        except Exception as error:
            skipped.append({"label": label, "reason": str(error)})
            continue
        loaded_by_label[label] = obj
        if label in REAR_CENTER_DEVICE_LABELS:
            obj.yaw = 0.0
        elif label in FRONT_FACING_CLOCK_LABELS:
            obj.yaw = float(rng.uniform(-math.pi / 12, math.pi / 12))
        else:
            obj.yaw = float(rng.uniform(-math.pi, math.pi))

        if label in REAR_CENTER_DEVICE_LABELS:
            rear_xy = choose_rear_center_pose(rng, obj, placed)
            placement = (rear_xy, 0.0, None) if rear_xy is not None else None
        else:
            rear_anchor = next(
                (
                    item
                    for item in placed.values()
                    if item.label in REAR_CENTER_DEVICE_LABELS and item.xy is not None
                ),
                None,
            )
            front_y_max = (
                float(rear_anchor.xy[1] - FRONT_CLEARANCE)
                if rear_anchor is not None and rear_anchor.xy is not None
                else None
            )
            placement = choose_pose(
                rng,
                obj,
                None,
                placed,
                front_y_max=front_y_max,
            )
        if placement is None:
            move_offscreen(obj)
            skipped.append({"label": label, "reason": "could not place on tabletop"})
            continue

        xy, z_offset, _ = placement
        place_loaded_object(obj, xy, z_offset, obj.yaw)
        placed[label] = obj

    if len(placed) < args.min_placed:
        raise RuntimeError(
            f"only placed {len(placed)} object(s), fewer than --min-placed={args.min_placed}"
        )

    rear_devices = [
        obj for obj in placed.values() if obj.label in REAR_CENTER_DEVICE_LABELS
    ]
    layout_violations: list[str] = []
    if len(rear_devices) > 1:
        layout_violations.append("more_than_one_rear_center_device")
    if rear_devices:
        rear = rear_devices[0]
        assert rear.xy is not None
        if not (REAR_CENTER_XLIM[0] <= rear.xy[0] <= REAR_CENTER_XLIM[1]):
            layout_violations.append(f"{rear.label}:not_centered_x")
        if not (REAR_CENTER_YLIM[0] <= rear.xy[1] <= REAR_CENTER_YLIM[1]):
            layout_violations.append(f"{rear.label}:not_at_rear_y")
        if abs(rear.yaw) > 1e-6:
            layout_violations.append(f"{rear.label}:not_facing_front")
        front_y_max = float(rear.xy[1] - FRONT_CLEARANCE)
        for obj in placed.values():
            if obj is rear or obj.xy is None:
                continue
            if obj.xy[1] > front_y_max + 1e-6:
                layout_violations.append(f"{obj.label}:not_in_front_of_{rear.label}")
    if layout_violations:
        raise RuntimeError("layout constraint violation: " + ", ".join(layout_violations))

    for _ in range(args.render_steps):
        scene.step()
    scene.update_render()
    physical_validation = validate_physical_layout(placed)
    if not physical_validation["valid"]:
        raise RuntimeError(
            "physical layout validation failed: "
            + ", ".join(physical_validation["violations"])
        )
    camera = add_camera(scene, rng, args.width, args.height)
    scene.update_render()

    image_name = f"scene_{scene_index:03d}.png"
    image_path = args.output_dir / image_name
    render_camera(camera, image_path)

    record = {
        "image": image_name,
        "sampling_policy": SAMPLING_POLICY,
        "sampled_labels": labels,
        "placed_objects": [object_record(obj) for obj in placed.values()],
        "relationships_sampled": selected_relations,
        "on_relationships": [],
        # Keep the legacy key, but distinguish unknown visibility from a
        # measured absence of occlusion. Never report a staged pair as truth.
        "occlusion": None,
        "occlusion_status": "not_measured",
        "physical_validation": physical_validation,
        "co_occurence_sampled": selected_co_pairs,
        "layout_policy": {
            "camera_side": "negative_y_front",
            "rear_center_device_labels": sorted(REAR_CENTER_DEVICE_LABELS),
            "rear_device_count": len(rear_devices),
            "rear_center_xlim": list(REAR_CENTER_XLIM),
            "rear_center_ylim": list(REAR_CENTER_YLIM),
            "rear_device_selection": "weighted_sampling_at_most_one",
            "rear_device_yaw_degrees": 0.0,
            "foreground_regions": ["left_front", "front_center", "right_front"],
            "violations": layout_violations,
            "valid": not layout_violations,
        },
        "skipped": skipped,
        "resolution": [args.width, args.height],
    }
    return record


def generate_scene(
    scene_index: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
    messy: dict[str, Any],
    label_specs: dict[str, list[dict[str, Any]]],
    variant_cache: dict[str, list[Variant]],
) -> dict[str, Any]:
    """Create one isolated scene and always release simulator resources."""

    import sapien

    scene = sapien.Scene()
    try:
        return _generate_scene_in_scene(
            scene,
            scene_index,
            rng,
            args,
            messy,
            label_specs,
            variant_cache,
        )
    finally:
        scene.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=10, help="Number of scenes to render.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--width", type=int, default=1024, help="Image width.")
    parser.add_argument("--height", type=int, default=768, help="Image height.")
    parser.add_argument("--min-objects", type=int, default=9, help="Minimum sampled labels.")
    parser.add_argument("--max-objects", type=int, default=11, help="Maximum sampled labels.")
    parser.add_argument("--min-placed", type=int, default=9, help="Minimum placed objects per scene.")
    parser.add_argument("--render-steps", type=int, default=4, help="Physics steps before rendering.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum candidate attempts; defaults to three times --scenes.",
    )
    parser.add_argument(
        "--allow-missing-on",
        action="store_true",
        default=True,
        help="Compatibility flag; natural generation never requires an 'on' relation.",
    )
    parser.add_argument(
        "--allow-missing-occlusion",
        action="store_true",
        default=True,
        help="Compatibility flag; natural generation never forces an occlusion pair.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for scene PNGs and manifest.json.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed scene.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scenes < 1:
        raise ValueError("--scenes must be >= 1")
    if args.min_objects < 1 or args.max_objects < args.min_objects:
        raise ValueError("--max-objects must be >= --min-objects >= 1")
    if args.min_placed < 1:
        raise ValueError("--min-placed must be >= 1")
    if args.width < 128 or args.height < 128:
        raise ValueError("--width and --height must be >= 128")
    max_attempts = args.max_attempts or args.scenes * 3
    if max_attempts < args.scenes:
        raise ValueError("--max-attempts must be >= --scenes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    messy = load_messy_config()
    specs = load_specs()
    current_labels = set(messy["objects"])
    label_specs = {
        label: spec_list
        for label, spec_list in build_label_specs(specs).items()
        if label in current_labels and label not in EXCLUDED_DESK_LABELS
    }
    variant_cache = {
        spec["output"]: discover_variants(spec)
        for spec in specs
        if any(label in label_specs for label in spec["objects"])
    }
    label_specs = {
        label: [
            spec
            for spec in spec_list
            if variant_cache.get(spec["output"])
        ]
        for label, spec_list in label_specs.items()
    }
    label_specs = {label: spec_list for label, spec_list in label_specs.items() if spec_list}

    manifest: dict[str, Any] = {
        "generator": "script/scence_generator.py",
        "messy_json": str(MESSY_JSON.relative_to(ROOT)),
        "asset_map": str(ASSET_MAP_JSON.relative_to(ROOT)),
        "seed": args.seed,
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "sapien_no_display": os.environ.get("SAPIEN_NO_DISPLAY"),
            "sapien_renderer": os.environ.get("SAPIEN_RENDERER"),
        },
        "generation_config": {
            "sampling_policy": SAMPLING_POLICY,
            "object_weight_semantics": "relative_weights_without_replacement",
            "forced_relationships": False,
            "forced_occlusion": False,
            "occlusion_status": "not_measured",
            "scenes": args.scenes,
            "width": args.width,
            "height": args.height,
            "min_objects": args.min_objects,
            "max_objects": args.max_objects,
            "min_placed": args.min_placed,
            "render_steps": args.render_steps,
            "max_attempts": max_attempts,
            "allow_missing_on": args.allow_missing_on,
            "allow_missing_occlusion": args.allow_missing_occlusion,
            "physical_validation": {
                "oriented_footprints": True,
                "placement_clearance": PLACEMENT_CLEARANCE,
                "vertical_collision_tolerance": VERTICAL_COLLISION_TOLERANCE,
                "table_penetration_tolerance": TABLE_PENETRATION_TOLERANCE,
                "max_support_gap": MAX_SUPPORT_GAP,
            },
        },
        "table": {
            "length": TABLE_LENGTH,
            "width": TABLE_WIDTH,
            "surface_z": TABLE_SURFACE_Z,
        },
        "available_json_objects": len(label_specs),
        "scenes": [],
        "failures": [],
    }

    scene_index = 0
    attempt_index = 0
    while scene_index < args.scenes and attempt_index < max_attempts:
        try:
            scene_record = generate_scene(
                scene_index,
                rng,
                args,
                messy,
                label_specs,
                variant_cache,
            )
            # These indices make records traceable after multiple manifests are
            # aggregated by script/batch_generate_scenes.py.
            scene_record["scene_index"] = scene_index
            scene_record["attempt_index"] = attempt_index
            manifest["scenes"].append(scene_record)
            print(
                f"PASS attempt_{attempt_index:03d} -> scene_{scene_index:03d}: "
                f"{len(scene_record['placed_objects'])} objects -> {scene_record['image']}",
                flush=True,
            )
            scene_index += 1
        except Exception as error:
            failure = {
                "attempt_index": attempt_index,
                "target_scene_index": scene_index,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            manifest["failures"].append(failure)
            print(
                f"FAIL attempt_{attempt_index:03d} for scene_{scene_index:03d}: {error}",
                file=sys.stderr,
                flush=True,
            )
            if args.fail_fast:
                break
        finally:
            attempt_index += 1

    manifest["summary"] = {
        "requested_scenes": args.scenes,
        "candidate_attempts": attempt_index,
        "max_attempts": max_attempts,
        "rendered_scenes": len(manifest["scenes"]),
        "failures": len(manifest["failures"]),
        "images": [scene["image"] for scene in manifest["scenes"]],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False), flush=True)
    if len(manifest["scenes"]) < args.scenes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
