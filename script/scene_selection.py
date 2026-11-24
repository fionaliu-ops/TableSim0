#!/usr/bin/env python3
"""Manifest and image based quality checks for generated TableSim scenes.

This module deliberately has no SAPIEN dependency.  It can therefore be used
on a login/CPU node to re-score manifests produced by GPU workers.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageStat


DEFAULT_TARGET_LABELS = (
    "mouse",
    "pen",
    "pencil",
    "eraser",
    "sticky note",
    "book",
)


@dataclass(frozen=True)
class SceneFilterConfig:
    """Thresholds used by :func:`evaluate_scene`.

    The default workspace follows the ALOHA reachable region documented in
    ``ASSETS_AND_ISAAC_SIM_SCENE_GUIDE.md``.  It is intentionally configurable
    because another embodiment may have a different workspace.
    """

    min_placed: int = 9
    max_skipped: int = 2
    require_on: bool = False
    require_occlusion: bool = False
    require_layout_valid: bool = True
    require_target_candidate: bool = True
    validate_images: bool = True
    target_labels: tuple[str, ...] = DEFAULT_TARGET_LABELS
    workspace_x: tuple[float, float] = (-0.36, 0.36)
    workspace_y: tuple[float, float] = (-0.27, 0.22)
    min_target_clearance: float = 0.015
    max_target_footprint: float = 0.30
    max_target_height: float = 0.20
    min_luminance: float = 15.0
    max_luminance: float = 245.0
    min_luminance_stddev: float = 3.0
    max_clipped_fraction: float = 0.98
    min_entropy: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _xy(record: dict[str, Any]) -> tuple[float, float] | None:
    value = record.get("xy")
    if not isinstance(value, list) or len(value) != 2 or not all(_finite_number(v) for v in value):
        return None
    return float(value[0]), float(value[1])


def _footprint(record: dict[str, Any]) -> tuple[float, float] | None:
    value = record.get("footprint")
    if not isinstance(value, list) or len(value) != 2 or not all(_finite_number(v) for v in value):
        return None
    width, depth = float(value[0]), float(value[1])
    if width <= 0 or depth <= 0:
        return None
    return width, depth


def _radius(record: dict[str, Any]) -> float | None:
    footprint = _footprint(record)
    if footprint is None:
        return None
    return 0.5 * math.hypot(*footprint)


def inspect_image(image_path: Path, config: SceneFilterConfig) -> dict[str, Any]:
    """Load an image and return cheap corruption/exposure diagnostics."""

    result: dict[str, Any] = {"path": str(image_path), "valid": False, "reasons": []}
    if not image_path.is_file():
        result["reasons"].append("image_missing")
        return result

    try:
        with Image.open(image_path) as source:
            source.load()
            result["format"] = source.format
            result["size"] = list(source.size)
            gray = source.convert("L")
            gray.thumbnail((256, 256))
            stat = ImageStat.Stat(gray)
            mean = float(stat.mean[0])
            stddev = float(stat.stddev[0])
            histogram = gray.histogram()
            pixels = max(1, sum(histogram))
            clipped = (sum(histogram[:3]) + sum(histogram[-3:])) / pixels
            entropy = float(gray.entropy())
    except Exception as error:  # PIL exposes several decoder-specific errors.
        result["reasons"].append(f"image_unreadable:{type(error).__name__}")
        return result

    result.update(
        {
            "mean_luminance": round(mean, 3),
            "luminance_stddev": round(stddev, 3),
            "clipped_fraction": round(clipped, 6),
            "entropy": round(entropy, 3),
        }
    )
    if mean < config.min_luminance:
        result["reasons"].append("image_too_dark")
    if mean > config.max_luminance:
        result["reasons"].append("image_too_bright")
    if stddev < config.min_luminance_stddev:
        result["reasons"].append("image_low_contrast")
    if clipped > config.max_clipped_fraction:
        result["reasons"].append("image_excessively_clipped")
    if entropy < config.min_entropy:
        result["reasons"].append("image_low_entropy")
    result["valid"] = not result["reasons"]
    return result


def _relation_pairs(scene: dict[str, Any]) -> tuple[set[str], dict[str, set[str]]]:
    """Return labels covered by an ``on`` subject and its support labels."""

    covered: set[str] = set()
    supports_by_subject: dict[str, set[str]] = {}
    relations = scene.get("on_relationships") or []
    if not isinstance(relations, list):
        return covered, supports_by_subject
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        subject = relation.get("subject")
        support = relation.get("object")
        if isinstance(subject, str) and isinstance(support, str):
            covered.add(support)
            supports_by_subject.setdefault(subject, set()).add(support)
    return covered, supports_by_subject


def find_target_candidates(
    scene: dict[str, Any],
    config: SceneFilterConfig,
) -> list[dict[str, Any]]:
    """Assess lightweight objects that could become a first pick-task target.

    This is a conservative geometry heuristic, not a grasp-planning result.  A
    target hidden behind the generator's explicit occluder, used as a support,
    outside the documented workspace, or packed too tightly is not advertised
    as task-ready.
    """

    objects = scene.get("placed_objects") or []
    if not isinstance(objects, list):
        return []
    objects = [item for item in objects if isinstance(item, dict)]
    covered_labels, supports_by_subject = _relation_pairs(scene)
    occlusion = scene.get("occlusion")
    occluded_label = occlusion.get("target") if isinstance(occlusion, dict) else None
    allowed = set(config.target_labels)
    candidates: list[dict[str, Any]] = []

    for index, obj in enumerate(objects):
        label = obj.get("label")
        if label not in allowed:
            continue
        reasons: list[str] = []
        xy = _xy(obj)
        footprint = _footprint(obj)
        height = obj.get("height")
        if xy is None:
            reasons.append("invalid_xy")
        elif not (
            config.workspace_x[0] <= xy[0] <= config.workspace_x[1]
            and config.workspace_y[0] <= xy[1] <= config.workspace_y[1]
        ):
            reasons.append("outside_workspace")
        if footprint is None:
            reasons.append("invalid_footprint")
        elif max(footprint) > config.max_target_footprint:
            reasons.append("target_footprint_too_large")
        if not _finite_number(height) or float(height) <= 0:
            reasons.append("invalid_height")
        elif float(height) > config.max_target_height:
            reasons.append("target_too_tall")
        if label in covered_labels:
            reasons.append("supports_an_on_object")
        if label == occluded_label:
            reasons.append("explicit_occlusion_target")

        clearance: float | None = None
        radius = _radius(obj)
        if xy is not None and radius is not None:
            edge_gaps: list[float] = []
            for other_index, other in enumerate(objects):
                if index == other_index:
                    continue
                other_label = other.get("label")
                # An object intentionally resting on a support overlaps its
                # support in XY; that overlap does not block a top grasp.
                if other_label in supports_by_subject.get(str(label), set()):
                    continue
                other_xy = _xy(other)
                other_radius = _radius(other)
                if other_xy is None or other_radius is None:
                    continue
                center_distance = math.hypot(xy[0] - other_xy[0], xy[1] - other_xy[1])
                edge_gaps.append(center_distance - radius - other_radius)
            clearance = min(edge_gaps) if edge_gaps else float("inf")
            if clearance < config.min_target_clearance:
                reasons.append("insufficient_approach_clearance")

        candidates.append(
            {
                "label": label,
                "asset": obj.get("asset"),
                "source": obj.get("source"),
                "model_id": obj.get("model_id"),
                "xy": list(xy) if xy is not None else obj.get("xy"),
                "clearance": None if clearance is None else (
                    None if math.isinf(clearance) else round(clearance, 5)
                ),
                "eligible": not reasons,
                "reasons": reasons,
            }
        )
    return candidates


def _validate_objects(scene: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    objects = scene.get("placed_objects")
    if not isinstance(objects, list):
        return ["placed_objects_not_a_list"]
    labels: list[str] = []
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            reasons.append(f"object_{index}:not_a_mapping")
            continue
        label = obj.get("label")
        if not isinstance(label, str) or not label:
            reasons.append(f"object_{index}:invalid_label")
        else:
            labels.append(label)
        if _xy(obj) is None:
            reasons.append(f"object_{index}:invalid_xy")
        if _footprint(obj) is None:
            reasons.append(f"object_{index}:invalid_footprint")
        height = obj.get("height")
        if not _finite_number(height) or float(height) <= 0:
            reasons.append(f"object_{index}:invalid_height")
        yaw = obj.get("yaw_degrees")
        if not _finite_number(yaw):
            reasons.append(f"object_{index}:invalid_yaw")
    if len(labels) != len(set(labels)):
        reasons.append("duplicate_object_labels")
    return reasons


def _score_scene(
    scene: dict[str, Any],
    config: SceneFilterConfig,
    candidates: list[dict[str, Any]],
    image_metrics: dict[str, Any] | None,
) -> float:
    """Produce a deterministic 0-100 ranking score for otherwise valid scenes."""

    placed_count = len(scene.get("placed_objects") or [])
    skipped_count = len(scene.get("skipped") or [])
    eligible = [item for item in candidates if item["eligible"]]
    score = min(20.0, 20.0 * placed_count / max(1, config.min_placed))
    # Optional relation records must not bias ranking toward staged layouts.
    score += 10.0 if not config.require_on or scene.get("on_relationships") else 0.0
    score += 10.0 if not config.require_occlusion or scene.get("occlusion") else 0.0
    layout = scene.get("layout_policy")
    score += 10.0 if not isinstance(layout, dict) or layout.get("valid", True) else 0.0
    score += min(20.0, 8.0 + 4.0 * len(eligible)) if eligible else 0.0
    finite_clearances = [item["clearance"] for item in eligible if item["clearance"] is not None]
    if eligible and not finite_clearances:
        score += 15.0
    elif finite_clearances:
        best = max(float(value) for value in finite_clearances)
        score += min(15.0, max(0.0, best / max(0.001, config.min_target_clearance) * 7.5))
    if image_metrics is None or image_metrics.get("valid"):
        score += 15.0
    score -= min(10.0, skipped_count * 2.5)
    return round(max(0.0, min(100.0, score)), 3)


def evaluate_scene(
    scene: dict[str, Any],
    image_path: Path,
    config: SceneFilterConfig,
) -> dict[str, Any]:
    """Evaluate one scene record and return an auditable decision."""

    reasons = _validate_objects(scene)
    placed = scene.get("placed_objects") if isinstance(scene.get("placed_objects"), list) else []
    skipped = scene.get("skipped") if isinstance(scene.get("skipped"), list) else []
    if len(placed) < config.min_placed:
        reasons.append(f"too_few_placed_objects:{len(placed)}<{config.min_placed}")
    if len(skipped) > config.max_skipped:
        reasons.append(f"too_many_skipped_objects:{len(skipped)}>{config.max_skipped}")
    if config.require_on and not scene.get("on_relationships"):
        reasons.append("missing_on_relationship")
    if config.require_occlusion and not scene.get("occlusion"):
        reasons.append("missing_occlusion")
    layout = scene.get("layout_policy")
    if config.require_layout_valid and isinstance(layout, dict):
        if not layout.get("valid", False) or layout.get("violations"):
            reasons.append("invalid_layout_policy")
    physical = scene.get("physical_validation")
    if isinstance(physical, dict):
        if not physical.get("valid", False) or physical.get("violations"):
            reasons.append("invalid_physical_layout")

    resolution = scene.get("resolution")
    image_metrics = inspect_image(image_path, config) if config.validate_images else None
    if image_metrics is not None:
        if not image_metrics["valid"]:
            reasons.extend(image_metrics["reasons"])
        if (
            isinstance(resolution, list)
            and len(resolution) == 2
            and image_metrics.get("size") != resolution
        ):
            reasons.append("image_resolution_mismatch")

    candidates = find_target_candidates(scene, config)
    eligible_candidates = [item for item in candidates if item["eligible"]]
    if config.require_target_candidate and not eligible_candidates:
        reasons.append("no_eligible_target_candidate")

    # Preserve order while removing duplicate reason strings.
    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "accepted": not unique_reasons,
        "score": _score_scene(scene, config, candidates, image_metrics),
        "reasons": unique_reasons,
        "placed_count": len(placed),
        "skipped_count": len(skipped),
        "target_candidates": candidates,
        "eligible_target_labels": [item["label"] for item in eligible_candidates],
        "image_metrics": image_metrics,
    }


def _label_set(result: dict[str, Any]) -> set[str]:
    scene = result.get("scene") or {}
    return {
        str(item.get("label"))
        for item in scene.get("placed_objects", [])
        if isinstance(item, dict) and item.get("label")
    }


def select_diverse(
    results: Iterable[dict[str, Any]],
    limit: int,
    diversity_bonus: float = 8.0,
) -> list[dict[str, Any]]:
    """Greedily select high-quality scenes while avoiding near duplicates."""

    remaining = [item for item in results if item.get("accepted")]
    remaining.sort(key=lambda item: (-float(item.get("score", 0)), str(item.get("scene_id", ""))))
    if limit <= 0 or limit >= len(remaining):
        return remaining

    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < limit:
        if not selected:
            selected.append(remaining.pop(0))
            continue

        selected_sets = [_label_set(item) for item in selected]

        def rank(item: dict[str, Any]) -> tuple[float, float, str]:
            labels = _label_set(item)
            similarities = []
            for other in selected_sets:
                union = labels | other
                similarities.append(len(labels & other) / len(union) if union else 1.0)
            novelty = 1.0 - max(similarities, default=0.0)
            combined = float(item.get("score", 0)) + diversity_bonus * novelty
            return combined, float(item.get("score", 0)), str(item.get("scene_id", ""))

        best = max(remaining, key=rank)
        remaining.remove(best)
        selected.append(best)
    return selected
