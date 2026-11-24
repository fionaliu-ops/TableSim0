from __future__ import annotations

import argparse
import atexit
import bisect
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from high_level_planner.perception import (
    DEFAULT_SAM3_CROP_ROI,
    LabelSlidingWindowPerceptionBackend,
    RecordingPerceptionBackend,
    RobotExclusionPerceptionBackend,
    Sam3PerceptionClient,
    Sam3TrackingClient,
    Sam3VideoMemoryPerceptionBackend,
    crop_image_file,
    load_prompt_catalog,
    parse_normalized_crop_roi,
    pixel_crop_bbox,
    render_detection_pipeline,
    result_audit,
)
from high_level_planner.scene_graph import (
    MultiFrameSceneGraphInitializer,
    SceneGraphConstructor,
    SceneGraphUpdater,
)
from high_level_planner.scene_graph.graph import SceneObject, unique_id
from high_level_planner.scene_graph.updater import (
    DETECTOR_VISIBILITY_POLICY,
    ON_RELATION_VISIBILITY_POLICY,
)


DEFAULT_ROBOT_PROMPTS = "robotic arm,robot arm,robot gripper,robot"
DEFAULT_CROP_ROI_TEXT = ",".join(
    f"{value:.10f}"
    for value in DEFAULT_SAM3_CROP_ROI
)
SEMANTIC_RELATIONS = {"on", "blocks", "occludes", "overlaps", "next_to"}
RELATION_PRIORITY = {
    "on": 0,
    "blocks": 0,
    "occludes": 0,
    "overlaps": 1,
    "next_to": 2,
    "left_of": 3,
    "right_of": 3,
    "behind_image": 4,
    "in_front_image": 4,
}
RELATION_COLORS = {
    "on": "#8e44ad",
    "blocks": "#c0392b",
    "occludes": "#c0392b",
    "overlaps": "#e67e22",
    "next_to": "#229954",
    "left_of": "#607d8b",
    "right_of": "#607d8b",
    "behind_image": "#78909c",
    "in_front_image": "#78909c",
}
RELATION_EDGE_VALUE_SPEC = {
    "format": "relationship: (delta_x, delta_y)",
    "distance_semantics": (
        "absolute_difference_between_source_and_target_"
        "object_center_xy_not_euclidean_magnitude"
    ),
    "distance_coordinate_frame": "image_normalized_xy",
    "distance_json_field": "[abs(delta_x), abs(delta_y)]",
    "next_to_direction": (
        "direction_is_left_or_right_for_target_relative_to_source"
    ),
    "visual_label_example": "next_to_right: (0.125, 0.040)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run incremental SAM3 Scene Graph updates on extracted PNG images, "
            "save every update, and render an operation/graph side-by-side MP4."
        )
    )
    parser.add_argument(
        "--frames-manifest",
        required=True,
        help=(
            "Image-sequence manifest produced by extract_video_frames.py. "
            "Scene Graph perception accepts only the PNG images listed here."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <video_stem>_scene_graph next to the input.",
    )
    parser.add_argument(
        "--output-video",
        default=None,
        help="Output MP4 path. Defaults to <output-dir>/<video_stem>_scene_graph_demo.mp4.",
    )
    parser.add_argument(
        "--instruction",
        default="Track tabletop objects while the robot operates on the desk.",
    )
    parser.add_argument(
        "--sam3-endpoint",
        default=os.environ.get("TABLESIM_SAM3_ENDPOINT", "http://sam3:8003"),
    )
    parser.add_argument(
        "--prompts",
        default="",
        help=(
            "Optional comma-separated SAM3 prompts overriding --prompts-file. "
            "The default empty value loads the existing prompt catalog."
        ),
    )
    parser.add_argument(
        "--prompts-file",
        default="image_collectors/prompts.json",
        help="JSON/text prompt catalog used when --prompts is empty.",
    )
    parser.add_argument("--max-prompts", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--min-mask-area", type=int, default=50)
    parser.add_argument("--max-detections", type=int, default=30)
    parser.add_argument("--sam3-timeout", type=float, default=300.0)
    parser.add_argument(
        "--sam3-video-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Treat the ordered 5 Hz PNG sequence as one SAM3 mask-memory "
            "tracking session. Disable only for the legacy independent-image path."
        ),
    )
    parser.add_argument(
        "--sam3-redetect-interval",
        type=int,
        default=5,
        help=(
            "Run the semantic image detector every N sampled frames to correct "
            "tracker drift and labels (default: 5 frames = 1 second at 5 Hz)."
        ),
    )
    parser.add_argument(
        "--sam3-tracker-match-iou",
        type=float,
        default=0.25,
        help="Minimum mask IoU for periodic detector-to-memory re-anchoring.",
    )
    parser.add_argument(
        "--sam3-server-project-root",
        default="/workspace/TableSim",
        help=(
            "TableSim project root visible inside the SAM3 sidecar. The crop "
            "directory is translated relative to the local project root."
        ),
    )
    parser.add_argument(
        "--initialization-window",
        type=float,
        default=20.0,
        help=(
            "Seconds during which unmatched detections may become nodes after "
            "multi-frame confirmation. The node set is locked afterward."
        ),
    )
    parser.add_argument(
        "--initialization-min-observations",
        type=int,
        default=2,
        help="Distinct initialization observations required to admit a new node.",
    )
    parser.add_argument(
        "--initialization-min-confidence",
        type=float,
        default=0.5,
        help=(
            "Minimum confidence for an unmatched detection to count toward "
            "initialization confirmation."
        ),
    )
    parser.add_argument(
        "--initialization-match-distance",
        type=float,
        default=0.08,
        help=(
            "Maximum normalized xy distance for unmatched detections to belong "
            "to the same initialization candidate track."
        ),
    )
    parser.add_argument(
        "--initialization-max-gap",
        type=float,
        default=3.0,
        help=(
            "Maximum seconds between observations of one initialization "
            "candidate track."
        ),
    )
    parser.add_argument(
        "--label-window-size",
        type=int,
        default=6,
        help=(
            "Label-correction window size. TableSim requires exactly 6: the "
            "current frame plus the five preceding frames."
        ),
    )
    parser.add_argument(
        "--label-window-match-distance",
        type=float,
        default=0.08,
        help=(
            "Maximum normalized bbox-center distance for matching the same "
            "position across the six-frame label window."
        ),
    )
    parser.add_argument(
        "--label-window-min-bbox-iou",
        type=float,
        default=0.20,
        help="Minimum bbox IoU for a sliding-window position match.",
    )
    parser.add_argument(
        "--label-window-min-size-similarity",
        type=float,
        default=0.50,
        help=(
            "Minimum bbox width/height similarity for a center-distance "
            "sliding-window match."
        ),
    )
    parser.add_argument(
        "--robot-exclusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run a separate robot query, remove robot-shaped semantic duplicates, "
            "and ensure robot hardware never contributes visibility evidence."
        ),
    )
    parser.add_argument(
        "--robot-prompts",
        default=DEFAULT_ROBOT_PROMPTS,
        help="Comma-separated auxiliary prompts excluded from the Scene Graph.",
    )
    parser.add_argument(
        "--robot-match-iou",
        type=float,
        default=0.65,
        help="BBox IoU used to remove object detections matching robot hardware.",
    )
    parser.add_argument(
        "--robot-mask-containment",
        type=float,
        default=0.65,
        help=(
            "Remove an object detection when this fraction of its mask is "
            "contained in a robot auxiliary mask."
        ),
    )
    parser.add_argument(
        "--sam3-crop-roi",
        default=DEFAULT_CROP_ROI_TEXT,
        help=(
            "Normalized left,top,right,bottom crop ROI. The default maps a "
            "640x480 image to x=0..640, y=60..450 (640x390), retaining the "
            "table edges while excluding the exposed lower ALOHA base."
        ),
    )
    parser.add_argument(
        "--visibility-policy",
        choices=(DETECTOR_VISIBILITY_POLICY, ON_RELATION_VISIBILITY_POLICY),
        default=ON_RELATION_VISIBILITY_POLICY,
        help=(
            "Post-update visibility policy. on_relation makes an object partially "
            "visible only when another Scene Graph object has an incoming on relation."
        ),
    )
    parser.add_argument("--max-relations", type=int, default=18)
    parser.add_argument("--output-height", type=int, default=720)
    parser.add_argument("--graph-width", type=int, default=960)
    parser.add_argument("--graph-dpi", type=int, default=120)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this script's existing generated artifacts in the output directory.",
    )
    return parser.parse_args()


def parse_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts.strip():
        raw_prompts = args.prompts.split(",")
    else:
        raw_prompts = load_prompt_catalog(args.prompts_file)

    prompts: list[str] = []
    seen: set[str] = set()
    for item in raw_prompts:
        prompt = str(item).strip()
        key = prompt.casefold()
        if prompt and key not in seen:
            prompts.append(prompt)
            seen.add(key)
    if not prompts:
        raise ValueError("At least one SAM3 prompt is required")
    return prompts[: max(1, args.max_prompts)]


def parse_robot_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for item in args.robot_prompts.split(","):
        prompt = item.strip()
        key = prompt.casefold()
        if prompt and key not in seen:
            prompts.append(prompt)
            seen.add(key)
    if args.robot_exclusion and not prompts:
        raise ValueError("--robot-exclusion requires at least one --robot-prompts value")
    return prompts


def prepare_output(
    output_dir: Path,
    output_video: Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    legacy_frames_dir = output_dir / "frames"
    updates_dir = output_dir / "updates"
    intermediate_dir = output_dir / "sam3_intermediate"
    crop_dir = output_dir / "sam3_cropped_inputs"
    intermediate_subdirs = (
        intermediate_dir / "object_query",
        intermediate_dir / "robot_query",
        intermediate_dir / "filtered",
        intermediate_dir / "pipeline",
        intermediate_dir / "json",
    )
    legacy_intermediate_subdirs = (
        intermediate_dir / "robot_filtered",
        intermediate_dir / "surface_query",
    )
    generated = [
        output_dir / "manifest.json",
        output_video,
        *legacy_frames_dir.glob("frame_*.png"),
        *updates_dir.glob("scene_graph_update_*.json"),
        *updates_dir.glob("scene_graph_update_*.png"),
        *crop_dir.glob("sam3_crop_*.png"),
        *(
            path
            for directory in (
                *intermediate_subdirs,
                *legacy_intermediate_subdirs,
            )
            for path in directory.glob("sam3_*")
        ),
    ]
    existing = [path for path in generated if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Generated output already exists ({existing[0]}). Pass --overwrite to replace it."
        )
    if overwrite:
        for path in existing:
            path.unlink()
    updates_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    for directory in intermediate_subdirs:
        directory.mkdir(parents=True, exist_ok=True)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    return updates_dir, intermediate_dir, crop_dir


def load_frames_manifest(
    manifest_path: Path,
) -> tuple[Path, dict[str, Any], list[int], list[Path], float]:
    """Load an extracted image sequence without passing video to perception."""

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema") != "tablesim.image_sequence.v1":
        raise ValueError(
            "Unsupported frames manifest schema: "
            f"{data.get('schema')!r}"
        )
    metadata = data.get("source_video_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("frames manifest is missing source_video_metadata")
    source_video_value = data.get("source_video")
    if not isinstance(source_video_value, str) or not source_video_value:
        raise ValueError("frames manifest is missing source_video")
    source_video = Path(source_video_value).expanduser()
    if not source_video.is_absolute():
        source_video = (manifest_path.parent / source_video).resolve()
    if not source_video.exists():
        raise FileNotFoundError(source_video)

    sampling = data.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("frames manifest is missing sampling metadata")
    frequency_hz = float(sampling.get("frequency_hz", 0.0))
    if frequency_hz <= 0.0:
        raise ValueError("frames manifest frequency_hz must be positive")

    records = data.get("frames")
    if not isinstance(records, list) or not records:
        raise ValueError("frames manifest must contain a non-empty frames list")
    frame_indices: list[int] = []
    frame_paths: list[Path] = []
    for expected_sample_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("each frames manifest record must be an object")
        if int(record.get("sample_index", -1)) != expected_sample_index:
            raise ValueError("frames manifest sample_index values must be contiguous")
        frame_index = int(record["source_frame_index"])
        image_value = record.get("image")
        if not isinstance(image_value, str) or not image_value:
            raise ValueError("frames manifest record is missing image")
        image_path = Path(image_value).expanduser()
        if not image_path.is_absolute():
            image_path = (manifest_path.parent / image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        frame_indices.append(frame_index)
        frame_paths.append(image_path)

    declared_count = int(sampling.get("image_count", len(records)))
    if declared_count != len(records):
        raise ValueError(
            f"frames manifest image_count={declared_count}, "
            f"but contains {len(records)} records"
        )
    return source_video, metadata, frame_indices, frame_paths, frequency_hz


def crop_frame_sequence(
    source_paths: Sequence[Path],
    *,
    crop_dir: Path,
    normalized_roi: Sequence[float],
) -> tuple[list[Path], list[dict[str, Any]]]:
    cropped_paths: list[Path] = []
    crop_records: list[dict[str, Any]] = []
    for sample_index, source_path in enumerate(source_paths):
        output_path = (
            crop_dir
            / f"sam3_crop_{sample_index:04d}_{source_path.stem}.png"
        )
        metadata = crop_image_file(
            source_path,
            output_path,
            normalized_roi=normalized_roi,
        )
        metadata["sample_index"] = sample_index
        cropped_paths.append(output_path)
        crop_records.append(metadata)
    return cropped_paths, crop_records


def sam3_server_resource_path(
    local_path: Path,
    *,
    server_project_root: str | Path,
) -> str:
    """Translate a local TableSim path to the sidecar's read-only mount."""

    project_root = Path(__file__).resolve().parents[1]
    resolved = local_path.resolve()
    try:
        relative = resolved.relative_to(project_root)
    except ValueError:
        return str(resolved)
    return str(Path(server_project_root) / relative)


def save_sam3_intermediate(
    *,
    frame_path: Path,
    source_frame_path: Path,
    frame_index: int,
    update_index: int,
    timestamp: float,
    intermediate_dir: Path,
    object_result: Any,
    robot_result: Any,
    pre_window_result: Any,
    filtered_result: Any,
    crop_metadata: dict[str, Any],
    node_update_policy: dict[str, Any],
    admitted_new_object_ids: Sequence[str],
) -> dict[str, str]:
    """Save the SAM3 and label-window stages used by a graph update."""

    if (
        object_result is None
        or pre_window_result is None
        or filtered_result is None
    ):
        raise RuntimeError("SAM3 trace recorder did not capture perception results")
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read SAM3 input image: {frame_path}")
    rendered = render_detection_pipeline(
        image,
        object_result=object_result,
        robot_result=robot_result,
        filtered_result=filtered_result,
    )
    base = f"{update_index:04d}"
    paths = {
        "sam3_object_query_image": (
            intermediate_dir / "object_query" / f"sam3_object_query_{base}.png"
        ),
        "sam3_robot_query_image": (
            intermediate_dir / "robot_query" / f"sam3_robot_query_{base}.png"
        ),
        "sam3_filtered_image": (
            intermediate_dir / "filtered" / f"sam3_filtered_{base}.png"
        ),
        "sam3_pipeline_image": (
            intermediate_dir / "pipeline" / f"sam3_pipeline_{base}.png"
        ),
        "sam3_trace_json": (
            intermediate_dir / "json" / f"sam3_trace_{base}.json"
        ),
    }
    rendered_by_record_key = {
        "sam3_object_query_image": rendered["object_query"],
        "sam3_robot_query_image": rendered["robot_query"],
        "sam3_filtered_image": rendered["filtered"],
        "sam3_pipeline_image": rendered["pipeline"],
    }
    for key, visualization in rendered_by_record_key.items():
        if not cv2.imwrite(str(paths[key]), visualization):
            raise RuntimeError(f"Unable to write SAM3 intermediate image: {paths[key]}")

    robot_exclusion = pre_window_result.metadata.get("robot_exclusion", {})
    tracking_mode = filtered_result.metadata.get("tracking_mode")
    perception_stage = {
        "initial_image_detection": (
            "initial_image_detection_after_threshold_and_mask_clustering"
        ),
        "mask_memory_propagation": (
            "mask_memory_propagation_before_robot_exclusion"
        ),
        "periodic_detector_correction": (
            "periodic_image_detector_correction"
        ),
    }.get(
        tracking_mode,
        "object_query_after_threshold_and_mask_clustering",
    )
    trace = {
        "schema": "tablesim.sam3_detection_trace.v1",
        "update_index": update_index,
        "source_frame_index": frame_index,
        "timestamp_seconds": round(timestamp, 6),
        "input_image": str(frame_path),
        "source_uncropped_image": str(source_frame_path),
        "sam3_input_crop": crop_metadata,
        "temporal_perception": {
            "backend": filtered_result.metadata.get(
                "temporal_backend"
            ),
            "session_id": filtered_result.metadata.get(
                "tracking_session_id"
            ),
            "frame_index": filtered_result.metadata.get(
                "tracking_frame_index"
            ),
            "mode": filtered_result.metadata.get("tracking_mode"),
            "image_detector_ran": filtered_result.metadata.get(
                "image_detector_ran"
            ),
            "reanchored_object_ids": filtered_result.metadata.get(
                "reanchored_object_ids",
                [],
            ),
            "robot_memory_policy": filtered_result.metadata.get(
                "robot_memory_policy"
            ),
        },
        "label_sliding_window": filtered_result.metadata.get(
            "label_sliding_window",
            {},
        ),
        "stage_order": [
            "source_uncropped_image",
            "cropped_input_image",
            perception_stage,
            "robot_auxiliary_query",
            "scene_graph_input_after_robot_exclusion",
            "six_frame_label_sliding_window",
            "scene_graph_node_matching_and_update",
        ],
        "object_query_after_threshold_and_mask_clustering": result_audit(
            object_result
        ),
        # Backwards-compatible alias for older trace readers. The data now
        # comes from mask clustering, not suppressive cross-prompt NMS.
        "object_query_after_threshold_and_mask_nms": result_audit(
            object_result
        ),
        perception_stage: result_audit(object_result),
        "robot_auxiliary_query": result_audit(robot_result),
        "scene_graph_input_after_robot_exclusion": result_audit(
            pre_window_result
        ),
        "six_frame_label_sliding_window": {
            "result": result_audit(filtered_result),
            "audit": filtered_result.metadata.get(
                "label_sliding_window",
                {},
            ),
        },
        "robot_exclusion": robot_exclusion,
        "scene_graph_update": {
            "node_ids_before": node_update_policy.get("node_ids_before", []),
            "node_ids_after": node_update_policy.get("node_ids_after", []),
            "matched_node_ids": node_update_policy.get(
                "current_observation_object_ids",
                [],
            ),
            "retained_without_current_detection_ids": (
                node_update_policy.get(
                    "retained_without_current_detection_ids",
                    [],
                )
            ),
            "ignored_unmatched_detections": node_update_policy.get(
                "ignored_unmatched_detections",
                [],
            ),
            "initialization_confirmed_new_object_ids": list(
                admitted_new_object_ids
            ),
            "multi_frame_initialization_used": True,
            "node_locking_used": True,
            "object_addition_validator_used": False,
            "legacy_initializer_and_validator_code_retained": True,
            "removed_object_ids": node_update_policy.get(
                "removed_object_ids",
                [],
            ),
        },
        "visualizations": {
            key: str(path)
            for key, path in paths.items()
            if key != "sam3_trace_json"
        },
    }
    paths["sam3_trace_json"].write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        key: str(path)
        for key, path in paths.items()
    }


def run_scene_graph_updates(
    *,
    frame_paths: Sequence[Path],
    source_frame_paths: Sequence[Path],
    crop_records: Sequence[dict[str, Any]],
    frame_indices: Sequence[int],
    fps: float,
    video_path: Path,
    updates_dir: Path,
    intermediate_dir: Path,
    prompts: Sequence[str],
    robot_prompts: Sequence[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sequence_lengths = {
        len(frame_paths),
        len(source_frame_paths),
        len(crop_records),
        len(frame_indices),
    }
    if len(sequence_lengths) != 1:
        raise ValueError(
            "cropped frames, source frames, crop records and frame indices "
            "must have the same length"
        )
    object_client = Sam3PerceptionClient(
        endpoint=args.sam3_endpoint,
        prompt_catalog=prompts,
        prompt_catalog_path=None,
        max_prompts=len(prompts),
        confidence_threshold=args.threshold,
        max_detections=args.max_detections,
        min_mask_area=args.min_mask_area,
        timeout=args.sam3_timeout,
    )
    health = object_client.health(timeout=min(10.0, args.sam3_timeout))
    if not health.get("model_loaded"):
        object_client.load_model()

    object_recorder = RecordingPerceptionBackend(object_client)
    robot_recorder: RecordingPerceptionBackend | None = None
    temporal_backend: Sam3VideoMemoryPerceptionBackend | None = None
    temporal_close_registered = False
    backend: Any = object_recorder
    if args.robot_exclusion:
        robot_recorder = RecordingPerceptionBackend(
            Sam3PerceptionClient(
                endpoint=args.sam3_endpoint,
                prompt_catalog=robot_prompts,
                prompt_catalog_path=None,
                max_prompts=len(robot_prompts),
                confidence_threshold=args.threshold,
                max_detections=args.max_detections,
                min_mask_area=args.min_mask_area,
                timeout=args.sam3_timeout,
            )
        )
        if args.sam3_video_memory:
            temporal_backend = Sam3VideoMemoryPerceptionBackend(
                image_backend=object_recorder,
                robot_backend=robot_recorder,
                tracking_client=Sam3TrackingClient(
                    endpoint=args.sam3_endpoint,
                    timeout=args.sam3_timeout,
                ),
                resource_path=sam3_server_resource_path(
                    frame_paths[0].parent,
                    server_project_root=args.sam3_server_project_root,
                ),
                redetect_interval=args.sam3_redetect_interval,
                tracker_match_iou=args.sam3_tracker_match_iou,
                robot_match_iou=args.robot_match_iou,
                robot_mask_containment=args.robot_mask_containment,
                min_mask_area=args.min_mask_area,
                new_object_memory_policy=(
                    "add_after_multi_frame_initialization_confirmation_"
                    "without_inventory_validator"
                ),
            )
            atexit.register(temporal_backend.close)
            temporal_close_registered = True
            backend = temporal_backend
        else:
            backend = RobotExclusionPerceptionBackend(
                object_recorder,
                robot_recorder,
                match_iou_threshold=args.robot_match_iou,
                mask_containment_threshold=args.robot_mask_containment,
            )
    elif args.sam3_video_memory:
        raise ValueError(
            "--sam3-video-memory requires --robot-exclusion so robot masks "
            "remain exclusion-only and never enter object memory"
        )
    label_corrector = LabelSlidingWindowPerceptionBackend(
        backend,
        window_size=args.label_window_size,
        max_center_distance=args.label_window_match_distance,
        min_bbox_iou=args.label_window_min_bbox_iou,
        min_size_similarity=args.label_window_min_size_similarity,
    )
    filtered_recorder = RecordingPerceptionBackend(label_corrector)

    constructor = SceneGraphConstructor(
        perception_backend=filtered_recorder,
        visibility_from_image_boundary=not args.robot_exclusion,
    )
    updater = SceneGraphUpdater(
        constructor=constructor,
        visibility_policy=args.visibility_policy,
    )
    max_gap_observations = max(
        1,
        int(math.ceil(args.initialization_max_gap / args.update_interval)),
    )
    initializer = MultiFrameSceneGraphInitializer(
        min_observations=args.initialization_min_observations,
        min_confidence=args.initialization_min_confidence,
        max_position_distance=args.initialization_match_distance,
        max_gap_observations=max_gap_observations,
    )
    current_graph = None
    snapshots: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for update_index, (
        frame_index,
        frame_path,
        source_frame_path,
        crop_metadata,
    ) in enumerate(
        zip(
            frame_indices,
            frame_paths,
            source_frame_paths,
            crop_records,
        )
    ):
        timestamp = frame_index / fps
        initialization_active = (
            args.initialization_window > 0.0
            and timestamp <= args.initialization_window + 1e-9
        )
        confirmed_new_object_ids: list[str] = []
        observation_graph = None
        if current_graph is None:
            current_graph = constructor.construct(
                frame_path,
                instruction=args.instruction,
            )
            current_graph = updater.apply_visibility_policy(current_graph)
            initial_node_ids = sorted(obj.id for obj in current_graph.objects)
            for obj in current_graph.objects:
                obj.state = updater.state_without_invisible(obj.state)
                obj.metadata = {
                    **obj.metadata,
                    "observed_in_current_frame": True,
                    "initial_scene_graph_node": True,
                }
            current_graph.metadata["node_update_policy"] = {
                "mode": (
                    "multi_frame_initialization"
                    if initialization_active
                    else "fixed_initial_node_set"
                ),
                "initialization_frame": True,
                "initialization_active": initialization_active,
                "multi_frame_initialization_enabled": True,
                "node_locking_enabled": True,
                "object_addition_validator_enabled": False,
                "object_addition_validator_code_retained": True,
                "nodes_can_be_added": initialization_active,
                "nodes_can_be_removed": False,
                "node_ids_before": [],
                "node_ids_after": initial_node_ids,
                "node_ids_unchanged": True,
                "current_observation_object_ids": initial_node_ids,
                "initial_node_ids": initial_node_ids,
                "new_object_ids": [],
                "retained_without_current_detection_ids": [],
                "retained_not_observed_object_ids": [],
                "ignored_unmatched_detection_count": 0,
                "ignored_unmatched_detections": [],
                "removed_object_ids": [],
            }
            if temporal_backend is not None:
                seed_response = temporal_backend.seed_scene_graph(
                    current_graph
                )
                current_graph.metadata["sam3_video_memory_seed"] = {
                    "session_id": seed_response["session_id"],
                    "frame_index": 0,
                    "seeded_object_count": seed_response[
                        "seeded_object_count"
                    ],
                    "stable_to_track": seed_response["stable_to_track"],
                }
            observation_graph = current_graph.copy()
        else:
            previous_node_ids = sorted(
                obj.id for obj in current_graph.objects
            )
            observation_graph = constructor.construct(
                frame_path,
                instruction=args.instruction,
            )
            provisional_graph = updater.merge(
                current_graph,
                observation_graph.copy(),
            )
            confirmations = []
            if initialization_active:
                ignored_ids = {
                    item["temporary_id"]
                    for item in provisional_graph.metadata[
                        "node_update_policy"
                    ].get("ignored_unmatched_detections", [])
                }
                ignored_objects = [
                    SceneObject.from_dict(obj.to_dict())
                    for obj in observation_graph.objects
                    if obj.id in ignored_ids
                ]
                confirmations = initializer.observe(
                    ignored_objects,
                    observation_index=update_index,
                )

            if confirmations:
                expanded_graph = current_graph.copy()
                existing_ids = [obj.id for obj in expanded_graph.objects]
                for confirmation in confirmations:
                    confirmed = SceneObject.from_dict(
                        confirmation.object.to_dict()
                    )
                    confirmed.id = unique_id(confirmed.id, existing_ids)
                    existing_ids.append(confirmed.id)
                    confirmed.metadata = {
                        **confirmed.metadata,
                        "initialization_confirmed_node": True,
                        "initialization_confirmation": (
                            confirmation.to_metadata()
                        ),
                        "added_without_object_inventory_validation": True,
                    }
                    expanded_graph.add_object(confirmed)
                    confirmed_new_object_ids.append(confirmed.id)
                current_graph = updater.merge(
                    expanded_graph,
                    observation_graph,
                )
            else:
                current_graph = provisional_graph

            policy = current_graph.metadata["node_update_policy"]
            policy.update(
                {
                    "mode": (
                        "multi_frame_initialization"
                        if initialization_active
                        else "fixed_after_initialization"
                    ),
                    "initialization_frame": False,
                    "initialization_active": initialization_active,
                    "multi_frame_initialization_enabled": True,
                    "node_locking_enabled": True,
                    "object_addition_validator_enabled": False,
                    "object_addition_validator_code_retained": True,
                    "nodes_can_be_added": initialization_active,
                    "node_ids_before": previous_node_ids,
                    "node_ids_after": sorted(
                        obj.id for obj in current_graph.objects
                    ),
                    "node_ids_unchanged": not confirmed_new_object_ids,
                    "new_object_ids": sorted(
                        confirmed_new_object_ids
                    ),
                    "confirmed_new_object_ids": sorted(
                        confirmed_new_object_ids
                    ),
                }
            )
            if temporal_backend is not None and confirmed_new_object_ids:
                memory_addition = temporal_backend.add_graph_objects(
                    current_graph,
                    confirmed_new_object_ids,
                    admission_reason=(
                        "multi_frame_initialization_confirmation_without_"
                        "inventory_validation"
                    ),
                )
                policy["sam3_video_memory_new_object_addition"] = (
                    memory_addition
                )

        if current_graph.metadata.get("source") != "sam3":
            raise RuntimeError(
                "Strict SAM3 image-sequence inference expected metadata.source='sam3', "
                f"got {current_graph.metadata.get('source')!r}"
            )

        current_graph.metadata["scene_graph_initialization"] = {
            "mode": "multi_frame_confirmation_then_fixed",
            "active": initialization_active,
            "window_seconds": args.initialization_window,
            "min_observations": args.initialization_min_observations,
            "min_confidence": args.initialization_min_confidence,
            "max_position_distance": args.initialization_match_distance,
            "max_gap_seconds": args.initialization_max_gap,
            "max_gap_observations": max_gap_observations,
            "multi_frame_initializer_used": True,
            "node_locking_used": True,
            "object_addition_validator_used": False,
            "legacy_code_retained": True,
            "label_correction": "six_frame_label_sliding_window",
            "confirmed_new_object_ids_this_update": sorted(
                confirmed_new_object_ids
            ),
            "confirmed_track_count_total": initializer.confirmed_track_count,
            "pending_candidate_tracks": initializer.pending_tracks(),
            "rejected_low_confidence_detection_count_total": (
                initializer.rejected_low_confidence_count
            ),
            "expired_candidate_track_count_total": (
                initializer.expired_track_count
            ),
        }
        current_graph.metadata.update(
            {
                "video_source": str(video_path),
                "video_update_index": update_index,
                "video_frame_index": frame_index,
                "video_timestamp_seconds": round(timestamp, 6),
                "sam3_input_crop": crop_metadata,
                "relation_edge_value": dict(
                    RELATION_EDGE_VALUE_SPEC
                ),
            }
        )
        snapshot = current_graph.to_dict()
        snapshots.append(snapshot)

        json_name = f"scene_graph_update_{update_index:04d}.json"
        png_name = f"scene_graph_update_{update_index:04d}.png"
        (updates_dir / json_name).write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        visible_objects = list(snapshot["objects"])
        policy = snapshot["metadata"].get("node_update_policy", {})
        observed_ids = set(policy.get("current_observation_object_ids", []))
        observed_objects = [
            obj for obj in snapshot["objects"] if obj["id"] in observed_ids
        ]
        corrected_detection_labels = Counter(
            obj.name for obj in (observation_graph or current_graph).objects
        )
        pre_window_result = label_corrector.last_raw_result
        pre_window_detection_labels = Counter(
            detection.label
            for detection in (
                pre_window_result.detections
                if pre_window_result is not None
                else []
            )
        )
        perception = snapshot["metadata"].get("object_level_perception", {})
        label_window = perception.get("label_sliding_window", {})
        robot_exclusion = perception.get("robot_exclusion", {})
        visibility_policy = snapshot["metadata"].get("visibility_policy", {})
        intermediate_paths = save_sam3_intermediate(
            frame_path=frame_path,
            source_frame_path=source_frame_path,
            frame_index=frame_index,
            update_index=update_index,
            timestamp=timestamp,
            intermediate_dir=intermediate_dir,
            object_result=(
                temporal_backend.last_image_detector_result
                if temporal_backend is not None
                else object_recorder.last_result
            ),
            robot_result=(
                robot_recorder.last_result
                if robot_recorder is not None
                else None
            ),
            pre_window_result=pre_window_result,
            filtered_result=filtered_recorder.last_result,
            crop_metadata=crop_metadata,
            node_update_policy=policy,
            admitted_new_object_ids=confirmed_new_object_ids,
        )
        record = {
            "update_index": update_index,
            "graph_version": snapshot["version"],
            "frame_index": frame_index,
            "timestamp_seconds": round(timestamp, 6),
            "source_image": str(source_frame_path),
            "observation_image": str(frame_path),
            "scene_graph_json": str(updates_dir / json_name),
            "scene_graph_image": str(updates_dir / png_name),
            "object_count": len(snapshot["objects"]),
            "object_ids": [obj["id"] for obj in snapshot["objects"]],
            "visible_object_count": len(visible_objects),
            "current_observation_object_count": len(observed_objects),
            "relation_count": len(snapshot["relations"]),
            "detected_labels": dict(
                sorted(Counter(obj["name"] for obj in observed_objects).items())
            ),
            "matched_existing_node_labels": dict(
                sorted(Counter(obj["name"] for obj in observed_objects).items())
            ),
            "sam3_detection_labels_after_robot_filter": dict(
                sorted(pre_window_detection_labels.items())
            ),
            "sam3_detection_labels_after_sliding_window": dict(
                sorted(corrected_detection_labels.items())
            ),
            "initialization_confirmed_new_object_ids": sorted(
                confirmed_new_object_ids
            ),
            "label_sliding_window_active": bool(
                label_window.get("active")
            ),
            "label_sliding_window_corrected_count": int(
                label_window.get("corrected_detection_count", 0)
            ),
            "label_sliding_window_corrections": label_window.get(
                "corrections",
                [],
            ),
            "initialization_active": initialization_active,
            "initialization_proposed_new_object_ids": [],
            "initialization_pending_candidate_count": len(
                initializer.pending_tracks()
            ),
            "object_addition_validation_attempted": False,
            "object_addition_validation_passed": None,
            "object_addition_validation_failed_check": None,
            "inference_seconds": perception.get("inference_seconds"),
            "perception_source": snapshot["metadata"].get("source"),
            "robot_auxiliary_detection_count": robot_exclusion.get(
                "auxiliary_robot_detection_count", 0
            ),
            "robot_filtered_object_detection_count": robot_exclusion.get(
                "filtered_object_detection_count", 0
            ),
            "visibility_corrected_to_full_count": len(
                visibility_policy.get(
                    "corrected_to_fully_visible_object_ids", []
                )
            ),
            "visibility_partial_due_to_on_count": len(
                visibility_policy.get("partial_due_to_incoming_on", {})
            ),
            **intermediate_paths,
        }
        if temporal_backend is not None:
            record.update(
                {
                    "sam3_tracking_session_id": (
                        temporal_backend.session_id
                    ),
                    "sam3_tracking_mode": perception.get(
                        "tracking_mode"
                    ),
                    "sam3_image_detector_ran": perception.get(
                        "image_detector_ran"
                    ),
                    "sam3_reanchored_object_ids": perception.get(
                        "reanchored_object_ids",
                        [],
                    ),
                }
            )
        records.append(record)
        print(
            f"[{update_index + 1:02d}/{len(frame_paths):02d}] "
            f"t={timestamp:05.2f}s objects={len(snapshot['objects'])} "
            f"phase=sliding-window "
            f"window_active={bool(label_window.get('active'))} "
            f"label_corrected={label_window.get('corrected_detection_count', 0)} "
            f"node_phase={'init' if initialization_active else 'fixed'} "
            f"confirmed_new={len(confirmed_new_object_ids)} "
            f"visible={len(visible_objects)} relations={len(snapshot['relations'])} "
            f"inference={perception.get('inference_seconds', '?')}s",
            flush=True,
        )
    if temporal_backend is not None:
        temporal_backend.close()
        if temporal_close_registered:
            atexit.unregister(temporal_backend.close)
    return snapshots, records


def stable_node_order(snapshots: Sequence[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    known: set[str] = set()
    for snapshot in snapshots:
        for obj in snapshot.get("objects", []):
            object_id = obj["id"]
            if object_id not in known:
                ordered.append(object_id)
                known.add(object_id)
    return ordered


def stable_positions(node_order: Sequence[str]) -> dict[str, tuple[float, float]]:
    count = max(1, len(node_order))
    columns = max(1, math.ceil(math.sqrt(count * 1.35)))
    rows = math.ceil(count / columns)
    xs = np.linspace(0.11, 0.89, columns)
    ys = np.linspace(0.77, 0.23, rows)
    return {
        node_id: (float(xs[index % columns]), float(ys[index // columns]))
        for index, node_id in enumerate(node_order)
    }


def short_id(object_id: str, max_chars: int = 20) -> str:
    if len(object_id) <= max_chars:
        return object_id
    return object_id[: max_chars - 1] + "…"


def node_label(obj: dict[str, Any]) -> str:
    attrs = obj.get("attributes", {})
    visibility = str(attrs.get("visibility", "?")).replace("_visible", "")
    state = str(obj.get("state", ""))
    if len(state) > 27:
        state = state[:26] + "…"
    xy = attrs.get("xy")
    footprint = attrs.get("footprint")
    xy_text = (
        f"xy ({float(xy[0]):.2f},{float(xy[1]):.2f})"
        if isinstance(xy, list) and len(xy) >= 2
        else "xy (?,?)"
    )
    footprint_text = (
        f"fp ({float(footprint[0]):.2f},{float(footprint[1]):.2f})"
        if isinstance(footprint, list) and len(footprint) >= 2
        else "fp (?,?)"
    )
    return (
        f"{obj.get('name', 'object')}\n"
        f"{short_id(str(obj.get('id', '')))}\n"
        f"{state}\n"
        f"{xy_text}\n"
        f"{footprint_text}\n"
        f"C {float(attrs.get('confidence', 0.0)):.2f} | V {visibility}\n"
        f"rigid {float(attrs.get('rigidity', 0.0)):.2f} | "
        f"fragile {float(attrs.get('fragility', 0.0)):.2f}\n"
        f"movable {'Y' if attrs.get('movable') else 'N'} | "
        f"portable {'Y' if attrs.get('portable') else 'N'}"
    )


def node_color(obj: dict[str, Any]) -> str:
    if obj.get("attributes", {}).get("visibility") == "partially_visible":
        return "#ffd59a"
    return "#b9ddff"


def select_relations(
    scene_graph: dict[str, Any],
    *,
    max_relations: int,
) -> tuple[list[dict[str, Any]], int]:
    relations = scene_graph.get("relations", [])
    ranked = sorted(
        relations,
        key=lambda rel: (
            RELATION_PRIORITY.get(rel.get("relation"), 9),
            -float(rel.get("confidence", 0.0)),
            str(rel.get("source")),
            str(rel.get("target")),
        ),
    )
    selected: list[dict[str, Any]] = []
    directional_pairs: set[tuple[str, str]] = set()
    for relation in ranked:
        name = str(relation.get("relation"))
        pair = tuple(sorted((str(relation.get("source")), str(relation.get("target")))))
        if name not in SEMANTIC_RELATIONS and pair in directional_pairs:
            continue
        selected.append(relation)
        if name not in SEMANTIC_RELATIONS:
            directional_pairs.add(pair)
        if len(selected) >= max(0, max_relations):
            break
    return selected, max(0, len(relations) - len(selected))


def draw_relation(
    ax: Any,
    relation: dict[str, Any],
    positions: dict[str, tuple[float, float]],
    relation_index: int,
) -> None:
    source = positions.get(str(relation.get("source")))
    target = positions.get(str(relation.get("target")))
    if source is None or target is None or source == target:
        return
    relation_name = str(relation.get("relation", "related_to"))
    color = RELATION_COLORS.get(relation_name, "#78909c")
    important = relation_name in SEMANTIC_RELATIONS
    curvature = 0.10 if relation_index % 2 == 0 else -0.10
    arrow = FancyArrowPatch(
        source,
        target,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=10 if important else 8,
        linewidth=1.5 if important else 0.8,
        color=color,
        alpha=0.72 if important else 0.38,
        connectionstyle=f"arc3,rad={curvature}",
        shrinkA=32,
        shrinkB=32,
        zorder=1,
    )
    ax.add_patch(arrow)

    middle_x = (source[0] + target[0]) / 2.0
    middle_y = (source[1] + target[1]) / 2.0 + (0.013 if curvature > 0 else -0.013)
    display_relation = relation_name.replace("_image", "")
    direction = relation.get("direction")
    if relation_name == "next_to" and direction in {"left", "right"}:
        display_relation = f"next_to_{direction}"
    distance = relation.get("distance")
    if isinstance(distance, list) and len(distance) >= 2:
        label = (
            f"{display_relation}: "
            f"({float(distance[0]):.3f}, {float(distance[1]):.3f})"
        )
    else:
        label = display_relation
    ax.text(
        middle_x,
        middle_y,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=4.7 if important else 4.4,
        color=color,
        alpha=0.92 if important else 0.72,
        bbox={
            "boxstyle": "round,pad=0.08",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.72,
        },
        zorder=2,
    )


def render_scene_graph_updates(
    snapshots: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    *,
    updates_dir: Path,
    width: int,
    height: int,
    dpi: int,
    max_relations: int,
) -> list[Path]:
    node_order = stable_node_order(snapshots)
    positions = stable_positions(node_order)
    columns = max(1, math.ceil(math.sqrt(max(1, len(node_order)) * 1.35)))
    rows = max(1, math.ceil(max(1, len(node_order)) / columns))
    box_width = min(0.21, 0.76 / columns)
    box_height = min(0.145, 0.48 / rows)
    image_paths: list[Path] = []

    for snapshot, record in zip(snapshots, records):
        fig = plt.figure(
            figsize=(width / dpi, height / dpi),
            dpi=dpi,
            facecolor="#fbfcfd",
        )
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")

        selected_relations, omitted = select_relations(
            snapshot,
            max_relations=max_relations,
        )
        for relation_index, relation in enumerate(selected_relations):
            draw_relation(ax, relation, positions, relation_index)

        objects = {obj["id"]: obj for obj in snapshot.get("objects", [])}
        for object_id in node_order:
            obj = objects.get(object_id)
            if obj is None:
                continue
            x, y = positions[object_id]
            box = FancyBboxPatch(
                (x - box_width / 2.0, y - box_height / 2.0),
                box_width,
                box_height,
                transform=ax.transAxes,
                boxstyle="round,pad=0.008",
                facecolor=node_color(obj),
                edgecolor="#34495e",
                linewidth=1.0,
                zorder=3,
            )
            ax.add_patch(box)
            ax.text(
                x,
                y,
                node_label(obj),
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=5.8,
                color="#17202a",
                linespacing=1.05,
                zorder=4,
            )

        metadata = snapshot.get("metadata", {})
        perception = metadata.get("object_level_perception", {})
        robot_exclusion = perception.get("robot_exclusion", {})
        visibility_policy = metadata.get("visibility_policy", {})
        ax.text(
            0.5,
            0.965,
            (
                f"Incremental Scene Graph  |  Update {record['update_index']:02d}  "
                f"|  t = {record['timestamp_seconds']:.2f} s"
            ),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=14,
            weight="bold",
            color="#1f2d3d",
        )
        ax.text(
            0.5,
            0.915,
            (
                f"source: {metadata.get('source')}   version: {snapshot.get('version')}   "
                f"objects: {record['object_count']} "
                f"{'initializing' if record.get('initialization_active') else 'fixed'}, "
                f"{record['current_observation_object_count']} observed this frame   "
                f"confirmed new: "
                f"{len(record.get('initialization_confirmed_new_object_ids', []))}   "
                f"label corrections: "
                f"{record.get('label_sliding_window_corrected_count', 0)}   "
                f"relations: {len(selected_relations)} shown / "
                f"{record['relation_count']} total ({omitted} omitted)   "
                f"inference: {perception.get('inference_seconds', '?')} s"
            ),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.3,
            color="#4d5656",
        )
        if visibility_policy.get("post_update_mode") == ON_RELATION_VISIBILITY_POLICY:
            ax.text(
                0.5,
                0.878,
                (
                    "Visibility: fully/partially only; robot occlusion ignored  |  "
                    "corrected to fully "
                    f"visible: {len(visibility_policy.get('corrected_to_fully_visible_object_ids', []))}  "
                    "|  partial iff object --on--> target"
                ),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=6.8,
                color="#6c3483",
                weight="bold",
            )
        elif robot_exclusion.get("enabled"):
            ax.text(
                0.5,
                0.878,
                (
                    "Robot influence excluded  |  auxiliary detections: "
                    f"{robot_exclusion.get('auxiliary_robot_detection_count', 0)}  "
                    "|  robot-shaped object detections removed: "
                    f"{robot_exclusion.get('filtered_object_detection_count', 0)}  "
                    "|  partial visibility: scene-object overlap only"
                ),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=6.8,
                color="#1e8449",
                weight="bold",
            )

        ax.text(
            0.025,
            0.055,
            "  fully visible  ",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.5,
            bbox={"boxstyle": "round,pad=0.24", "fc": "#b9ddff", "ec": "#34495e"},
        )
        ax.text(
            0.155,
            0.055,
            "  partially visible  ",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.5,
            bbox={"boxstyle": "round,pad=0.24", "fc": "#ffd59a", "ec": "#34495e"},
        )
        ax.text(
            0.975,
            0.055,
            "Topological graph view (no spatial-layout panel)",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=6.5,
            color="#5d6d7e",
        )

        output_path = updates_dir / f"scene_graph_update_{record['update_index']:04d}.png"
        fig.savefig(
            output_path,
            dpi=dpi,
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        image_paths.append(output_path)
    return image_paths


def fit_panel(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    panel[y : y + resized_height, x : x + resized_width] = resized
    return panel


def draw_video_label(frame: np.ndarray, timestamp: float) -> None:
    cv2.rectangle(frame, (18, 16), (308, 64), (22, 31, 42), thickness=-1)
    cv2.putText(
        frame,
        "Operation video",
        (31, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"t = {timestamp:05.2f} s",
        (178, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (208, 224, 240),
        1,
        cv2.LINE_AA,
    )


def compose_video(
    *,
    video_path: Path,
    output_path: Path,
    graph_image_paths: Sequence[Path],
    frame_indices: Sequence[int],
    fps: float,
    output_height: int,
    graph_width: int,
    source_aspect: float,
) -> None:
    left_width = int(round(output_height * source_aspect))
    if left_width % 2:
        left_width += 1
    if graph_width % 2:
        graph_width += 1
    total_width = left_width + graph_width
    if output_height % 2:
        output_height += 1

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{total_width}x{output_height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    encoder = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if encoder.stdin is None or encoder.stderr is None:
        raise RuntimeError("Unable to open ffmpeg encoder pipes")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        encoder.kill()
        raise RuntimeError(f"Unable to open source video: {video_path}")

    graph_cache: dict[int, np.ndarray] = {}
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            update_index = max(0, bisect.bisect_right(frame_indices, frame_index) - 1)
            graph_image = graph_cache.get(update_index)
            if graph_image is None:
                graph_image = cv2.imread(str(graph_image_paths[update_index]))
                if graph_image is None:
                    raise RuntimeError(
                        f"Unable to read graph image: {graph_image_paths[update_index]}"
                    )
                graph_image = cv2.resize(
                    graph_image,
                    (graph_width, output_height),
                    interpolation=cv2.INTER_AREA,
                )
                graph_cache = {update_index: graph_image}

            left_panel = fit_panel(frame, left_width, output_height)
            draw_video_label(left_panel, frame_index / fps)
            composite = np.concatenate([left_panel, graph_image], axis=1)
            cv2.line(
                composite,
                (left_width, 0),
                (left_width, output_height),
                (70, 70, 70),
                thickness=3,
            )
            encoder.stdin.write(composite.tobytes())
            frame_index += 1
    finally:
        capture.release()
        encoder.stdin.close()

    stderr = encoder.stderr.read().decode("utf-8", errors="replace")
    return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr}")


def relative_artifact(path: Path, output_dir: Path) -> str:
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def write_manifest(
    *,
    output_dir: Path,
    video_path: Path,
    output_video: Path,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
    prompts: Sequence[str],
    frames_manifest_path: Path,
    args: argparse.Namespace,
) -> Path:
    crop_roi = parse_normalized_crop_roi(args.sam3_crop_roi)
    crop_bbox = pixel_crop_bbox(
        [metadata["width"], metadata["height"]],
        crop_roi,
    )
    normalized_records = []
    for record in records:
        item = dict(record)
        for key in (
            "source_image",
            "observation_image",
            "scene_graph_json",
            "scene_graph_image",
            "sam3_object_query_image",
            "sam3_robot_query_image",
            "sam3_filtered_image",
            "sam3_pipeline_image",
            "sam3_trace_json",
        ):
            item[key] = relative_artifact(Path(item[key]), output_dir)
        normalized_records.append(item)

    confirmed_initialization_ids = [
        object_id
        for record in normalized_records
        for object_id in record.get(
            "initialization_confirmed_new_object_ids",
            [],
        )
    ]
    first_fixed_record = next(
        (
            record
            for record in normalized_records
            if not record.get("initialization_active", False)
        ),
        None,
    )
    label_correction_count = sum(
        int(record.get("label_sliding_window_corrected_count", 0))
        for record in normalized_records
    )
    label_correction_frame_count = sum(
        int(record.get("label_sliding_window_corrected_count", 0)) > 0
        for record in normalized_records
    )
    manifest = {
        "source_video": str(video_path),
        "output_video": relative_artifact(output_video, output_dir),
        "perception_backend": "sam3",
        "temporal_perception_backend": (
            "sam3_video_memory"
            if args.sam3_video_memory
            else "independent_sam3_images"
        ),
        "spatial_graph_generated": False,
        "reasoning_graph_generated": False,
        "sam3_intermediate_saved": True,
        "video": metadata,
        "frame_input": {
            "mode": "image_sequence",
            "frames_manifest": str(frames_manifest_path),
            "sampling_frequency_hz": 1.0 / args.update_interval,
            "sampling_interval_seconds": args.update_interval,
            "image_count": len(records),
            "perception_input_type": "png_image",
            "source_video_used_for_perception": False,
            "all_png_images_share_one_tracking_session": (
                args.sam3_video_memory
            ),
            "sam3_input_crop": {
                "enabled": True,
                "applied_before_sam3": True,
                "normalized_roi": list(crop_roi),
                "source_size": [metadata["width"], metadata["height"]],
                "pixel_bbox": list(crop_bbox),
                "cropped_size": [
                    crop_bbox[2] - crop_bbox[0],
                    crop_bbox[3] - crop_bbox[1],
                ],
                "coordinate_frame": "cropped_image",
            },
        },
        "update_config": {
            "update_interval_seconds": args.update_interval,
            "update_count": len(records),
            "sam3_video_memory": {
                "enabled": args.sam3_video_memory,
                "session_api": (
                    "/v1/tracking/sessions/*"
                    if args.sam3_video_memory
                    else None
                ),
                "input": "ordered_5hz_png_sequence",
                "first_frame": (
                    "semantic_detection_then_mask_to_stable_scene_graph_id"
                ),
                "later_frames": "sam3_mask_memory_propagation",
                "image_redetection_interval_frames": (
                    args.sam3_redetect_interval
                ),
                "image_redetection_interval_seconds": round(
                    args.sam3_redetect_interval * args.update_interval,
                    6,
                ),
                "tracker_detector_mask_iou_threshold": (
                    args.sam3_tracker_match_iou
                ),
                "periodic_redetection_purpose": (
                    "correct_tracker_drift_and_semantic_label"
                ),
                "robot_mask_policy": (
                    "exclusion_only_never_seed_reanchor_or_object_memory"
                ),
                "new_object_memory_policy": (
                    "add_after_multi_frame_initialization_confirmation_"
                    "without_inventory_validator"
                ),
                "scene_graph_node_policy": (
                    "multi_frame_initialization_then_fixed_nodes_never_deleted"
                ),
            },
            "label_sliding_window": {
                "enabled": True,
                "window_size_frames": args.label_window_size,
                "current_frame_count": 1,
                "preceding_frame_count": args.label_window_size - 1,
                "first_eligible_update_index": args.label_window_size - 1,
                "early_frame_policy": (
                    "do_not_correct_until_five_preceding_frames_exist"
                ),
                "matching_order": [
                    "stable_scene_object_id",
                    "bbox_iou_or_normalized_center_distance_with_size_similarity",
                ],
                "max_normalized_center_distance": (
                    args.label_window_match_distance
                ),
                "minimum_bbox_iou": args.label_window_min_bbox_iou,
                "minimum_bbox_size_similarity": (
                    args.label_window_min_size_similarity
                ),
                "vote_source": "labels_at_same_position_in_preceding_five_frames",
                "winner": "unique_highest_frequency_label",
                "different_current_label_action": (
                    "replace_current_label_with_history_winner"
                ),
                "tie_action": "keep_current_label",
                "history_uses_corrected_labels": False,
                "history_uses_raw_sam3_labels": True,
                "corrected_detection_count": label_correction_count,
                "frames_with_corrections": label_correction_frame_count,
            },
            "instruction": args.instruction,
            "query_catalog": {
                "source": (
                    "command_line"
                    if args.prompts.strip()
                    else str(args.prompts_file)
                ),
                "query_count": len(prompts),
                "query_limit": args.max_prompts,
            },
            "detected_labels_semantics": (
                "backward-compatible alias of current_observation_node_labels"
            ),
            "robot_exclusion": {
                "enabled": args.robot_exclusion,
                "object_match_bbox_iou_threshold": args.robot_match_iou,
                "object_mask_containment_threshold": (
                    args.robot_mask_containment
                ),
                "auxiliary_detections_added_to_scene_graph": False,
                "image_boundary_counts_as_partial_visibility": (
                    not args.robot_exclusion
                ),
                "partial_visibility_sources": (
                    ["object_bbox_overlap"]
                    if args.robot_exclusion
                    else ["object_bbox_overlap", "image_boundary"]
                ),
            },
            "visibility_policy": {
                "mode": args.visibility_policy,
                "allowed_values": ["fully_visible", "partially_visible"],
                "incoming_relation": "other_object --on--> target_object",
                "without_incoming_on": "attributes.visibility=fully_visible",
                "invisible_state_allowed": False,
                "robot_occlusion_counts_as_partial_visibility": False,
            },
            "node_update_policy": {
                "mode": "multi_frame_initialization_then_fixed",
                "multi_frame_initialization": {
                    "enabled": True,
                    "used_by_video_pipeline": True,
                    "code_retained": True,
                    "window_seconds": args.initialization_window,
                    "minimum_observations": (
                        args.initialization_min_observations
                    ),
                    "minimum_confidence": (
                        args.initialization_min_confidence
                    ),
                    "maximum_position_distance": (
                        args.initialization_match_distance
                    ),
                    "maximum_gap_seconds": (
                        args.initialization_max_gap
                    ),
                },
                "node_locking": {
                    "enabled": True,
                    "used_by_video_pipeline": True,
                    "fixed_node_set": True,
                    "fixed_from_update_index": (
                        first_fixed_record["update_index"]
                        if first_fixed_record is not None
                        else None
                    ),
                    "fixed_from_timestamp_seconds": (
                        first_fixed_record["timestamp_seconds"]
                        if first_fixed_record is not None
                        else None
                    ),
                },
                "initial_object_count": (
                    normalized_records[0]["object_count"]
                    if normalized_records
                    else 0
                ),
                "confirmed_initialization_object_ids": (
                    confirmed_initialization_ids
                ),
                "final_object_count": (
                    normalized_records[-1]["object_count"]
                    if normalized_records
                    else 0
                ),
                "new_object_addition_validation": {
                    "enabled": False,
                    "used_by_video_pipeline": False,
                    "code_retained": True,
                    "check_order": [
                        "object_count",
                        "object_types",
                        "per_type_counts",
                    ],
                    "attempt_count": 0,
                    "pass_count": 0,
                    "rejection_count": 0,
                },
                "nodes_can_be_added_during_initialization": True,
                "nodes_can_be_added_after_initialization": False,
                "nodes_can_be_removed": False,
                "unmatched_current_detections": (
                    "multi_frame_confirmed_during_initialization_"
                    "otherwise_ignored_with_audit"
                ),
                "temporarily_unmatched_existing_nodes": (
                    "retained_without_state_or_confidence_degradation"
                ),
            },
            "relation_edge_value": dict(RELATION_EDGE_VALUE_SPEC),
            "confidence_threshold": args.threshold,
            "min_mask_area": args.min_mask_area,
            "max_detections": args.max_detections,
            "sam3_endpoint": args.sam3_endpoint,
            "sam3_intermediate": {
                "saved_for_every_input_image": True,
                "trace_schema": "tablesim.sam3_detection_trace.v1",
                "stages": [
                    "cropped_input_image",
                    "object_query_after_threshold_and_mask_clustering",
                    "robot_auxiliary_query",
                    "scene_graph_input_after_robot_exclusion",
                    "six_frame_label_sliding_window",
                    "scene_graph_node_matching_and_update",
                ],
                "mask_visualization": "semi_transparent_overlay",
                "compressed_masks_embedded_in_trace_json": False,
            },
        },
        "render_config": {
            "output_height": args.output_height,
            "graph_width": args.graph_width,
            "max_relations_drawn": args.max_relations,
            "layout": "stable_topological_grid",
        },
        "updates": normalized_records,
    }
    path = output_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    args = parse_args()
    if args.initialization_window < 0.0:
        raise ValueError("--initialization-window must be non-negative")
    if args.initialization_min_observations < 1:
        raise ValueError(
            "--initialization-min-observations must be at least 1"
        )
    if not 0.0 <= args.initialization_min_confidence <= 1.0:
        raise ValueError(
            "--initialization-min-confidence must be in [0, 1]"
        )
    if args.initialization_match_distance < 0.0:
        raise ValueError(
            "--initialization-match-distance must be non-negative"
        )
    if args.initialization_max_gap <= 0.0:
        raise ValueError(
            "--initialization-max-gap must be greater than zero"
        )
    if args.label_window_size != 6:
        raise ValueError("--label-window-size must be exactly 6")
    if args.label_window_match_distance < 0.0:
        raise ValueError(
            "--label-window-match-distance must be non-negative"
        )
    if not 0.0 <= args.label_window_min_bbox_iou <= 1.0:
        raise ValueError("--label-window-min-bbox-iou must be in [0, 1]")
    if not 0.0 <= args.label_window_min_size_similarity <= 1.0:
        raise ValueError(
            "--label-window-min-size-similarity must be in [0, 1]"
        )
    if args.sam3_redetect_interval < 1:
        raise ValueError("--sam3-redetect-interval must be at least 1")
    if not 0.0 <= args.sam3_tracker_match_iou <= 1.0:
        raise ValueError("--sam3-tracker-match-iou must be in [0, 1]")
    frames_manifest_path = Path(args.frames_manifest).expanduser().resolve()
    if not frames_manifest_path.exists():
        raise FileNotFoundError(frames_manifest_path)
    (
        video_path,
        metadata,
        frame_indices,
        frame_paths,
        sampling_frequency_hz,
    ) = load_frames_manifest(frames_manifest_path)
    args.update_interval = 1.0 / sampling_frequency_hz

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else video_path.parent / f"{video_path.stem}_scene_graph"
    )
    output_video = (
        Path(args.output_video).expanduser().resolve()
        if args.output_video
        else output_dir / f"{video_path.stem}_scene_graph_demo.mp4"
    )
    prompts = parse_prompts(args)
    robot_prompts = parse_robot_prompts(args)
    crop_roi = parse_normalized_crop_roi(args.sam3_crop_roi)
    updates_dir, intermediate_dir, crop_dir = prepare_output(
        output_dir,
        output_video,
        overwrite=args.overwrite,
    )
    cropped_frame_paths, crop_records = crop_frame_sequence(
        frame_paths,
        crop_dir=crop_dir,
        normalized_roi=crop_roi,
    )

    print(
        f"video: {video_path} ({metadata['frame_count']} frames, "
        f"{metadata['fps']:.3f} FPS, {metadata['duration_seconds']:.3f}s)",
        flush=True,
    )
    print(
        f"image inputs: {len(frame_indices)} PNG frames at "
        f"{1.0 / args.update_interval:.3f} Hz "
        f"({args.update_interval:.3f}s interval)",
        flush=True,
    )
    print(f"frames manifest: {frames_manifest_path}", flush=True)
    print(
        "SAM3 input crop: "
        f"normalized ROI={tuple(round(value, 6) for value in crop_roi)}, "
        f"pixel bbox={tuple(pixel_crop_bbox([metadata['width'], metadata['height']], crop_roi))}; "
        f"{len(cropped_frame_paths)} PNG crops -> {crop_dir}",
        flush=True,
    )
    print(
        "label correction: 6-frame sliding window "
        "(current + preceding 5); inactive for updates 0-4; "
        f"position distance <= {args.label_window_match_distance:.3f}, "
        f"bbox IoU >= {args.label_window_min_bbox_iou:.2f}",
        flush=True,
    )
    print(
        "scene graph initialization: "
        f"0-{args.initialization_window:.2f}s, "
        f">={args.initialization_min_observations} observations, "
        f"confidence >= {args.initialization_min_confidence:.2f}, "
        f"xy distance <= {args.initialization_match_distance:.3f}; "
        "node set locked afterward; inventory validator disabled",
        flush=True,
    )
    print(f"SAM3 prompts ({len(prompts)}): {', '.join(prompts)}", flush=True)
    print(
        "SAM3 temporal mode: "
        + (
            "one mask-memory session; first-frame seed; "
            f"semantic re-detection every {args.sam3_redetect_interval} "
            "sampled frames"
            if args.sam3_video_memory
            else "legacy independent image requests"
        ),
        flush=True,
    )
    if args.robot_exclusion:
        print(
            f"robot exclusion prompts ({len(robot_prompts)}): "
            f"{', '.join(robot_prompts)} "
            f"(IoU >= {args.robot_match_iou:.2f} or object-mask containment "
            f">= {args.robot_mask_containment:.2f})",
            flush=True,
        )
    snapshots, records = run_scene_graph_updates(
        frame_paths=cropped_frame_paths,
        source_frame_paths=frame_paths,
        crop_records=crop_records,
        frame_indices=frame_indices,
        fps=metadata["fps"],
        video_path=video_path,
        updates_dir=updates_dir,
        intermediate_dir=intermediate_dir,
        prompts=prompts,
        robot_prompts=robot_prompts,
        args=args,
    )
    graph_images = render_scene_graph_updates(
        snapshots,
        records,
        updates_dir=updates_dir,
        width=args.graph_width,
        height=args.output_height,
        dpi=args.graph_dpi,
        max_relations=args.max_relations,
    )
    compose_video(
        video_path=video_path,
        output_path=output_video,
        graph_image_paths=graph_images,
        frame_indices=frame_indices,
        fps=metadata["fps"],
        output_height=args.output_height,
        graph_width=args.graph_width,
        source_aspect=metadata["width"] / metadata["height"],
    )
    manifest_path = write_manifest(
        output_dir=output_dir,
        video_path=video_path,
        output_video=output_video,
        metadata=metadata,
        records=records,
        prompts=prompts,
        frames_manifest_path=frames_manifest_path,
        args=args,
    )

    print(f"saved update images: {len(graph_images)} -> {updates_dir}", flush=True)
    print(
        f"saved SAM3 cropped inputs: {len(cropped_frame_paths)} -> {crop_dir}",
        flush=True,
    )
    print(
        f"saved SAM3 intermediate traces: {len(records)} -> {intermediate_dir}",
        flush=True,
    )
    print(f"manifest: {manifest_path}", flush=True)
    print(f"video: {output_video}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
