#!/usr/bin/env python3
"""Batch-generate, validate, rank, and materialize TableSim desk scenes.

The existing ``scence_generator.py`` remains the single-batch renderer.  This
entrypoint orchestrates deterministic batches and then performs simulator-free
manifest/image screening.  It is safe to run filtering again without rendering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scene_selection import (  # noqa: E402
    DEFAULT_TARGET_LABELS,
    SceneFilterConfig,
    evaluate_scene,
    select_diverse,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def derive_seed(base_seed: int, batch_index: int, stride: int) -> int:
    """Derive stable NumPy-compatible seeds for independently runnable batches."""

    return (base_seed + batch_index * stride) % (2**32)


def batch_directory(output_dir: Path, batch_index: int, seed: int) -> Path:
    return output_dir / "batches" / f"batch_{batch_index:05d}_seed_{seed}"


def generator_command(
    args: argparse.Namespace,
    batch_index: int,
    seed: int,
    target_dir: Path,
) -> list[str]:
    command = [
        args.python,
        str(args.generator),
        "--scenes",
        str(args.scenes_per_batch),
        "--seed",
        str(seed),
        "--output-dir",
        str(target_dir),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--min-objects",
        str(args.min_objects),
        "--max-objects",
        str(args.max_objects),
        "--min-placed",
        str(args.min_placed),
        "--render-steps",
        str(args.render_steps),
    ]
    if args.max_attempts is not None:
        command.extend(["--max-attempts", str(args.max_attempts)])
    if args.allow_missing_on:
        command.append("--allow-missing-on")
    if args.allow_missing_occlusion:
        command.append("--allow-missing-occlusion")
    if args.generator_fail_fast:
        command.append("--fail-fast")
    return command


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _complete_batch(
    manifest_path: Path,
    expected_seed: int,
    expected_scenes: int,
    *,
    require_natural: bool = False,
) -> bool:
    manifest = _read_json(manifest_path)
    if manifest is None or manifest.get("seed") != expected_seed:
        return False
    config = manifest.get("generation_config") or {}
    if require_natural and (
        config.get("forced_relationships") is not False
        or config.get("forced_occlusion") is not False
    ):
        raise ValueError(
            f"cannot resume a legacy staged batch: {manifest_path}; "
            "choose a new output directory to keep old and natural scenes separate"
        )
    scenes = manifest.get("scenes")
    summary = manifest.get("summary")
    if not isinstance(scenes, list) or not isinstance(summary, dict):
        return False
    return len(scenes) == expected_scenes and summary.get("rendered_scenes") == expected_scenes


def run_generation(args: argparse.Namespace) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for offset in range(args.batches):
        batch_index = args.batch_start_index + offset
        seed = derive_seed(args.seed, batch_index, args.seed_stride)
        target_dir = batch_directory(args.output_dir, batch_index, seed)
        manifest_path = target_dir / "manifest.json"
        command = generator_command(args, batch_index, seed, target_dir)
        status: dict[str, Any] = {
            "batch_index": batch_index,
            "seed": seed,
            "directory": _relative(target_dir, args.output_dir),
            "command": command,
        }

        if args.resume and _complete_batch(
            manifest_path, seed, args.scenes_per_batch,
            require_natural=args.generator.resolve() == (ROOT / "script" / "scence_generator.py").resolve(),
        ):
            status.update({"status": "reused", "returncode": 0})
            statuses.append(status)
            print(f"REUSE batch={batch_index} seed={seed} {target_dir}", flush=True)
            continue
        if target_dir.exists() and not args.resume:
            raise FileExistsError(
                f"batch directory already exists: {target_dir}; use --resume or choose a new output"
            )

        print("RUN " + subprocess.list2cmdline(command), flush=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        started = _utc_now()
        completed = subprocess.run(command, cwd=ROOT, check=False)
        status.update(
            {
                "status": "completed" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "started_at": started,
                "finished_at": _utc_now(),
                "manifest_exists": manifest_path.is_file(),
            }
        )
        statuses.append(status)
        if completed.returncode != 0:
            print(f"BATCH FAILED batch={batch_index} seed={seed}", file=sys.stderr, flush=True)
            if args.fail_fast:
                break
    return statuses


def aggregate_batches_flat(args: argparse.Namespace) -> dict[str, Any]:
    """Copy completed batch scenes into the output root and merge manifests."""

    _clear_generated_files(args.output_dir, ("scene_*.png", "manifest.json"))
    scenes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    source_manifests: list[str] = []
    incomplete_batches: list[dict[str, Any]] = []
    first_manifest: dict[str, Any] | None = None
    candidate_attempts = 0
    max_attempts = 0

    for offset in range(args.batches):
        batch_index = args.batch_start_index + offset
        seed = derive_seed(args.seed, batch_index, args.seed_stride)
        target_dir = batch_directory(args.output_dir, batch_index, seed)
        manifest_path = target_dir / "manifest.json"
        manifest = _read_json(manifest_path)
        if manifest is None or not _complete_batch(
            manifest_path, seed, args.scenes_per_batch
        ):
            incomplete_batches.append(
                {
                    "batch_index": batch_index,
                    "seed": seed,
                    "manifest": _relative(manifest_path, args.output_dir),
                }
            )
            continue

        if first_manifest is None:
            first_manifest = manifest
        source_manifest = _relative(manifest_path, args.output_dir)
        source_manifests.append(source_manifest)
        summary = manifest.get("summary") or {}
        candidate_attempts += int(summary.get("candidate_attempts") or 0)
        max_attempts += int(summary.get("max_attempts") or 0)

        for failure in manifest.get("failures") or []:
            if not isinstance(failure, dict):
                continue
            merged_failure = dict(failure)
            merged_failure.update(
                {
                    "source_batch_index": batch_index,
                    "source_seed": seed,
                    "source_manifest": source_manifest,
                }
            )
            failures.append(merged_failure)

        for local_index, scene in enumerate(manifest["scenes"]):
            if not isinstance(scene, dict):
                raise ValueError(f"invalid scene record in {manifest_path}")
            source_image_name = str(scene.get("image", ""))
            source_image = target_dir / source_image_name
            if not source_image.is_file():
                raise FileNotFoundError(source_image)

            global_index = len(scenes)
            image_name = f"scene_{global_index:03d}.png"
            shutil.copy2(source_image, args.output_dir / image_name)
            merged_scene = dict(scene)
            merged_scene.update(
                {
                    "image": image_name,
                    "scene_index": global_index,
                    "source_batch_index": batch_index,
                    "source_seed": seed,
                    "source_scene_index": scene.get("scene_index", local_index),
                    "source_image": source_image_name,
                    "source_manifest": source_manifest,
                }
            )
            scenes.append(merged_scene)

    requested_scenes = args.batches * args.scenes_per_batch
    first_manifest = first_manifest or {}
    generation_config = dict(first_manifest.get("generation_config") or {})
    generation_config.update(
        {
            "scenes": requested_scenes,
            "isolated_processes": True,
            "batches": args.batches,
            "scenes_per_batch": args.scenes_per_batch,
            "seed_stride": args.seed_stride,
        }
    )
    payload = {
        "generator": "script/batch_generate_scenes.py",
        "scene_generator": first_manifest.get(
            "generator", "script/scence_generator.py"
        ),
        "aggregation": "flat",
        "messy_json": first_manifest.get("messy_json"),
        "asset_map": first_manifest.get("asset_map"),
        "seed": args.seed,
        "runtime": first_manifest.get("runtime"),
        "generation_config": generation_config,
        "table": first_manifest.get("table"),
        "available_json_objects": first_manifest.get("available_json_objects"),
        "source_manifests": source_manifests,
        "incomplete_batches": incomplete_batches,
        "scenes": scenes,
        "failures": failures,
        "summary": {
            "requested_scenes": requested_scenes,
            "candidate_attempts": candidate_attempts,
            "max_attempts": max_attempts,
            "rendered_scenes": len(scenes),
            "failures": len(failures),
            "incomplete_batches": len(incomplete_batches),
            "images": [scene["image"] for scene in scenes],
        },
    }
    _json_dump(args.output_dir / "manifest.json", payload)
    return payload


def discover_manifests(input_dir: Path) -> list[Path]:
    """Find generator manifests while ignoring this pipeline's own outputs."""

    if input_dir.is_file():
        candidates = [input_dir]
    else:
        candidates = sorted(input_dir.rglob("manifest.json"))
    manifests: list[Path] = []
    for path in candidates:
        relative_parts = set(path.relative_to(input_dir).parts[:-1]) if input_dir.is_dir() else set()
        if relative_parts & {"selected", "review"}:
            continue
        payload = _read_json(path)
        if payload is None or not isinstance(payload.get("scenes"), list):
            continue
        generator = str(payload.get("generator", ""))
        if "scence_generator.py" not in generator:
            continue
        manifests.append(path)
    return manifests


def build_filter_config(args: argparse.Namespace) -> SceneFilterConfig:
    return SceneFilterConfig(
        min_placed=args.filter_min_placed,
        max_skipped=args.max_skipped,
        require_on=not args.allow_missing_on,
        require_occlusion=not args.allow_missing_occlusion,
        require_layout_valid=not args.allow_invalid_layout,
        require_target_candidate=not args.allow_no_target,
        validate_images=not args.skip_image_checks,
        target_labels=tuple(args.target_labels),
        workspace_x=tuple(args.workspace_x),
        workspace_y=tuple(args.workspace_y),
        min_target_clearance=args.min_target_clearance,
        max_target_footprint=args.max_target_footprint,
        max_target_height=args.max_target_height,
        min_luminance=args.min_luminance,
        max_luminance=args.max_luminance,
        min_luminance_stddev=args.min_luminance_stddev,
        max_clipped_fraction=args.max_clipped_fraction,
        min_entropy=args.min_entropy,
    )


def evaluate_manifests(
    manifest_paths: list[Path],
    output_root: Path,
    config: SceneFilterConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    manifest_summaries: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = _read_json(manifest_path)
        assert manifest is not None
        source_key = _relative(manifest_path, output_root)
        batch_summary = {
            "manifest": source_key,
            "seed": manifest.get("seed"),
            "scene_count": len(manifest["scenes"]),
            "generation_failures": len(manifest.get("failures") or []),
        }
        manifest_summaries.append(batch_summary)
        for index, scene in enumerate(manifest["scenes"]):
            if not isinstance(scene, dict):
                continue
            image_name = scene.get("image")
            image_path = manifest_path.parent / str(image_name)
            scene_key = f"{source_key}::{image_name or index}"
            scene_id = hashlib.sha1(scene_key.encode("utf-8")).hexdigest()[:12]
            decision = evaluate_scene(scene, image_path, config)
            result = {
                "scene_id": scene_id,
                "source_manifest": source_key,
                "source_image": _relative(image_path, output_root),
                "seed": manifest.get("seed"),
                "scene_index": scene.get("scene_index", index),
                "attempt_index": scene.get("attempt_index"),
                **decision,
                "scene": scene,
                "_image_path": image_path,
            }
            results.append(result)
    results.sort(key=lambda item: (not item["accepted"], -item["score"], item["scene_id"]))
    return results, manifest_summaries


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def write_catalog(
    output_dir: Path,
    results: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    config: SceneFilterConfig,
) -> None:
    accepted = sum(bool(item["accepted"]) for item in results)
    payload = {
        "pipeline": "script/batch_generate_scenes.py",
        "created_at": _utc_now(),
        "filter_config": config.to_dict(),
        "summary": {
            "manifests": len(manifests),
            "evaluated_scenes": len(results),
            "accepted_scenes": accepted,
            "rejected_scenes": len(results) - accepted,
        },
        "source_manifests": manifests,
        "scenes": [_public_result(item) for item in results],
    }
    _json_dump(output_dir / "catalog.json", payload)

    with (output_dir / "catalog.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scene_id",
                "accepted",
                "score",
                "seed",
                "scene_index",
                "source_image",
                "placed_count",
                "skipped_count",
                "eligible_target_labels",
                "reasons",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "scene_id": item["scene_id"],
                    "accepted": item["accepted"],
                    "score": item["score"],
                    "seed": item["seed"],
                    "scene_index": item["scene_index"],
                    "source_image": item["source_image"],
                    "placed_count": item["placed_count"],
                    "skipped_count": item["skipped_count"],
                    "eligible_target_labels": "|".join(item["eligible_target_labels"]),
                    "reasons": "|".join(item["reasons"]),
                }
            )


def _clear_generated_files(directory: Path, patterns: tuple[str, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_file():
                path.unlink()


def materialize_selection(
    output_dir: Path,
    selected: list[dict[str, Any]],
    config: SceneFilterConfig,
    mode: str,
) -> dict[str, Any]:
    selected_dir = output_dir / "selected"
    _clear_generated_files(selected_dir, ("scene_*", "manifest.json"))
    records: list[dict[str, Any]] = []
    for rank, result in enumerate(selected):
        source = result["_image_path"]
        suffix = source.suffix.lower() or ".png"
        destination = selected_dir / f"scene_{rank:05d}_{result['scene_id']}{suffix}"
        actual_mode = mode
        if mode == "copy":
            shutil.copy2(source, destination)
        elif mode == "hardlink":
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
                actual_mode = "copy_fallback"
        elif mode != "none":
            raise ValueError(f"unsupported materialization mode: {mode}")
        record = _public_result(result)
        record["selection_rank"] = rank
        record["selected_image"] = (
            _relative(destination, output_dir) if mode != "none" else None
        )
        record["materialization"] = actual_mode
        records.append(record)

    payload = {
        "pipeline": "script/batch_generate_scenes.py",
        "created_at": _utc_now(),
        "filter_config": config.to_dict(),
        "summary": {"selected_scenes": len(records), "materialization": mode},
        "scenes": records,
    }
    _json_dump(selected_dir / "manifest.json", payload)
    return payload


def _contact_sheet(
    results: list[dict[str, Any]],
    destination: Path,
    columns: int,
    thumbnail_size: tuple[int, int],
) -> None:
    if not results:
        return
    cell_width = thumbnail_size[0]
    label_height = 56
    cell_height = thumbnail_size[1] + label_height
    rows = (len(results) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    for index, result in enumerate(results):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        try:
            with Image.open(result["_image_path"]) as source:
                image = source.convert("RGB")
                image.thumbnail(thumbnail_size)
                offset_x = x + (cell_width - image.width) // 2
                sheet.paste(image, (offset_x, y))
        except Exception:
            draw.rectangle((x, y, x + cell_width - 1, y + thumbnail_size[1] - 1), outline="red")
        targets = ",".join(result.get("eligible_target_labels") or []) or "none"
        reasons = ",".join(result.get("reasons") or []) or "accepted"
        draw.text((x + 4, y + thumbnail_size[1] + 3), f"{result['scene_id']} score={result['score']}", fill="white")
        draw.text((x + 4, y + thumbnail_size[1] + 19), f"targets={targets}"[:55], fill=(180, 230, 180))
        draw.text((x + 4, y + thumbnail_size[1] + 35), reasons[:55], fill=(240, 180, 180))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def write_review_sheets(
    output_dir: Path,
    selected: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    columns: int,
    page_size: int,
    rejected_limit: int,
) -> None:
    review_dir = output_dir / "review"
    _clear_generated_files(review_dir, ("accepted_*.png", "rejected_*.png"))
    width = 300
    height = 225
    groups = (("accepted", selected), ("rejected", rejected[:rejected_limit]))
    for prefix, items in groups:
        for start in range(0, len(items), page_size):
            page = start // page_size
            _contact_sheet(
                items[start : start + page_size],
                review_dir / f"{prefix}_{page:03d}.png",
                columns,
                (width, height),
            )


def validate_args(args: argparse.Namespace) -> None:
    if args.batches < 1:
        raise ValueError("--batches must be >= 1")
    if args.batch_start_index < 0:
        raise ValueError("--batch-start-index must be >= 0")
    if args.seed_stride < 1:
        raise ValueError("--seed-stride must be >= 1")
    if args.scenes_per_batch < 1:
        raise ValueError("--scenes-per-batch must be >= 1")
    if args.min_objects < 1 or args.max_objects < args.min_objects:
        raise ValueError("--max-objects must be >= --min-objects >= 1")
    if args.min_placed < 1:
        raise ValueError("--min-placed must be >= 1")
    if args.width < 128 or args.height < 128:
        raise ValueError("--width and --height must be >= 128")
    if args.filter_min_placed < 1:
        raise ValueError("--filter-min-placed must be >= 1")
    if args.selected_count < 0 or args.min_selected < 0:
        raise ValueError("--selected-count and --min-selected must be >= 0")
    if args.workspace_x[0] >= args.workspace_x[1] or args.workspace_y[0] >= args.workspace_y[1]:
        raise ValueError("workspace lower bounds must be smaller than upper bounds")
    if args.min_target_clearance < 0:
        raise ValueError("--min-target-clearance must be >= 0")
    if args.review_columns < 1 or args.review_page_size < 1 or args.review_rejected_count < 0:
        raise ValueError("review dimensions/counts must be positive (rejected count may be zero)")
    if args.filter_only and args.generate_only:
        raise ValueError("--filter-only and --generate-only are mutually exclusive")
    if args.flat_output and args.filter_only:
        raise ValueError("--flat-output cannot be used with --filter-only")
    if not args.filter_only and not args.generator.is_file():
        raise FileNotFoundError(f"generator not found: {args.generator}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "image_collectors" / "scenes_batches",
        help="Pipeline output root (batches, catalog, selected, and review).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Manifest tree for --filter-only; defaults to --output-dir.",
    )
    parser.add_argument("--filter-only", action="store_true", help="Skip rendering and re-filter manifests.")
    parser.add_argument("--generate-only", action="store_true", help="Render batches without aggregating them.")
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help=(
            "Also copy generated scenes to the output root as scene_000.png, "
            "scene_001.png, ... and write one merged manifest.json."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print deterministic generator commands only.")
    parser.add_argument("--resume", action="store_true", help="Reuse complete batches and retry incomplete ones.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop launching batches after a failed batch.")
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--batch-start-index", type=int, default=0, help="Global first batch index for sharding.")
    parser.add_argument("--scenes-per-batch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--seed-stride", type=int, default=1_000_003)
    parser.add_argument("--python", default=sys.executable, help="Python executable used for the renderer.")
    parser.add_argument(
        "--generator",
        type=Path,
        default=ROOT / "script" / "scence_generator.py",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--min-objects", type=int, default=9)
    parser.add_argument("--max-objects", type=int, default=11)
    parser.add_argument("--min-placed", type=int, default=9, help="Generator placement threshold.")
    parser.add_argument("--render-steps", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument(
        "--allow-missing-on", action="store_true", default=True,
        help="Compatibility flag; missing 'on' records are allowed by default.",
    )
    parser.add_argument(
        "--allow-missing-occlusion", action="store_true", default=True,
        help="Compatibility flag; missing occlusion records are allowed by default.",
    )
    parser.add_argument("--generator-fail-fast", action="store_true")

    parser.add_argument("--filter-min-placed", type=int, default=9)
    parser.add_argument("--max-skipped", type=int, default=2)
    parser.add_argument("--allow-invalid-layout", action="store_true")
    parser.add_argument("--allow-no-target", action="store_true")
    parser.add_argument("--skip-image-checks", action="store_true")
    parser.add_argument("--target-labels", nargs="+", default=list(DEFAULT_TARGET_LABELS))
    parser.add_argument("--workspace-x", type=float, nargs=2, default=(-0.36, 0.36), metavar=("MIN", "MAX"))
    parser.add_argument("--workspace-y", type=float, nargs=2, default=(-0.27, 0.22), metavar=("MIN", "MAX"))
    parser.add_argument("--min-target-clearance", type=float, default=0.015)
    parser.add_argument("--max-target-footprint", type=float, default=0.30)
    parser.add_argument("--max-target-height", type=float, default=0.20)
    parser.add_argument("--min-luminance", type=float, default=15.0)
    parser.add_argument("--max-luminance", type=float, default=245.0)
    parser.add_argument("--min-luminance-stddev", type=float, default=3.0)
    parser.add_argument("--max-clipped-fraction", type=float, default=0.98)
    parser.add_argument("--min-entropy", type=float, default=2.0)
    parser.add_argument(
        "--selected-count",
        type=int,
        default=0,
        help="Maximum diverse scenes to keep; 0 keeps every accepted scene.",
    )
    parser.add_argument("--min-selected", type=int, default=1, help="Exit nonzero below this selected count.")
    parser.add_argument("--diversity-bonus", type=float, default=8.0)
    parser.add_argument("--materialize", choices=("copy", "hardlink", "none"), default="copy")
    parser.add_argument("--no-review-sheets", action="store_true")
    parser.add_argument("--review-columns", type=int, default=4)
    parser.add_argument("--review-page-size", type=int, default=20)
    parser.add_argument("--review-rejected-count", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.generator = args.generator.resolve()
    args.input_dir = (args.input_dir or args.output_dir).resolve()
    validate_args(args)

    planned = []
    for offset in range(args.batches):
        batch_index = args.batch_start_index + offset
        seed = derive_seed(args.seed, batch_index, args.seed_stride)
        target = batch_directory(args.output_dir, batch_index, seed)
        planned.append(generator_command(args, batch_index, seed, target))
    if args.dry_run:
        for command in planned:
            print(subprocess.list2cmdline(command))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generation_statuses: list[dict[str, Any]] = []
    if not args.filter_only:
        generation_statuses = run_generation(args)
        _json_dump(
            args.output_dir / "generation_run.json",
            {
                "pipeline": "script/batch_generate_scenes.py",
                "created_at": _utc_now(),
                "base_seed": args.seed,
                "seed_stride": args.seed_stride,
                "batch_start_index": args.batch_start_index,
                "batches_requested": args.batches,
                "scenes_per_batch": args.scenes_per_batch,
                "batches": generation_statuses,
            },
        )
    flat_payload: dict[str, Any] | None = None
    if args.flat_output:
        flat_payload = aggregate_batches_flat(args)
        print(json.dumps(flat_payload["summary"], ensure_ascii=False), flush=True)
    if args.generate_only:
        flat_incomplete = (
            flat_payload is not None
            and flat_payload["summary"]["rendered_scenes"]
            < flat_payload["summary"]["requested_scenes"]
        )
        if any(item.get("returncode") for item in generation_statuses) or flat_incomplete:
            raise SystemExit(1)
        return

    input_dir = args.input_dir if args.filter_only else args.output_dir
    manifest_paths = discover_manifests(input_dir)
    if not manifest_paths:
        raise FileNotFoundError(f"no scence_generator manifests found under {input_dir}")
    config = build_filter_config(args)
    results, manifest_summaries = evaluate_manifests(manifest_paths, args.output_dir, config)
    write_catalog(args.output_dir, results, manifest_summaries, config)

    selected = select_diverse(results, args.selected_count, args.diversity_bonus)
    selected_payload = materialize_selection(args.output_dir, selected, config, args.materialize)
    rejected = [item for item in results if not item["accepted"]]
    if not args.no_review_sheets:
        write_review_sheets(
            args.output_dir,
            selected,
            rejected,
            args.review_columns,
            args.review_page_size,
            args.review_rejected_count,
        )

    summary = {
        "manifests": len(manifest_paths),
        "evaluated": len(results),
        "accepted": sum(bool(item["accepted"]) for item in results),
        "selected": selected_payload["summary"]["selected_scenes"],
        "rejected": len(rejected),
        "catalog": str(args.output_dir / "catalog.json"),
        "selection_manifest": str(args.output_dir / "selected" / "manifest.json"),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)

    generation_failed = any(item.get("returncode") for item in generation_statuses)
    if generation_failed or len(selected) < args.min_selected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
