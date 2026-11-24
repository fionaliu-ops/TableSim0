from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2


def video_metadata(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0.0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid video metadata: fps={fps}, frames={frame_count}, "
            f"size={width}x{height}"
        )
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": frame_count / fps,
    }


def sample_frame_indices(
    *,
    frame_count: int,
    source_fps: float,
    sampling_frequency_hz: float,
) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be greater than zero")
    if source_fps <= 0.0:
        raise ValueError("source_fps must be greater than zero")
    if sampling_frequency_hz <= 0.0:
        raise ValueError("sampling_frequency_hz must be greater than zero")
    if sampling_frequency_hz > source_fps:
        raise ValueError(
            "sampling_frequency_hz cannot exceed the source video FPS"
        )

    step = source_fps / sampling_frequency_hz
    indices: list[int] = []
    sample_index = 0
    while True:
        frame_index = int(round(sample_index * step))
        if frame_index >= frame_count:
            break
        if not indices or frame_index > indices[-1]:
            indices.append(frame_index)
        sample_index += 1
    return indices or [0]


def extract_frames(
    video_path: Path,
    frame_indices: Sequence[int],
    frames_dir: Path,
) -> list[Path]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    requested = set(frame_indices)
    output_paths: list[Path] = []
    frame_index = 0
    while len(output_paths) < len(frame_indices):
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in requested:
            output_path = frames_dir / f"frame_{frame_index:06d}.png"
            if not cv2.imwrite(str(output_path), frame):
                capture.release()
                raise RuntimeError(f"Unable to write frame: {output_path}")
            output_paths.append(output_path)
        frame_index += 1
    capture.release()

    if len(output_paths) != len(frame_indices):
        raise RuntimeError(
            f"Extracted {len(output_paths)} frames, expected "
            f"{len(frame_indices)}"
        )
    return output_paths


def write_frames_manifest(
    *,
    output_dir: Path,
    video_path: Path,
    metadata: dict[str, Any],
    sampling_frequency_hz: float,
    frame_indices: Sequence[int],
    frame_paths: Sequence[Path],
) -> Path:
    records = [
        {
            "sample_index": sample_index,
            "source_frame_index": frame_index,
            "timestamp_seconds": round(frame_index / metadata["fps"], 6),
            "image": str(path.relative_to(output_dir)),
        }
        for sample_index, (frame_index, path) in enumerate(
            zip(frame_indices, frame_paths)
        )
    ]
    manifest = {
        "schema": "tablesim.image_sequence.v1",
        "source_video": str(video_path),
        "source_video_metadata": metadata,
        "sampling": {
            "frequency_hz": float(sampling_frequency_hz),
            "interval_seconds": 1.0 / float(sampling_frequency_hz),
            "image_count": len(records),
            "image_format": "png",
        },
        "frames": records,
    }
    manifest_path = output_dir / "frames_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a PNG image sequence from a video before Scene Graph "
            "generation."
        )
    )
    parser.add_argument("--video", required=True, help="Source video path.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for frames/ and frames_manifest.json.",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=5.0,
        help="Frame sampling frequency in Hz. Defaults to 5 Hz.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace previously extracted frames in this output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    frames_dir = output_dir / "frames"
    manifest_path = output_dir / "frames_manifest.json"
    existing = [*frames_dir.glob("frame_*.png")]
    if manifest_path.exists():
        existing.append(manifest_path)
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Extracted frame output already exists ({existing[0]}). "
            "Pass --overwrite to replace it."
        )
    if args.overwrite:
        for path in existing:
            path.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)

    metadata = video_metadata(video_path)
    frame_indices = sample_frame_indices(
        frame_count=metadata["frame_count"],
        source_fps=metadata["fps"],
        sampling_frequency_hz=args.frequency,
    )
    frame_paths = extract_frames(video_path, frame_indices, frames_dir)
    manifest_path = write_frames_manifest(
        output_dir=output_dir,
        video_path=video_path,
        metadata=metadata,
        sampling_frequency_hz=args.frequency,
        frame_indices=frame_indices,
        frame_paths=frame_paths,
    )

    print(
        f"source: {video_path} ({metadata['fps']:.3f} FPS, "
        f"{metadata['frame_count']} frames)",
        flush=True,
    )
    print(
        f"sampling: {args.frequency:.3f} Hz -> {len(frame_paths)} PNG images",
        flush=True,
    )
    print(f"frames: {frames_dir}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
