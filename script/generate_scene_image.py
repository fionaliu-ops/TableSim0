"""Independent Seedream smoke test; does not execute a reasoning graph or validator."""

from __future__ import annotations

import argparse

from high_level_planner.scene_graph.graph import SceneGraph
from high_level_planner.scene_image_generator import (
    SeedreamConfig,
    SeedreamSceneGenerator,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Defaults to project config/seedream.json")
    parser.add_argument(
        "--prompt",
        required=True,
        help="Desired scene or reference-image edit instruction",
    )
    parser.add_argument(
        "--reference-image",
        action="append",
        default=[],
        help="Explicit local reference; repeat for multiple images",
    )
    parser.add_argument(
        "--scene-graph",
        help="Optional target Scene Graph JSON, not the whole reasoning graph",
    )
    parser.add_argument(
        "--state-id", help="Optional reasoning state label; metadata only"
    )
    parser.add_argument("--output-dir", default="planner_outputs/scene_images")
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument("--size")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SeedreamConfig.load(
        args.config, model=args.model, base_url=args.endpoint, size=args.size
    )
    scene = SceneGraph.from_json(args.scene_graph) if args.scene_graph else None
    result = SeedreamSceneGenerator(config, args.output_dir).generate(
        args.prompt,
        reference_images=args.reference_image,
        scene_graph=scene,
        state_id=args.state_id,
    )
    print("image:", result.image_path)
    print("generation_manifest:", result.manifest_path)
    print("evidence_kind:", result.evidence_kind)
    print("verified:", result.verified)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
