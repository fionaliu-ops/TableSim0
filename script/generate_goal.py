"""Generate a goal image, then perceive its Scene Graph (image-first only)."""
import argparse
import os

from high_level_planner.goal_generator import GoalGenerator
from high_level_planner.scene_graph.graph import SceneGraph
from high_level_planner.scene_image_generator import load_goal_image_config
from script.goal_config import configured_tokens, load_api_settings


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--layout-image", help="Read an existing terminal layout; skip image generation and terminal SAM3")
    parser.add_argument("--scene-graph", help="Optional cached initial graph for the same photo")
    parser.add_argument("--instruction", default="Organize the desk.")
    parser.add_argument("--output-dir", default="planner_outputs/goal_predictions")
    parser.add_argument("--config", help="JSON image API settings")
    parser.add_argument("--image-config")
    parser.add_argument("--output-style", choices=("photo", "layout"), default="layout")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--sam3-endpoint", default=os.environ.get("TABLESIM_SAM3_ENDPOINT", "http://127.0.0.1:8003"))
    return parser.parse_args()


def image_settings(args):
    settings = load_api_settings(args.config)
    overrides = {field: settings["image"][key]
                 for key, field in (("url", "base_url"), ("model", "model"))
                 if key in settings.get("image", {})}
    return settings, load_goal_image_config(args.image_config, **overrides)


def main():
    args = parse_args()
    settings, image_config = image_settings(args)
    with configured_tokens(settings, image_config):
        result = GoalGenerator(image_config=image_config, output_dir=args.output_dir,
                               sam3_endpoint=args.sam3_endpoint, max_attempts=args.max_attempts,
                               output_style=args.output_style).generate(
            args.image, initial_scene_graph=SceneGraph.from_json(args.scene_graph) if args.scene_graph else None,
            instruction=args.instruction, layout_image=args.layout_image)
    print("goal_bundle:", result.bundle_path)
    print("terminal_scene_graph:", result.output_dir / "terminal_scene_graph.json")
    print("terminal_image:", result.image.image_path)
    print("ready_for_planning:", result.ready_for_planning)
    return 0 if result.ready_for_planning else 2


if __name__ == "__main__":
    raise SystemExit(main())
