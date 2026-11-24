"""Replay cached observations with full replanning and a shared fixed goal."""

import argparse
import json
from pathlib import Path

from high_level_planner.reasoning_graph.generator import ReasoningGraphGenerator
from high_level_planner.reasoning_graph.language_models.openai_compatible import APIConfig, OpenAICompatibleLanguageModel
from high_level_planner.scene_graph.graph import SceneGraph
from .full_replan_planner import FullReplanPlanner


def main():
    parser = argparse.ArgumentParser(description="Full replanning on every cached observation")
    parser.add_argument("--scene-graph", required=True)
    parser.add_argument("--target-scene-graph", required=True)
    parser.add_argument("--observations", nargs="*", default=[])
    parser.add_argument("--instruction", default="整理桌面")
    parser.add_argument("--model")
    parser.add_argument("--api-base")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--num-branches", type=int, default=2)
    parser.add_argument("--position-tolerance", type=float, default=0.02)
    parser.add_argument("--output-dir", default="planner_outputs/baseline/full_replan")
    args = parser.parse_args()
    model = OpenAICompatibleLanguageModel(APIConfig.from_env(model=args.model, base_url=args.api_base))
    generator = ReasoningGraphGenerator(model=model, max_depth=args.max_depth,
                                        beam_width=args.beam_width, num_branches=args.num_branches)
    planner = FullReplanPlanner(reasoning_generator=generator)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = planner.plan_initial_scene(
        SceneGraph.from_json(args.scene_graph), SceneGraph.from_json(args.target_scene_graph),
        args.instruction, position_tolerance=args.position_tolerance)

    def save(index, result):
        path = output_dir / f"round_{index:03d}.json"
        path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"round {index}: goal_path={result.search_result.goal_state is not None} -> {path}")

    save(0, output)
    for index, path in enumerate(args.observations, 1):
        output = planner.update_target_conditioned_scene(
            SceneGraph.from_json(path), args.instruction,
            observation_source="cached_observation", coverage={"path": path})
        save(index, output)


if __name__ == "__main__":
    main()
