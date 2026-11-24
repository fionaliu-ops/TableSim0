"""Static image-first planning, or offline planning from two Scene Graphs."""
import argparse
import json
import os
from pathlib import Path

from high_level_planner.goal_generator import GoalGenerator, GoalGenerationError
from high_level_planner.reasoning_graph import APIConfig, OpenAICompatibleLanguageModel
from high_level_planner.reasoning_graph.target_conditioned import generate_between_scenes
from high_level_planner.scene_graph.graph import SceneGraph
from high_level_planner.planner.ucs import uniform_cost_search
from high_level_planner.interface.vla_interface import Pi05VLAInterface
from high_level_planner.ucs_path_output import write_ucs_path_output
from script.generate_goal import image_settings
from script.goal_config import configured_tokens


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Initial photo; generates/perceives a goal image")
    parser.add_argument("--layout-image", help="Use an existing goal layout instead of generating another image")
    parser.add_argument("--scene-graph", help="Initial graph (optional cache with --image)")
    parser.add_argument("--target-scene-graph", help="Use an existing terminal graph; no image generation")
    parser.add_argument("--instruction", default="Organize the desk.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--image-config")
    parser.add_argument("--output-style", choices=("photo", "layout"), default="layout")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--sam3-endpoint", default=os.environ.get("TABLESIM_SAM3_ENDPOINT", "http://127.0.0.1:8003"))
    parser.add_argument("--backend", choices=("api", "rule_based"), default="api")
    parser.add_argument("--max-states", type=int, default=128)
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument("--api-token")
    parser.add_argument("--api-mode", choices=("auto", "chat_completions", "responses"), default="auto")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--stream", action="store_true", help="Stream chat responses and retain complete answer audits")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--max-actions", type=int, default=64)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--num-branches", type=int, default=5)
    parser.add_argument("--root-beam-width", type=int)
    parser.add_argument("--position-tolerance", type=float, default=0.02,
                        help="Target position tolerance in the graphs' coordinate units")
    parser.add_argument("--disable-ucs-path-output", action="store_true")
    args = parser.parse_args()
    if args.target_scene_graph and args.layout_image:
        parser.error("Choose --layout-image or --target-scene-graph, not both")
    if args.target_scene_graph and not args.scene_graph:
        parser.error("--target-scene-graph requires --scene-graph")
    if not args.target_scene_graph and not args.image:
        parser.error("Supply --image, or both --scene-graph and --target-scene-graph")
    return args


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings, image_config = image_settings(args)
    reasoning = settings.get("reasoning", {})
    config = APIConfig.from_env(model=args.model or reasoning.get("model"),
                                base_url=args.endpoint or reasoning.get("url"))
    from dataclasses import replace
    config = replace(config, api_token=args.api_token or config.api_token, api_mode=args.api_mode,
                     reasoning_effort=args.reasoning_effort or reasoning.get("reasoning_effort"),
                     timeout=args.timeout if args.timeout is not None else reasoning.get("timeout", 600), temperature=reasoning.get("temperature", config.temperature),
                     stream=args.stream or reasoning.get("stream", False), max_output_tokens=args.max_output_tokens,
                     audit_dir=str(output / "model_audit"))
    initial = SceneGraph.from_json(args.scene_graph) if args.scene_graph else None
    prediction = None
    with configured_tokens(settings, image_config):
        if args.target_scene_graph:
            target = SceneGraph.from_json(args.target_scene_graph)
        else:
            prediction = GoalGenerator(image_config=image_config, output_dir=output / "goals",
                                       sam3_endpoint=args.sam3_endpoint, output_style=args.output_style,
                                       max_attempts=args.max_attempts).generate(
                args.image, initial_scene_graph=initial, instruction=args.instruction, layout_image=args.layout_image)
            if not prediction.ready_for_planning:
                raise GoalGenerationError(f"Goal identity needs review: {prediction.bundle_path}")
            initial, target = prediction.initial_scene_graph, prediction.scene_graph
        initial.to_json(output / "initial_scene_graph.json")
        from high_level_planner.reasoning_graph.state import SimpleSceneGraph
        (output / "terminal_scene_graph.json").write_text(json.dumps(SimpleSceneGraph.from_scene_graph(target).to_dict(), indent=2)+"\n")
        graph = generate_between_scenes(initial, target, args.instruction,
                                        model=OpenAICompatibleLanguageModel(config) if args.backend == "api" else None,
                                        backend=args.backend, max_states=args.max_states,
                                        max_depth=args.max_actions, beam_width=args.beam_width,
                                        num_branches=args.num_branches, root_beam_width=args.root_beam_width,
                                        position_tolerance=args.position_tolerance)
    if prediction:
        graph.metadata["goal_bundle_path"] = str(prediction.bundle_path)
    result = uniform_cost_search(graph)
    initial.to_json(output / "initial_scene_graph.json")
    initial.to_json(output / "scene_graph.json")
    from high_level_planner.reasoning_graph.state import SimpleSceneGraph
    (output / "terminal_scene_graph.json").write_text(json.dumps(SimpleSceneGraph.from_scene_graph(target).to_dict(), indent=2)+"\n")
    graph.to_json(output / "reasoning_graph.json")
    if prediction:
        prediction.record_plan(goal_reached=result.goal_state is not None,
                               reasoning_graph_path=output / "reasoning_graph.json")
    write_ucs_path_output(graph, result, output, enabled=not args.disable_ucs_path_output)
    request = Pi05VLAInterface().format_subtask(result.next_subtask)
    (output / "vla_request.json").write_text(json.dumps(request.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"reasoning_mode": "initial_terminal_graphs", "goal_reached": result.goal_state is not None,
               "path_length": len(result.path), "total_cost": result.total_cost,
               "backend": args.backend, "node_count": len(graph.nodes), "edge_count": len(graph.edges),
               "goal_ids": sorted(graph.goal_ids), "goal_count": len(graph.goal_ids),
               "model_execution": graph.metadata.get("model_execution"),
               "physical_execution_verified": False}
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("reasoning_graph:", output / "reasoning_graph.json")
    print("goal_reached:", summary["goal_reached"])
    return 0 if summary["goal_reached"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
