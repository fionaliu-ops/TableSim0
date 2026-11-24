"""Offline MCTS decisions and paired search comparison. Does not execute robots."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from high_level_planner.reasoning_graph.graph import ReasoningGraph
from high_level_planner.scene_graph.graph import SceneGraph
from .goal_mcts import GoalConditionedMCTSPlanner, BudgetedUCSPlanner
from .mcts import MCTSConfig
from .tidiness import TidinessMCTSPlanner, GridProposal


def strict_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(v) for v in value]
    return value


def result_dict(result):
    return {"path": [edge.to_dict() for edge in result.path], "total_cost": result.total_cost,
            "goal_state": result.goal_state, "start_state": result.start_state,
            "visited": result.visited, "predicted_goal_path_found": result.goal_state is not None,
            "physical_success_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["compare", "heuristic"], required=True)
    parser.add_argument("--reasoning-graph")
    parser.add_argument("--scene-graph")
    parser.add_argument("--instruction", default="整理桌面")
    parser.add_argument("--threshold", type=float, default=.85)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--rollout-depth", type=int, default=4)
    parser.add_argument("--max-transitions", type=int, default=4096)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", help="Paired comparison seeds")
    parser.add_argument("--output", default="planner_outputs/baseline/mcts/result.json")
    args = parser.parse_args()
    config = MCTSConfig(iterations=args.iterations, max_depth=args.max_depth,
                        rollout_depth=args.rollout_depth, max_transitions=args.max_transitions,
                        time_limit_seconds=args.time_limit, seed=args.seed)
    output = {"mode": args.mode, "config": asdict(config)}
    if args.mode == "compare":
        if not args.reasoning_graph:
            parser.error("compare requires --reasoning-graph")
        raw = Path(args.reasoning_graph).read_bytes()
        graph = ReasoningGraph.from_dict(json.loads(raw))
        graph.validate()
        output.update(graph_sha256=hashlib.sha256(raw).hexdigest(),
                      graph_generation_included=False, trials=[])
        for i, seed in enumerate(args.seeds or [args.seed]):
            trial_config = MCTSConfig(**{**asdict(config), "seed": seed})
            mcts = GoalConditionedMCTSPlanner(config=trial_config)
            ucs = BudgetedUCSPlanner(max_transitions=config.max_transitions,
                                    time_limit_seconds=config.time_limit_seconds)
            # Alternate order to reduce simple warm-cache timing bias.
            planners = [("ucs", ucs), ("mcts", mcts)]
            if i % 2:
                planners.reverse()
            trial = {"seed": seed}
            for label, planner in planners:
                result = planner.plan(graph)
                trial[label] = {"result": result_dict(result), "metrics": planner.metrics}
            output["trials"].append(trial)
    else:
        if not args.scene_graph:
            parser.error("tidiness modes require --scene-graph")
        scene = SceneGraph.from_json(args.scene_graph)
        planner = TidinessMCTSPlanner(config=config, threshold=args.threshold,
            proposal=GridProposal(grid_size=args.grid_size, max_candidates=args.max_candidates))
        output["decision"] = planner.plan_initial_scene(scene, args.instruction).to_dict()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(strict_json(output), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
