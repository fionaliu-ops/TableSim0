"""Replay the formerly failing cup and keyboard transfers without an API call."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from high_level_planner.scene_graph.graph import SceneGraph
from script.compare_scene003_modalities import install_recording, match_book_instances, request_from_action, write
from script.dynamic_sim import build_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = json.loads((args.source / "dynamic_trace.json").read_text())
    graph = SceneGraph.from_json(args.source / "initial_sam3_scene_graph.json")
    env = install_recording(build_environment("scene_003", "scene_003", args.output_dir))
    results = []
    try:
        env.reset(0)
        env.bind_graph(graph, match_book_instances(graph, env.task))
        for index in (0, 2):
            action = trace["decisions"][index]["model_action"]
            name = env.object_map[action["source_object_id"]]
            try:
                observation = env.step(request_from_action(action))
                result = {"source_decision": index, "object": name, "status": "executed",
                          "placement": observation.get("placement")}
            except Exception as exc:
                result = {"source_decision": index, "object": name, "status": "failed",
                          "error": f"{type(exc).__name__}: {exc}", "placement": env.last_placement}
            result["object_pose"] = env.task.objects[name].get_pose().p.tolist()
            results.append(result)
            if result["status"] != "executed":
                break
        output = {"source_trace": str(args.source), "reasoning_graph_called": False,
                  "external_model_called": False, "results": results}
        write(args.output_dir / "replay_result.json", output)
    finally:
        env.close()
    print(json.dumps(output, ensure_ascii=False), flush=True)
    return 0 if all(x["status"] == "executed" for x in results) and len(results) == 2 else 2


if __name__ == "__main__":
    raise SystemExit(main())
