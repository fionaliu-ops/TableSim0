"""Batch direct-vs-Reasoning-Graph comparison for constrained-v2 scenes."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from high_level_planner.scene_graph.attributes import ObjectAttributes
from high_level_planner.scene_graph.graph import Relation, SceneGraph, SceneObject
from constract.compact_layout import compact_target, planning_budgets

SCENES = ROOT / "image_collectors/scenes_constrained_v2"


def image_xy(x, y):
    return [round(float(x / .95 + .5), 6), round(float(.38 - y / .8), 6)]


def graph_from_manifest(scene, *, terminal=False):
    if terminal:
        return compact_target(graph_from_manifest(scene))
    objects = []
    placed = scene["placed_objects"]
    for item in placed:
        label = str(item["label"])
        object_id = f"{label.replace(' ', '_')}_sam3"
        xy = item.get("xy", [0, 0])
        # The manifest uses simulation coordinates. Convert sizes and centers
        # consistently; these are planar proxies, not image measurements.
        width, depth = item.get("footprint", [0.1, 0.1])
        angle = math.radians(float(item.get("yaw_degrees", 0)))
        footprint = [(abs(math.cos(angle)) * width + abs(math.sin(angle)) * depth) / .95,
                     (abs(math.sin(angle)) * width + abs(math.cos(angle)) * depth) / .8]
        attrs = ObjectAttributes.from_label(
            label, confidence=1.0, xy=image_xy(*xy), footprint=footprint,
            height=item.get("height"), z_offset=float(item.get("z_offset", 0.0)),
            yaw_degrees=float(item.get("yaw_degrees", 0.0)), source=item.get("source"),
            model_id=str(item.get("model_id")), kind=item.get("kind"),
            extra={"geometry_frame": "image_normalized", "geometry_units": "normalized"},
        )
        if label in {"monitor", "laptop", "printer"}:
            attrs.movable = False
            attrs.portable = False
        state = "on desk"
        support_id = "table"
        if (item.get("relation") or {}).get("predicate") == "on":
            state = "on " + str(item["relation"]["object"])
            support_id = str(item["relation"]["object"]).replace(" ", "_") + "_sam3"
        objects.append(SceneObject(object_id, label, state=state, attributes=attrs,
                                   metadata={"planning_pose": {
                                       "xy": list(attrs.xy), "frame": "image_normalized",
                                       "yaw_degrees": attrs.yaw_degrees, "support": support_id}}))
    relations = []
    for item in placed:
        rel = item.get("relation") or {}
        if rel.get("predicate") == "on":
            relations.append(Relation(
                f"{str(rel['subject']).replace(' ', '_')}_sam3",
                f"{str(rel['object']).replace(' ', '_')}_sam3", "on", confidence=1.0,
            ))
    return SceneGraph(objects=objects, relations=relations,
                      image_path=str(SCENES / scene["image"]),
                      instruction="Organize the desk.", version=1,
                      metadata={"source": "scenes_constrained_v2_manifest",
                                "terminal_target": terminal,
                                "geometry_source": "manifest_projection_proxy_not_image_detection"})


def prepare_inputs(output, *, max_actions=None, max_states=None, branches=3):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((SCENES / "manifest.json").read_text())
    paths = []
    for index, scene in enumerate(manifest["scenes"][:10]):
        name = f"scene_{index:03d}"
        directory = output / name / "inputs"
        directory.mkdir(parents=True, exist_ok=True)
        initial = graph_from_manifest(scene)
        terminal = graph_from_manifest(scene, terminal=True)
        actions, states = planning_budgets(initial, max_actions, max_states, branches)
        (directory / "planning_budget.json").write_text(json.dumps(
            {"max_actions": actions, "max_states": states}, indent=2) + "\n")
        initial.to_json(directory / "initial_scene_graph.json")
        terminal.to_json(directory / "terminal_scene_graph.json")
        (directory / "scene_manifest.json").write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n")
        paths.append((name, directory / "initial_scene_graph.json", directory / "terminal_scene_graph.json"))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "constract/runs" /
                        datetime.now().strftime("scenes_constrained_v2_%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--config", type=Path, default=ROOT / "config/goal_generator.local.json")
    parser.add_argument("--max-states", type=int, default=None, help="Default: scene-dependent search budget")
    parser.add_argument("--max-actions", type=int, default=None, help="Default: scene-dependent action budget")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--num-branches", type=int, choices=(3, 4, 5), default=3)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=300,
                        help="Per external API request timeout in seconds")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()
    if args.parallel < 1 or args.beam_width < 1 or args.timeout <= 0:
        parser.error("Concurrency, beam width and timeout must be positive")
    paths = prepare_inputs(args.output, max_actions=args.max_actions,
                           max_states=args.max_states, branches=args.num_branches)
    (args.output / "batch_config.json").write_text(json.dumps({
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }, indent=2) + "\n")
    if not args.run:
        print(json.dumps({"prepared": len(paths), "output": str(args.output)}, ensure_ascii=False))
        return 0
    results = []
    def launch(item):
        name, initial, terminal = item
        run_dir = args.output / name / "comparison"
        budget = json.loads((initial.parent / "planning_budget.json").read_text())
        command = [sys.executable, "-m", "constract.compare", "--config", str(args.config),
                   "--initial", str(initial), "--terminal", str(terminal),
                   "--output-dir", str(run_dir), "--max-states", str(budget["max_states"]),
                   "--max-actions", str(budget["max_actions"]), "--beam-width", str(args.beam_width),
                   "--num-branches", str(args.num_branches), "--max-output-tokens", str(args.max_output_tokens)]
        if args.timeout is not None:
            command += ["--timeout", str(args.timeout)]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        (args.output / name / "batch_stdout.log").write_text(completed.stdout + completed.stderr)
        comparison = run_dir / "comparison.json"
        data = json.loads(comparison.read_text()) if comparison.exists() else {"results": {}}
        results.append({"scene": name, "returncode": completed.returncode,
                        "results": data.get("results", {})})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    if args.parallel <= 1:
        for item in paths:
            launch(item)
    else:
        # Keep each compare process independent while bounding API pressure.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=min(args.parallel, len(paths))) as pool:
            futures = [pool.submit(launch, item) for item in paths]
            for future in as_completed(futures):
                future.result()
    summary = {"scene_count": len(results), "scenes": results,
               "target_source": "manifest-derived compact target; not human scene-by-scene annotation",
               "input_directory": str(SCENES)}
    (args.output / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
