"""Diagnose scene 003 execution independently of language-model planning.

The probe uses the live SAM3 initial observation for pixel/world grounding,
then asks the normal dynamic adapter to execute a collision-free layout which
satisfies scene_003.check_success().  It does not alter object definitions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from high_level_planner.perception.sam3_client import Sam3PerceptionClient
from high_level_planner.scene_graph.constructor import SceneGraphConstructor
from script.compare_scene003_modalities import install_recording, match_book_instances, request_from_action, write
from script.dynamic_sim import build_environment


PLAN = [
    ("phone", [-0.045, -0.240], "temporary buffer"),
    ("keyboard", [0.220, -0.200], "temporary buffer"),
    # The mouse initially blocks the cup destination.  A temporary buffer
    # action is necessary before placing the compact final layout.
    ("mouse", [0.280, -0.100], "temporary buffer"),
    ("cup", [0.180, -0.050], "left-arm handoff"),
    ("cup", [0.270, 0.100], "final"),
    ("phone", [0.080, -0.230], "final"),
    ("book", [-0.155, 0.025], "stack on folder"),
    ("mouse", [0.300, -0.030], "second buffer"),
    ("keyboard", [0.100, 0.125], "final"),
    ("mouse", [0.120, -0.050], "final"),
]


def image_xy(transform, world_xy):
    linear = np.asarray(transform[:2], dtype=float)
    bias = np.asarray(transform[2], dtype=float)
    return ((np.asarray(world_xy, dtype=float) - bias) @ np.linalg.inv(linear)).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-graph", type=Path, help="Explicit cached initial perception for execution diagnostics")
    parser.add_argument("--sam3-endpoint", default="http://127.0.0.1:8003")
    parser.add_argument('--align-yaw', action='store_true')
    parser.add_argument('--goal-image', type=Path)
    parser.add_argument('--resume-plan-index', type=int, help='Explicit first plan index for suffix diagnostics')
    parser.add_argument('--fixture-step', type=int, help='Last completed source step to restore for a suffix-only diagnostic')
    parser.add_argument('--prefix-fixture', type=Path,
                        help='Execution-only diagnostic: restore poses before book pickup from an earlier probe; never a full-flow result')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = install_recording(build_environment("scene_003", "scene_003", args.output_dir))
    trace = {"purpose": "execution_isolation", "plan": PLAN, "steps": []}
    try:
        observation = env.reset(0)
        if args.initial_graph and args.prefix_fixture:
            from high_level_planner.scene_graph.graph import SceneGraph
            cached = SceneGraph.from_json(args.initial_graph)
            env.bind_graph(cached, match_book_instances(cached, env.task))
        plan = PLAN
        if args.goal_image:
            from script.scene003_image_goal import Scene003ImageGoal
            env.image_goal = Scene003ImageGoal(env.task, args.goal_image)
            trace['image_goal_targets'] = {k: v.tolist() for k,v in env.image_goal.targets.items()}
        if args.align_yaw:
            plan = [*PLAN, ('stapler', [-.280, -.130], 'final')]
        if env.image_goal is not None:
            plan = [(n, env.image_goal.targets[n].tolist() if phase == 'final' and i >= 4 and n != 'phone' else xy, phase) for i,(n,xy,phase) in enumerate(plan)]
            plan.append(('phone', env.image_goal.targets['phone'].tolist(), 'final'))
        trace["plan"] = plan
        if args.prefix_fixture:
            import sapien
            source = json.loads(args.prefix_fixture.read_text())
            prefix = (source['steps'][args.fixture_step] if args.fixture_step is not None else next(step for step in source['steps'] if step['object']=='cup' and step['phase']=='final' and step['status']=='executed'))
            if prefix['status'] != 'executed':
                raise ValueError('Fixture step must be a completed action')
            for name, pose in prefix['object_poses'].items():
                env.task.objects[name].actor.set_pose(sapien.Pose(pose[:3], pose[3:]))
                env._clear_attachment_velocity(name)
            start_index = args.resume_plan_index if args.resume_plan_index is not None else ((args.fixture_step + 1) if args.fixture_step is not None else 5)
            trace["resume_plan_index"] = start_index
            plan = plan[start_index:]
            book = env.task.objects['book'].get_pose()
            folder = env.task.objects['folder'].get_pose()
            if np.linalg.norm(book.p[:2] - folder.p[:2]) <= .04:
                env._pin_released_pose('book', book)
            trace['purpose']='suffix_execution_diagnostic_from_pose_fixture'
            trace['prefix_fixture']=str(args.prefix_fixture)
            observation=env.observe()
        if args.initial_graph:
            from high_level_planner.scene_graph.graph import SceneGraph
            graph = SceneGraph.from_json(args.initial_graph)
            trace['perception_source'] = str(args.initial_graph)
            trace['live_sam3_called'] = False
        else:
            graph = SceneGraphConstructor(
                perception_backend=Sam3PerceptionClient(endpoint=args.sam3_endpoint)
            ).construct(observation["image"], instruction="Observe all tabletop objects")
            trace['live_sam3_called'] = True
        graph.to_json(args.output_dir / "initial_sam3_scene_graph.json")
        if not (args.initial_graph and args.prefix_fixture):
            env.bind_graph(graph, match_book_instances(graph, env.task))
        # The live contact monitor rejects actual contact.  Two centimetres is
        # enough planning clearance for these measured collision footprints.
        env.thin_placement_margin = 0.01
        env.max_placement_deviation = 0.0
        default_arm = env._arm
        # SAM3 may omit the partly occluded stapler. Explicit simulator IDs are
        # only used by this execution diagnostic, never passed off as detections.
        for name in env.task.objects:
            if name not in env.object_map.values():
                env.object_map['sim_' + name] = name
        ids = {name: object_id for object_id, name in env.object_map.items()}
        for name, target, phase in plan:
            if name == "cup":
                cup_arm = "left" if phase == "left-arm handoff" else "right"
                env._arm = lambda object_name, selected=cup_arm: selected if object_name == "cup" else default_arm(object_name)
            else:
                env._arm = default_arm
            env.max_placement_deviation = (0.08 if phase == "left-arm handoff"
                                           or name == "keyboard"
                                           or (name == "cup" and phase == "final")
                                           or (name == "mouse" and phase in {"final", "second buffer"})
                                           or (name == "phone" and phase == "final") else 0.0)
            if env.image_goal is not None and np.linalg.norm(np.asarray(target) - env.image_goal.targets[name]) < .001:
                env.max_placement_deviation = .024
            normalized = image_xy(env.transform, target)
            action_type = ("stack" if phase == "stack on folder" else
                           "push" if phase == "physical push" else "move")
            action_target = ids["folder"] if phase == "stack on folder" else "table"
            action = {"action_type": action_type, "source_object_id": ids[name],
                      "target": action_target, "pose": {"frame": "image_normalized",
                      "xy": normalized, "support": action_target},
                      "reason": "execution isolation probe"}
            if args.align_yaw and phase in {'final', 'stack on folder'} and name != 'cup':
                action['pose']['yaw_degrees'] = -90.0 if name in {'book', 'stapler'} else 0.0
            item = {"object": name, "arm": env._arm(name), "phase": phase, "world_target": target,
                    "normalized_target": normalized, "status": "executing"}
            trace["steps"].append(item); write(args.output_dir / "execution_probe.json", trace)
            observation = env.step(request_from_action(action))
            item.update(status="executed", placement=observation.get("placement"),
                        object_poses=observation["object_poses"])
            write(args.output_dir / "execution_probe.json", trace)
        env.park_arms()
        trace["fixed_scene_success"] = bool(env.evaluate())
        if args.align_yaw:
            trace['orientation_checks'] = env.orientation_status({'phone':0., 'keyboard':0., 'mouse':0., 'book':-90., 'folder':0., 'stapler':-90.})
        trace['orientation_success'] = all(c['passed'] for c in trace.get('orientation_checks', {}).values())
        trace['image_goal'] = env.image_goal.evaluate(env) if env.image_goal else None
        trace["status"] = "success" if trace['orientation_success'] and (trace['image_goal']['passed'] if env.image_goal else trace['fixed_scene_success']) else "goal_not_reached"
    except Exception as exc:
        trace.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if env.arm_safety is not None:
            trace['arm_safety'] = dict(env.arm_safety.stats)
        if env.image_goal is not None:
            trace['image_goal'] = env.image_goal.evaluate(env)
        try:
            trace["final_object_poses"] = env.observe()["object_poses"]
        except Exception as exc:
            trace["final_observation_error"] = f"{type(exc).__name__}: {exc}"
        env.close()
        write(args.output_dir / "execution_probe.json", trace)
    print(json.dumps({k: trace.get(k) for k in ("status", "fixed_scene_success", "error")},
                     ensure_ascii=False), flush=True)
    return 0 if trace.get("status") == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
