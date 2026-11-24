"""Run IM/SayPlan/LERa with TableSim's shared camera-grounded RGB-D executor."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from baseline.paper_demo import PLANNERS, make_planner
from baseline.paper_common import PaperObservation
from baseline.paper_runner import run_paper_episode, observed_action_success
from high_level_planner.evaluation import EpisodeMetrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=PLANNERS, required=True)
    parser.add_argument('--scene', default='scene_003')
    parser.add_argument('--task-config', default='scene_003')
    parser.add_argument('--target-scene-graph', type=Path, required=True)
    parser.add_argument('--goal-image', type=Path, required=True)
    parser.add_argument('--initial-plan', type=Path)
    parser.add_argument('--instruction', default='整理桌面')
    parser.add_argument('--api-config', type=Path)
    parser.add_argument('--sam3-endpoint', default='http://127.0.0.1:8003')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=30)
    parser.add_argument('--max-model-calls', type=int, default=200)
    parser.add_argument('--position-tolerance', type=float, default=.02)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_steps < 1 or args.max_model_calls < 1: parser.error('Budgets must be positive')
    if not 0 <= args.position_tolerance < float('inf'): parser.error('Invalid position tolerance')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new output directory to preserve run evidence')
    from script.dynamic_sim import build_environment
    from script.rgbd_execution import RGBDAdapter, RGBDGraspExecutor, RGBDSceneGraphConstructor
    from script.compare_scene003_modalities import install_recording
    from script.goal_config import dynamic_reasoning_config, load_api_settings
    from script.dual_arm_safety import ArmClearanceError, TrajectoryClearanceError
    from high_level_planner.perception.sam3_client import Sam3PerceptionClient
    from high_level_planner.reasoning_graph.language_models.openai_compatible import OpenAICompatibleLanguageModel
    from high_level_planner.reasoning_graph.target_conditioned import TargetGraphEvaluator
    from high_level_planner.scene_graph.graph import SceneGraph
    from high_level_planner.scene_graph.updater import SceneGraphUpdater
    from high_level_planner.scene_graph.label_review import VisualLabelReviewer

    target = SceneGraph.from_json(args.target_scene_graph)
    goal_hash = hashlib.sha256(args.goal_image.read_bytes()).hexdigest()
    expected_hash = target.metadata.get('derived_from_image_sha256')
    if expected_hash and expected_hash != goal_hash: parser.error('Target graph/image hash mismatch')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = dynamic_reasoning_config(load_api_settings(args.api_config), args.output_dir/'model_audit')
    base = build_environment(args.scene, args.task_config, args.output_dir)
    env = install_recording(RGBDAdapter(base.task, base.settings, base.output_dir))
    env.graph_validation_mode = env.symbolic_grasp_validation = True

    class SensorPerception(Sam3PerceptionClient):
        def detect(self, image, *, instruction=None):
            key = str(Path(image).resolve())
            cache = getattr(self, '_cache', {})
            if key not in cache:
                detected = super().detect(image, instruction=instruction)
                width, height = detected.image_size
                border = max(2., .005*min(width, height))
                detected.detections = [d for d in detected.detections
                    if min(d.bbox[0], d.bbox[1], width-d.bbox[2], height-d.bbox[3]) > border]
                geometry = getattr(env, 'sensor_packets', {}).get(key)
                cache[key] = geometry.filter(detected) if geometry else detected
                self._cache = cache
            return deepcopy(cache[key])

    perception = SensorPerception(endpoint=args.sam3_endpoint, confidence_threshold=.12)
    env.perception = perception
    constructor = RGBDSceneGraphConstructor(sensor_packets=lambda: getattr(env, 'sensor_packets', {}),
        perception_backend=perception, label_reviewer=VisualLabelReviewer(
            OpenAICompatibleLanguageModel(replace(config, audit_dir=str(args.output_dir/'label_model_audit'))),
            audit_dir=args.output_dir/'label_review', preserve_uncertain=True))
    updater = SceneGraphUpdater(constructor)
    observed = None
    initial = None
    last_moved = None
    observation_count = 0

    def observe():
        nonlocal observed, initial, observation_count, last_moved
        frame = env.observe()
        scene = constructor.construct(frame['image'], instruction=args.instruction) if observed is None else (
            updater.update_scene_graph(observed.scene, frame['image'], instruction=args.instruction, moved_object_id=last_moved))
        last_moved = None
        if initial is None: initial = deepcopy(scene)
        if target.metadata.get('yaw_semantics') == 'visual_delta_from_initial_reference_degrees':
            from script.online_rearrangement import refine_visual_rotations
            refine_visual_rotations(initial, scene, initial.image_path, frame['image'], perception,
                audit_path=args.output_dir/f'visual_rotation_{observation_count:03d}.json')
        scene.to_json(args.output_dir/f'observed_scene_{observation_count:03d}.json')
        observation_count += 1
        env.bind_graph(scene)
        observed = PaperObservation(frame['image'], scene)
        return observed

    def execute(request):
        nonlocal last_moved
        try:
            env.step(request)
            last_moved = request.object_target['source']
            return {'success': True}
        except Exception as exc:
            fatal = isinstance(exc, ArmClearanceError) and not isinstance(exc, TrajectoryClearanceError)
            details = {k: deepcopy(v) for k, v in getattr(exc, 'details', {}).items()
                       if k in {'failure_category', 'phase', 'blocking_object_ids', 'object_moved'}}
            details['error_type'] = type(exc).__name__
            if env.held is not None and not fatal:
                try:
                    env.symbolic_grasp_controller.recover_held()
                    details['recovery'] = 'shared_controller_return_and_release'
                except Exception as recovery:
                    details['recovery_error_type'] = type(recovery).__name__
                    fatal = True
            return {'success': False, 'fatal': fatal, 'details': details}

    episode_metrics = EpisodeMetrics(method=args.method, instruction=args.instruction, output_dir=args.output_dir)
    episode_metrics.context = {'scene': args.scene, 'seed': args.seed, 'max_steps': args.max_steps,
                              'max_model_calls': args.max_model_calls, 'goal_image_sha256': goal_hash}
    try:
        env.reset(args.seed)
        env.symbolic_grasp_controller = RGBDGraspExecutor(env)
        env.symbolic_grasp_controller.install()
        env.symbolic_grasp_controller.yaw_semantics = target.metadata.get('yaw_semantics', 'world_absolute_degrees')
        env.symbolic_grasp_controller.clear_camera_view()
        first = observe()
        evaluator = TargetGraphEvaluator(first.scene, target, args.position_tolerance)
        def evaluate(observation):
            report = evaluator.evaluate(observation.scene, args.instruction)
            identity_ready = all(o.metadata.get('visual_identity_confirmed') is not False for o in observation.scene.objects)
            return {'success': bool(report.terminal and env.held is None and identity_ready),
                    'goal_checks': report.goal_checks, 'payload_released': env.held is None,
                    'identity_ready': identity_ready, 'source': 'fresh_camera_scene_graph'}
        planner = make_planner(args.method, model=OpenAICompatibleLanguageModel(config), initial_scene=first.scene,
            target_scene=target, target_image=args.goal_image, output_dir=args.output_dir,
            max_steps=args.max_steps, max_model_calls=args.max_model_calls,
            **({'position_tolerance': args.position_tolerance} if args.method == 'sayplan' else {}))
        plan = json.loads(args.initial_plan.read_text()) if args.initial_plan else None
        manifest = {'method': args.method, 'adapted_method': True, 'scene': args.scene, 'seed': args.seed,
            'model': config.model, 'temperature': config.temperature,
            'max_output_tokens': config.max_output_tokens, 'goal_image_sha256': goal_hash,
            'target_graph_sha256': hashlib.sha256(args.target_scene_graph.read_bytes()).hexdigest(),
            'initial_plan_sha256': hashlib.sha256(args.initial_plan.read_bytes()).hexdigest() if args.initial_plan else None,
            'max_steps': args.max_steps, 'max_model_calls': args.max_model_calls,
            'executor': 'shared_rgbd_symbolic_grasp', 'contact_grasp_physics_verified': False,
            'simulation_truth_scope': 'none_rgbd_and_robot_encoders',
            'initial_plan_source': 'supplied' if plan else 'method_specific',
            'position_tolerance': args.position_tolerance, 'goal_revision_enabled': False}
        (args.output_dir/'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        # Return the already captured first observation exactly once.
        pending = [first]
        def next_observation(): return pending.pop() if pending else observe()
        result = run_paper_episode(planner, observe=next_observation, execute=execute, evaluate=evaluate,
            instruction=args.instruction, initial_plan=plan, max_steps=args.max_steps,
            target_scene=target, position_tolerance=args.position_tolerance,
            episode_metrics=episode_metrics,
            check_action=lambda a,b,e,f: observed_action_success(a,b,e,f,position_tolerance=args.position_tolerance),
            output_path=args.output_dir/'episode.json')
        print(json.dumps({'status': result['status'], 'metrics': result['metrics'],
                          'evaluation_metrics': result['evaluation_metrics']}, ensure_ascii=False))
        return 0 if result['status'] == 'external_success' else 1
    except Exception as exc:
        episode_metrics.errors.append(f'{type(exc).__name__}: {exc}')
        episode_metrics.write('episode_error')
        raise
    finally:
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())
