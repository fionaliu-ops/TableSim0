"""Paper baselines on cached observations; no simulated or physical execution."""
import argparse
import json
from pathlib import Path

from baseline import InnerMonologuePlanner, SayPlanPlanner, LERaPlanner, PaperObservation
from baseline.paper_common import describe_scene
from high_level_planner.reasoning_graph.language_models.openai_compatible import OpenAICompatibleLanguageModel
from high_level_planner.scene_graph.graph import SceneGraph
from script.goal_config import dynamic_reasoning_config, load_api_settings


PLANNERS = {'inner_monologue': InnerMonologuePlanner, 'sayplan': SayPlanPlanner, 'lera': LERaPlanner}


def make_planner(method, *, model, initial_scene, target_scene, target_image=None,
                 output_dir=None, **kwargs):
    catalog = {obj.id: obj.name for obj in initial_scene.objects}
    if set(catalog) != {obj.id for obj in target_scene.objects}:
        raise ValueError('Initial/goal object IDs must agree; align identity before running baselines')
    planner_class = PLANNERS[method]
    options = dict(model=model, object_catalog=catalog,
        goal_description=json.dumps(describe_scene(target_scene), ensure_ascii=False),
        target_image=target_image, audit_dir=Path(output_dir)/'method_audit' if output_dir else None, **kwargs)
    if method == 'sayplan': options['target_scene'] = target_scene
    return planner_class(**options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=PLANNERS, required=True)
    parser.add_argument('--scene-graph', type=Path, required=True)
    parser.add_argument('--target-scene-graph', type=Path, required=True)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--target-image', type=Path)
    parser.add_argument('--initial-plan', type=Path, help='JSON ACTION list, for recovery-only comparison')
    parser.add_argument('--replay', type=Path, help='JSON list of {success, image?, scene_graph?, fatal?}, one actual observation per prior decision')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--instruction', default='整理桌面')
    parser.add_argument('--max-model-calls', type=int, default=200)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.method == 'lera' and args.image is None: parser.error('LERa requires --image')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new output directory to preserve run evidence')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial, target = SceneGraph.from_json(args.scene_graph), SceneGraph.from_json(args.target_scene_graph)
    config = dynamic_reasoning_config(load_api_settings(args.config), args.output_dir/'model_audit')
    planner = make_planner(args.method, model=OpenAICompatibleLanguageModel(config), initial_scene=initial,
        target_scene=target, target_image=args.target_image, output_dir=args.output_dir,
        max_model_calls=args.max_model_calls)
    shared_plan = json.loads(args.initial_plan.read_text()) if args.initial_plan else None
    decisions = []
    try:
        output = planner.start(PaperObservation(args.image, initial), args.instruction, initial_plan=shared_plan)
        decisions.append(output.to_dict())
        events = json.loads(args.replay.read_text()) if args.replay else []
        def replay_path(value):
            path = Path(value)
            return path if path.is_absolute() else args.replay.parent / path
        for event in events:
            if output.status != 'action': raise ValueError('Replay has feedback after planner stopped')
            planner.record_execution_result(output.next_subtask, success=event['success'],
                details=event.get('details', {}), fatal=event.get('fatal', False))
            image = replay_path(event['image']) if event.get('image') else None
            scene = SceneGraph.from_json(replay_path(event['scene_graph'])) if event.get('scene_graph') else None
            output = planner.step(PaperObservation(image, scene))
            decisions.append(output.to_dict())
    finally:
        (args.output_dir/'replay_result.json').write_text(json.dumps({
            'mode': 'cached_observation_replay', 'physical_execution_verified': False,
            'decisions': decisions, 'metrics': planner.metrics,
            'simulator_history': getattr(planner, 'simulator_history', [])}, ensure_ascii=False, indent=2))
    print(json.dumps({'status': output.status, 'metrics': planner.metrics}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
