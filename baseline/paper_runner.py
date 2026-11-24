"""Shared physical-observation runner; no planner completion is a certificate."""
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from .paper_common import describe_scene, positive_integer
from high_level_planner.evaluation import EpisodeMetrics


def observed_action_success(before, after, edge, feedback, *, position_tolerance=.02, yaw_tolerance=5.):
    """Common task checker using only new observed facts, not predicted successors."""
    from high_level_planner.reasoning_graph.geometry import pose
    from high_level_planner.reasoning_graph.feasibility import articulation_state
    if feedback['success'] is not True or after.scene is None: return False
    scene = after.scene
    obj = scene.get_object(edge.source_object_id)
    if obj is None or obj.metadata.get('observed_in_current_frame') is False or obj.metadata.get('visual_identity_confirmed') is False:
        return False
    kind, params, target = edge.action_type, edge.parameters, edge.object_target.get('target')
    if kind in {'move', 'place'} and 'pose' in params:
        actual, expected = pose(scene, obj.id), params['pose']
        return bool(actual and actual['frame'] == expected['frame']
            and math.dist(actual['xy'], expected['xy']) <= position_tolerance
            and ('yaw_degrees' not in expected or obj.attributes.yaw_degrees is not None and
                 abs((obj.attributes.yaw_degrees-expected['yaw_degrees']+180)%360-180) <= yaw_tolerance))
    relation = {'stack': 'on', 'insert': 'inside', 'attach': 'attached_to'}.get(kind)
    if kind == 'place' and params.get('support_placement'): relation = 'on'
    if relation:
        return any(r.source == obj.id and r.target == target and r.relation in
            ({'inside', 'in'} if relation == 'inside' else {relation}) for r in scene.relations)
    if kind in {'align', 'reorient'}:
        expected = params.get('target_yaw_degrees')
        if expected is not None:
            return obj.attributes.yaw_degrees is not None and abs((obj.attributes.yaw_degrees-expected+180)%360-180) <= yaw_tolerance
        return bool(target and scene.find_relation(obj.id, target, 'aligned_with'))
    if kind in {'open', 'close'}:
        return articulation_state(obj) == ('open' if kind == 'open' else 'closed')
    return False


def run_paper_episode(planner, *, observe, execute, evaluate, instruction,
                      initial_plan=None, check_action=observed_action_success,
                      max_steps=50, output_path=None, target_scene=None, position_tolerance=.02,
                      episode_metrics=None):
    """observe()->PaperObservation; execute(VLARequest)->{success,details,fatal}.

    evaluate(observation)->{success:bool,...} is independent of the planner;
    the simulation adapter must also check released payload/identity uncertainty.
    All callbacks share the same environment for all three methods.
    """
    positive_integer(max_steps, 'max_steps')
    planner.reset()
    started = time.perf_counter()
    evaluation_metrics = episode_metrics or EpisodeMetrics(method=planner.method, instruction=instruction,
        output_dir=Path(output_path).parent if output_path else None)
    result = {'method': planner.method, 'status': 'running', 'steps': [],
              'external_evaluation': {'success': False}, 'robot_execution_verified': False}
    def save():
        result.update(metrics=planner.metrics, total_seconds=time.perf_counter()-started)
        if hasattr(planner, 'simulator_history'): result['simulator_history'] = deepcopy(planner.simulator_history)
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(path)
    try:
        with evaluation_metrics.phase('perception'):
            observed = observe()
        output = None
        for index in range(max_steps + 1):
            external = evaluate(observed)
            if not isinstance(external, dict) or not isinstance(external.get('success'), bool):
                raise ValueError('Evaluator must supply an independent boolean success')
            result['external_evaluation'] = external
            if index == 0 and target_scene is not None:
                evaluation_metrics.configure(observed.scene, target_scene, position_tolerance)
            evaluation_metrics.observe(observed.scene, evaluation=external,
                                       payload_released=external.get('payload_released'))
            if external['success']:
                result['status'] = 'external_success'
                break
            if index == max_steps:
                result['status'] = 'step_limit'
                break
            with evaluation_metrics.phase('planning'):
                output = planner.start(observed, instruction, initial_plan=initial_plan) if index == 0 else planner.step(observed)
            record = {'index': index, 'decision': output.to_dict(), 'external_before': deepcopy(external)}
            result['steps'].append(record)
            save()
            if output.status != 'action':
                result['status'] = 'false_completion' if output.status == 'done' else 'planner_blocked'
                break
            tick = time.perf_counter()
            try:
                feedback = execute(output.vla_request)
            except Exception as exc:
                evaluation_metrics.execution(success=False, seconds=time.perf_counter()-tick,
                                             details={'error_type': type(exc).__name__})
                raise
            record['execution_seconds'] = time.perf_counter() - tick
            if not isinstance(feedback, dict) or not isinstance(feedback.get('success'), bool):
                raise ValueError('Executor must supply a boolean success')
            if not isinstance(feedback.get('fatal', False), bool): raise ValueError('fatal must be boolean')
            evaluation_metrics.execution(success=feedback['success'], seconds=record['execution_seconds'],
                details=feedback.get('details'), fatal=feedback.get('fatal', False))
            if feedback.get('disturbance') is not None:
                evaluation_metrics.disturbance(feedback['disturbance'])
            record['controller_feedback'] = deepcopy(feedback)
            with evaluation_metrics.phase('perception'):
                following = observe()
            success = check_action(observed, following, output.next_subtask, feedback)
            if not isinstance(success, bool): raise ValueError('Task checker must return boolean')
            evaluation_metrics.verify_action(success)
            record['task_checker_success'] = success
            record['observed_after'] = str(following.image) if following.image is not None else None
            planner.record_execution_result(output.next_subtask, success=success,
                details=feedback.get('details', {}), fatal=feedback.get('fatal', False))
            observed = following
            save()
            if feedback.get('fatal', False):
                result['status'] = 'safety_stop'
                result['external_evaluation'] = evaluate(observed)
                evaluation_metrics.observe(observed.scene, evaluation=result['external_evaluation'],
                    payload_released=result['external_evaluation'].get('payload_released'))
                break
    except Exception as exc:
        result.update(status='episode_error', error=f'{type(exc).__name__}: {exc}')
        evaluation_metrics.errors.append(result['error'])
    finally:
        result['evaluation_metrics'] = evaluation_metrics.write(result['status'])
        save()
    return result
