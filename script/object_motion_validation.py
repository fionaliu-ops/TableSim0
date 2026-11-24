"""Continuous object-motion execution for isolating a visual high-level plan.

This intentionally bypasses robot IK. Simulator truth is restricted to this
executor's geometry and camera grounding; visual observers never receive it.
"""
from __future__ import annotations

import json
import math
from copy import deepcopy

import numpy as np


def interpolated_poses(start, finish, timestep, speed=.12, angular_speed=math.radians(45)):
    """Yield bounded continuous poses, including the exact final endpoint."""
    start, finish = np.asarray(start, float).copy(), np.asarray(finish, float).copy()
    start[3:] /= np.linalg.norm(start[3:]); finish[3:] /= np.linalg.norm(finish[3:])
    if np.dot(start[3:], finish[3:]) < 0:
        finish[3:] *= -1
    angle = 2*math.acos(np.clip(np.dot(start[3:], finish[3:]), 0, 1))
    duration = max(.4, np.linalg.norm(finish[:3]-start[:3])/speed, angle/angular_speed)
    # Cubic ease-in/out has a maximum derivative of 1.5.
    count = max(1, math.ceil(1.5*duration/timestep))
    for t in np.linspace(1/count, 1, count):
        alpha = t*t*(3-2*t)
        pose = start+alpha*(finish-start)
        pose[3:] /= np.linalg.norm(pose[3:])
        yield pose


class ObjectMotionExecutor:
    def __init__(self, env):
        self.env, self.task = env, env.task
        self.audit = env.output_dir / 'object_motion.jsonl'

    def log(self, **record):
        with self.audit.open('a') as stream:
            stream.write(json.dumps({'step': self.env.step_index, **record})+'\n')

    def world_point(self, normalized_xy, height):
        config = self.task.cameras.get_config()['head_camera']
        intrinsic, extrinsic = np.asarray(config['intrinsic_cv']), np.asarray(config['extrinsic_cv'])
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack((extrinsic, [0, 0, 0, 1]))
        camera = np.linalg.inv(extrinsic)
        rgb = self.task.get_obs()['observation']['head_camera']['rgb']
        size = np.array([rgb.shape[1], rgb.shape[0]])
        ray = camera[:3, :3] @ np.linalg.solve(intrinsic, np.r_[np.asarray(normalized_xy)*size, 1])
        if abs(ray[2]) < 1e-8:
            raise ValueError('Camera ray does not intersect the placement plane')
        return camera[:3, 3]+ray*((height-camera[2, 3])/ray[2])

    def place(self, action, name):
        import sapien
        env, task = self.env, self.task
        actor = task.objects[name]
        parameters = action.get('parameters', {})
        placement = parameters.get('pose') or parameters.get('relative_pose') or {}
        if not placement:
            placement = parameters.get('placement_target', {}).get('parameters', {}).get('pose', {})
        observed = deepcopy(actor.get_pose())
        target_name = action.get('target')
        support = env._name(target_name) if target_name and target_name != 'table' else None
        if support:
            support_pose = task.objects[support].get_pose()
            bottom = float(support_pose.p[2]+task.object_geometry[support]['height']+.004)
        else:
            bottom = float(task._table_z()+.006)
        height = float(task.object_geometry[name]['height'])
        if placement.get('frame') == 'image_normalized' and placement.get('xy') is not None:
            center = self.world_point(placement['xy'], bottom+height/2)
            target_xy = center[:2]
        elif support:
            target_xy = np.asarray(task.objects[support].get_pose().p[:2])
        elif action['kind'] in {'align', 'reorient'}:
            target_xy, bottom = np.asarray(observed.p[:2]), float(observed.p[2])
        else:
            raise ValueError('Graph action lacks a visual placement target')
        orientation = np.asarray(observed.q)
        yaw = placement.get('yaw_degrees', parameters.get('target_yaw_degrees'))
        if yaw is not None:
            old_yaw = math.degrees(env._object_yaw(name))
            orientation = task._yaw_quat(float(yaw)-old_yaw, base_quat=orientation)
        target = np.r_[target_xy, bottom, orientation]
        start = np.r_[observed.p, observed.q]
        # Recompute clearance from the live scene, never from saved buffers.
        tops = [float(obj.get_pose().p[2]+task.object_geometry[key]['height'])
                for key, obj in task.objects.items() if key != name]
        clearance = max(.04, .5*height)
        travel_z = max(start[2], bottom, max(tops, default=bottom))+clearance
        lift = start.copy(); lift[2] = travel_z
        transfer = target.copy(); transfer[2] = travel_z
        setter = getattr(actor, 'actor', actor)
        for component in setter.get_components():
            if hasattr(component, 'set_kinematic'):
                component.set_kinematic(True)
        previous = start.copy()
        ticks, max_delta = 0, 0.
        for phase, destination in [('lift', lift), ('transfer', transfer), ('lower', target)]:
            for sample in interpolated_poses(previous, destination, task.scene.get_timestep()):
                max_delta = max(max_delta, float(np.linalg.norm(sample[:3]-previous[:3])))
                setter.set_pose(sapien.Pose(sample[:3], sample[3:]))
                env._clear_attachment_velocity(name)
                task.scene.step()
                task._sync_attached_actor()  # Recording hook; robot remains stationary.
                previous = sample
                ticks += 1
            env._capture_motion_frame('object_'+phase)
        env.last_placement = {'object': name, 'mode': 'continuous_object_motion_no_robot_ik',
                              'requested_xy': target_xy.tolist(), 'actual_xy': list(map(float, actor.get_pose().p[:2])),
                              'placement_error': float(np.linalg.norm(actor.get_pose().p[:2]-target_xy)),
                              'robot_grasp_verified': False, 'simulation_ticks': ticks,
                              'max_translation_per_tick_m': max_delta}
        self.log(action=action, **env.last_placement)


def run_video_path(env, planner, initial, target, graph, search, instruction):
    """Execute the complete generated path; report visual evidence separately.

An inconclusive observation does not truncate the requested demonstration.
It also never promotes a predicted node to a verified observed state.
"""
    from script.dynamic_sim import LoopResult
    from high_level_planner.reasoning_graph.node_localization import locate_reasoning_node
    from high_level_planner.reasoning_graph.target_conditioned import TargetGraphEvaluator
    if not search.path or search.goal_state is None:
        raise RuntimeError('A generated path to GOAL is required for complete video validation')
    current, records = initial, []
    symbolic_grasp = bool(getattr(env, 'symbolic_grasp_validation', False))
    execution_model = 'symbolic_gripper_follow' if symbolic_grasp else 'continuous_object_motion_no_robot_ik'
    prefix = 'symbolic_grasp' if symbolic_grasp else 'object_motion'
    trace_path = env.output_dir/'dynamic_trace.json'
    def save(status):
        payload = {'status': status, 'execution_model': execution_model,
                   'high_level_model': graph.metadata.get('model_execution'),
                   'planned_actions': len(search.path), 'steps': records,
                   'simulation_truth_scope': 'execution_only', 'robot_grasp_verified': False,
                   'robot_motion_executed': symbolic_grasp,
                   'observed_goal': False, 'reasoning_goal': True}
        trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    for index, edge in enumerate(search.path):
        request = planner.vla_interface.format_subtask(edge)
        record = {'index': index, 'reasoning_from': edge.from_state, 'reasoning_to': edge.to_state,
                  'request': request.to_dict(), 'status': 'pending'}
        records.append(record)
        try:
            observation = env.step(request)
            record.update(status='executed', observation=str(observation['image']),
                          placement=env.last_placement)
        except Exception as exc:
            record.update(status='execution_failed', error=f'{type(exc).__name__}: {exc}')
            save(prefix+'_execution_failed')
            raise
        try:
            current = planner.scene_updater.update_scene_graph(current, observation['image'], instruction=instruction)
            localization = locate_reasoning_node(current, graph, candidate_ids={edge.from_state, edge.to_state})
            record['node_localization'] = {
                'selected_node_id': localization.node_id, 'expected_node_id': edge.to_state,
                'conclusive': localization.conclusive,
                'matches_expected': localization.conclusive and localization.node_id == edge.to_state,
                'score': localization.score, 'margin': localization.margin,
                'evidence_object_ids': localization.evidence_object_ids,
                'source': 'camera_rgb_scene_graph'}
            current.to_json(env.output_dir/f'scene_graph_{index+1:03d}.json')
        except Exception as exc:
            record['visual_observation_error'] = f'{type(exc).__name__}: {exc}'
        save('running_'+prefix+'_validation')
    evaluator = TargetGraphEvaluator(initial, target)
    final_observation_available = 'visual_observation_error' not in records[-1]
    report = None
    if final_observation_available:
        evaluator.prepare(current, instruction)
        report = evaluator.evaluate(current, instruction)
    all_nodes_verified = all(r.get('node_localization', {}).get('matches_expected', False) for r in records)
    observed_goal = bool(report and report.terminal)
    status = (prefix+'_visually_verified' if all_nodes_verified and observed_goal
              else prefix+'_video_complete_needs_visual_review')
    payload = save(status)
    payload.update(observed_goal=observed_goal, all_nodes_visually_verified=all_nodes_verified,
                   final_visual_observation_available=final_observation_available,
                   final_visual_goal_report=report.to_dict() if report else None)
    trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return LoopResult(status, len(records), observed_goal, True, trace_path)
