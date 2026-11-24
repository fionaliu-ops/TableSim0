"""Local execution of visual planner commands, with audited contact/follow modes.

Simulator state in this module is controller feedback only. No poses or contact
measurements are passed to the high-level model. No actor/joint pose rollback.
"""
from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np


def smooth_joint_path(path, current, max_step=.002, settle_ticks=150):
    """Bound drive increments and settle; never write measured joint positions."""
    previous = np.asarray(current, dtype=float)
    samples = []
    for target in np.asarray(path['position'], dtype=float):
        count = max(1, int(np.ceil(np.max(np.abs(target-previous))/max_step)))
        samples.extend(previous + alpha*(target-previous)
                       for alpha in np.linspace(1/count, 1, count))
        previous = target
    if not samples:
        raise ValueError('Empty robot trajectory')
    positions = np.vstack((samples, np.repeat(previous[None], settle_ticks, axis=0)))
    return dict(path, position=positions, velocity=np.zeros_like(positions))


def supervise_action(directory, index, request, observation, timeout=900):
    """Publish an action for the local agent; accept execution-only choices."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    message = {'index': index, 'request': request, 'observation': str(observation),
               'allowed_decisions': {'execute': True, 'grasp_mode': 'auto/contact/follow',
                                     'arm': 'left/right (optional)'}}
    temporary = directory / f'request_{index:03d}.tmp'
    temporary.write_text(json.dumps(message, ensure_ascii=False, indent=2))
    temporary.replace(directory / f'request_{index:03d}.json')
    decision_path = directory / f'decision_{index:03d}.json'
    deadline = time.monotonic()+timeout
    while not decision_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('Local execution supervisor did not provide a decision')
        time.sleep(.25)
    decision = json.loads(decision_path.read_text())
    if set(decision)-{'execute', 'grasp_mode', 'arm', 'reason'}:
        raise ValueError('Supervisor may choose execution settings, not replace planner targets')
    if decision.get('execute') is not True:
        raise RuntimeError('Action stopped by local execution supervisor')
    if decision.get('grasp_mode', 'auto') not in {'auto', 'contact', 'follow'}:
        raise ValueError('Invalid grasp mode')
    if decision.get('arm') not in {None, 'left', 'right'}:
        raise ValueError('Invalid arm')
    return decision


class HybridExecutor:
    def __init__(self, env):
        self.env, self.task = env, env.task
        self.audit = env.output_dir / 'hybrid_execution.jsonl'

    def log(self, event, **details):
        with self.audit.open('a') as stream:
            stream.write(json.dumps({'step': self.env.step_index, 'event': event,
                                     **details}, ensure_ascii=False)+'\n')

    def install(self):
        original = self.task.take_dense_action
        def smooth(control, save_freq=-1):
            control = dict(control)
            for arm in ('left', 'right'):
                path = control.get(arm+'_arm')
                if path is None:
                    continue
                entity = getattr(self.task.robot, arm+'_entity')
                names = [joint.get_name() for joint in entity.get_active_joints()]
                indices = [names.index(joint.get_name())
                           for joint in getattr(self.task.robot, arm+'_arm_joints')]
                control[arm+'_arm'] = smooth_joint_path(path, entity.get_qpos()[indices])
            return original(control, save_freq)
        self.task.take_dense_action = smooth

    def body(self, name):
        entity = getattr(self.task.objects[name], 'actor', self.task.objects[name])
        return next((c for c in entity.get_components()
                     if hasattr(c, 'set_kinematic')), None)

    def contacts(self, name, arm):
        fingers = {joint[0].child_link.get_name()
                   for joint in getattr(self.task.robot, arm+'_gripper')}
        entity_name = self.task.objects[name].get_name()
        touched = set()
        for contact in self.task.scene.get_contacts():
            names = {body.entity.name for body in contact.bodies}
            if entity_name in names:
                # Ignore contacts with no force (broadphase proximity alone).
                if any(np.linalg.norm(point.impulse) > 1e-7 for point in contact.points):
                    touched.update(names & fingers)
        return len(fingers) == 2 and touched == fingers

    def move(self, arm, pose, label):
        start = np.asarray(self.task._current_ee_pose(arm), dtype=float)
        target = np.asarray(pose, dtype=float)
        if np.dot(start[3:], target[3:]) < 0:
            target[3:] *= -1
        angle = 2*math.acos(np.clip(abs(float(np.dot(start[3:], target[3:]))), 0, 1))
        count = max(1, math.ceil(np.linalg.norm(target[:3]-start[:3])/.05),
                    math.ceil(angle/math.radians(15)))
        for index, alpha in enumerate(np.linspace(1/count, 1, count)):
            waypoint = start+alpha*(target-start)
            waypoint[3:] /= np.linalg.norm(waypoint[3:])
            self._move_waypoint(arm, waypoint, f'{label}_{index+1}_of_{count}')

    def _move_waypoint(self, arm, pose, label):
        # `python -m script.dynamic_sim` defines its class in __main__;
        # importing script.dynamic_sim here creates a different exception type.
        DynamicSimError = self.env.motion_error_type
        for attempt in range(3):
            try:
                self.env._graph_validation_motion(arm, list(pose), label)
                actual = np.asarray(self.task._current_ee_pose(arm))
                dot = abs(float(np.dot(actual[3:]/np.linalg.norm(actual[3:]),
                                       np.asarray(pose[3:])/np.linalg.norm(pose[3:]))))
                angular = math.degrees(2*math.acos(np.clip(dot, 0, 1)))
                if angular > 8:
                    raise DynamicSimError(f'Wrist orientation incomplete at {label}: {angular:.2f} degrees')
                self.log('motion_reached', label=label, attempt=attempt+1,
                         position_error_m=float(np.linalg.norm(actual[:3]-pose[:3])),
                         orientation_error_degrees=angular)
                return
            except DynamicSimError as exc:
                paths = getattr(self.task, arm+'_joint_path', [])
                status = (str(paths[-1].get('failure_status', paths[-1].get('status')))
                          if paths else None)
                self.log('motion_incomplete', label=label, attempt=attempt+1, error=str(exc),
                         planner_success=bool(self.task.plan_success), planner_status=status)
                if attempt == 2:
                    raise

    def grasp_pose(self, name, arm, physical):
        import transforms3d as t3d
        geometry = self.task.object_geometry[name]
        actor = self.task.objects[name]
        p = np.asarray(actor.get_pose().p)
        top = float(p[2]+geometry['height'])
        pose = np.asarray(self.task._top_down_pose(p[:2], top, arm, clearance=.04))
        rotation = t3d.quaternions.quat2mat(pose[3:])
        if physical:
            # Actual finger pads, not the nominal tool origin, straddle the
            # upper body. These are gripper dimensions, never scene buffers.
            contact = np.r_[p[:2], top-min(.018, geometry['height']*.45)]
            pose[:3] = contact-rotation[:, 0]*.147
        return pose, rotation

    def eligible(self, name, arm, rotation):
        geometry = self.task.object_geometry[name]
        footprint = geometry.get('footprint')
        if self.body(name) is None or footprint is None:
            return False, 'no_dynamic_body_or_geometry'
        yaw = self.env._object_yaw(name)
        planar = np.array([[math.cos(yaw), -math.sin(yaw)],
                           [math.sin(yaw), math.cos(yaw)]])
        closing = planar.T @ rotation[:2, 1]
        width = float(np.abs(closing) @ np.asarray(footprint))
        aperture = 2*max(getattr(self.task.robot, arm+'_gripper_scale'))
        if width+.006 > aperture:
            return False, f'projected_width_{width:.4f}_exceeds_aperture_{aperture:.4f}'
        if geometry['height'] < .012:
            return False, 'insufficient_top_pinch_height'
        return True, 'dynamic_body_fits_top_pinch'

    def pick(self, name, arm, policy):
        env, task = self.env, self.task
        actor = task.objects[name]
        grasp, rotation = self.grasp_pose(name, arm, physical=True)
        eligible, reason = self.eligible(name, arm, rotation)
        self.log('grasp_assessment', object=name, arm=arm, eligible=eligible, reason=reason)
        physical = eligible and policy != 'follow'
        if policy == 'contact' and not eligible:
            raise RuntimeError(f'Contact-only grasp unavailable: {reason}')
        if not physical:
            grasp, _ = self.grasp_pose(name, arm, physical=False)
        approach = grasp.copy(); approach[2] += .08
        env._gripper(arm, False)
        self.move(arm, approach, 'hybrid_approach')
        if physical:
            self.body(name).set_kinematic(False)
        self.move(arm, grasp, 'hybrid_grasp')
        env._gripper(arm, True)
        if physical:
            before = np.asarray(actor.get_pose().p).copy()
            bilateral = self.contacts(name, arm)
            if bilateral:
                lift = np.asarray(task._current_ee_pose(arm)); lift[2] += .025
                self.move(arm, lift, 'contact_test_lift')
                rise = float(actor.get_pose().p[2]-before[2])
                bilateral = self.contacts(name, arm) and rise >= .015
            else:
                rise = 0.
            self.log('contact_grasp_test', object=name, bilateral_contact=bilateral,
                     object_lift_m=rise, attachment_used=False)
            if bilateral:
                env.held = (name, arm)
                return 'physical_contact'
            if policy == 'contact':
                raise RuntimeError('Contact grasp failed; contact-only policy forbids attachment')
            # Reacquire at the object's CURRENT pose; never attach remotely or
            # restore its old pose after a failed contact test.
            self.body(name).set_kinematic(True)
            env._clear_attachment_velocity(name)
            env._gripper(arm, False)
            grasp, _ = self.grasp_pose(name, arm, physical=False)
            approach = grasp.copy(); approach[2] += .06
            self.move(arm, approach, 'fallback_approach')
            self.move(arm, grasp, 'fallback_grasp')
            env._gripper(arm, True)
        if self.body(name) is not None:
            self.body(name).set_kinematic(True)
        task._attach_actor(name, arm)
        env.held = (name, arm)
        self.log('follow_attached', object=name, after_closed_grasp=True,
                 reason=reason if not physical else 'contact_test_failed')
        return 'assisted_follow'

    def place(self, action, name):
        import sapien
        from script.dynamic_sim import DynamicSimError
        env, task = self.env, self.task
        if env.held is not None:
            raise DynamicSimError('Cannot start a new action with an unreleased object')
        policy = getattr(env, 'execution_policy', {})
        arm = policy.get('arm') or env._arm(name)
        mode = self.pick(name, arm, policy.get('grasp_mode', 'auto'))
        actor = task.objects[name]
        observed = deepcopy(actor.get_pose())
        relative = np.linalg.inv(task._pose_matrix(task._current_ee_pose(arm))) @ observed.to_transformation_matrix()
        parameters = action.get('parameters', {})
        placement = parameters.get('pose') or parameters.get('relative_pose') or {}
        xy, bottom = env._destination(action, name)
        orientation = np.asarray(observed.q)
        yaw = placement.get('yaw_degrees', parameters.get('target_yaw_degrees'))
        if yaw is not None:
            matrix = observed.to_transformation_matrix()
            old_yaw = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
            orientation = task._yaw_quat(float(yaw)-old_yaw, base_quat=orientation)
        target = sapien.Pose([*map(float, xy), float(bottom)], orientation)
        release_pose = task._matrix_pose(target.to_transformation_matrix() @ np.linalg.inv(relative))
        release = np.r_[release_pose.p, release_pose.q]
        lift = np.asarray(task._current_ee_pose(arm)); lift[2] += .10
        high = release.copy(); high[2] = max(high[2]+.10, lift[2])
        for label, pose in [('hybrid_lift', lift), ('hybrid_transfer', high), ('hybrid_lower', release)]:
            self.move(arm, pose, label)
            if mode == 'physical_contact' and not self.contacts(name, arm):
                raise DynamicSimError(f'Physical grasp lost during {label}')
        actual = deepcopy(actor.get_pose())
        error = float(np.linalg.norm(actual.p-target.p))
        if error > .02:
            raise DynamicSimError(f'Object did not reach release pose: {error:.4f} m')
        env._gripper(arm, False)
        if mode == 'assisted_follow':
            # Clearing attachment is sufficient: no set_pose even at release.
            task._attached_actor = task._attached_arm = task._ee_to_actor = None
            env._clear_attachment_velocity(name)
        env.held = None
        retreat = np.asarray(task._current_ee_pose(arm)); retreat[2] += .10
        self.move(arm, retreat, 'hybrid_retreat')
        env.last_placement = {'object': name, 'mode': mode, 'arm': arm,
                              'requested_xy': list(map(float, xy)),
                              'actual_xy': list(map(float, actor.get_pose().p[:2])),
                              'placement_error': float(np.linalg.norm(actor.get_pose().p[:2]-xy))}
        self.log('released', **env.last_placement)
