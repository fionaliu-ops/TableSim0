"""Visible robot grasp/follow/release, with compliant symbolic attachment.

Joint drives move the real simulated robot. Object translation always comes
from the measured fingertip position after a closed grasp, never a goal path.
The ideal grasp keeps objects upright; yaw follows the actual wrist yaw.
"""
from __future__ import annotations

import json
import math
from copy import deepcopy

import numpy as np
from scipy.optimize import least_squares
import transforms3d as t3d

from script.object_motion_validation import ObjectMotionExecutor
from script.dual_arm_safety import sphere_clearance


class GraspBlockedError(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__('Grasp blocked by '+', '.join(details['blocking_object_ids']))


class SymbolicGraspExecutor:
    pad_offset = .147

    def __init__(self, env, *, visual_only=False):
        self.env, self.task = env, env.task
        self.grounding = None if visual_only else ObjectMotionExecutor(env)
        self.audit = env.output_dir/'symbolic_grasp.jsonl'
        self.attachment = None
        self.yaw_semantics = "world_absolute_degrees"
        self.initial_yaws = {} if visual_only else {name:env._object_yaw(name) for name in self.task.objects}
        self.models = {}
        self.home = {}
        self.arm_bases = {}
        for arm in ('left', 'right'):
            robot = self.task.robot
            entity = getattr(robot, arm+'_entity')
            self.arm_bases[arm] = np.asarray(getattr(robot, arm+'_arm_joints')[0].global_pose.p).copy()
            joint = getattr(robot, arm+'_ee')
            links = entity.get_links()
            link_index = next(i for i, link in enumerate(links) if link.name == joint.child_link.name)
            names = [j.name for j in entity.get_active_joints()]
            indices = [names.index(j.name) for j in getattr(robot, arm+'_arm_joints')]
            limits = np.asarray(entity.get_qlimits())[indices]
            limits = np.clip(limits, -2*math.pi, 2*math.pi)
            correction = (np.linalg.inv(joint.child_link.get_pose().to_transformation_matrix())
                          @ joint.global_pose.to_transformation_matrix())
            self.models[arm] = (entity, entity.create_pinocchio_model(), link_index,
                                indices, limits, correction)
            self.home[arm] = np.asarray(entity.get_qpos()).copy()
            predicted = self.fk(arm, entity.get_qpos())[0]
            if np.linalg.norm(predicted-self.tip(arm)[0]) > 1e-5:
                raise RuntimeError('Fingertip FK calibration disagrees with measured robot pose')

    def log(self, event, **data):
        with self.audit.open('a') as stream:
            stream.write(json.dumps({'step': self.env.step_index, 'event': event, **data})+'\n')

    def install(self):
        original = self.task._sync_attached_actor
        def sync():
            self.sync_attachment()
            original()  # Includes dense video capture, after object following.
        self.task._sync_attached_actor = sync
        # A symbolic grasp is not a friction experiment. Keep prop contact
        # from fighting the robot drives; robot/table and arm/arm remain active.
        props = list(self.task.objects.values())
        if getattr(self.task, 'monitor', None) is not None:
            props.append(self.task.monitor)
        for actor in props:
            for component in getattr(actor, 'actor', actor).get_components():
                if hasattr(component, 'set_kinematic'):
                    component.set_kinematic(True)
                for shape in getattr(component, 'get_collision_shapes', lambda: [])():
                    shape.set_collision_groups([0, 0, 0, 0])

    def tip(self, arm):
        ee = np.asarray(self.task._current_ee_pose(arm), float)
        rotation = t3d.quaternions.quat2mat(ee[3:])
        return ee[:3]+rotation[:, 0]*self.pad_offset, rotation

    def fk(self, arm, qpos):
        entity, model, index, _, _, correction = self.models[arm]
        model.compute_forward_kinematics(np.asarray(qpos))
        world = (entity.get_root_pose().to_transformation_matrix()
                 @ model.get_link_pose(index).to_transformation_matrix() @ correction)
        robot = self.task.robot
        rotation = (world[:3, :3] @ getattr(robot, arm+'_global_trans_matrix')
                    @ getattr(robot, arm+'_delta_matrix'))
        offset = getattr(robot, arm+'_gripper_bias')-.12+self.pad_offset
        return world[:3, 3]+rotation[:, 0]*offset, rotation

    def solve(self, arm, xyz, start, closing_yaw=None, tool_axis=None):
        _, _, _, indices, bounds, _ = self.models[arm]
        seed = np.asarray(start, float)[indices]
        base = np.asarray(start, float).copy()
        desired_closing = (None if closing_yaw is None else
                           np.array([math.cos(closing_yaw), math.sin(closing_yaw), 0.]))
        safety = self.env.arm_safety
        stationary = safety.spheres("right" if arm == "left" else "left")
        clearance_target = safety.minimum_clearance+safety.planning_margin+.02
        def residual(q):
            base[indices] = q
            point, rotation = self.fk(arm, base)
            components = [12*(point-xyz), .22*(rotation[:, 0]-(np.array([0, 0, -1]) if tool_axis is None else tool_axis)), .001*(q-seed)]
            clearance = sphere_clearance(safety.spheres(arm, base), stationary)
            components.append(np.array([8*max(0., clearance_target-clearance)]))
            if desired_closing is not None:
                components.append(.12*(rotation[:, 1]-desired_closing))
            return np.concatenate(components)
        def jacobian(q):
            columns = []
            for index in range(len(q)):
                delta = np.zeros_like(q); delta[index] = .001
                columns.append((residual(q+delta)-residual(q-delta))/.002)
            return np.column_stack(columns)
        best = None
        for guess in (seed, self.home[arm][indices]):
            solution = least_squares(residual, np.clip(guess, bounds[:, 0]+1e-6, bounds[:, 1]-1e-6),
                                     bounds=(bounds[:, 0], bounds[:, 1]), jac=jacobian,
                                     max_nfev=180, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            candidate = np.asarray(start, float).copy(); candidate[indices] = solution.x
            point, rotation = self.fk(arm, candidate)
            error = float(np.linalg.norm(point-xyz))
            score = error+.005*float(np.linalg.norm(solution.x-seed))
            if best is None or score < best[0]:
                best = (score, error, candidate)
            if error < .003:
                break
        if best[1] > .012:
            raise self.env.motion_error_type(f'Numerical fingertip IK unreachable: {best[1]:.4f} m')
        return best[2]

    def drive(self, arm, qtargets, label):
        entity, _, _, indices, _, _ = self.models[arm]
        current = np.asarray(entity.get_qpos())[indices]
        samples = []
        for qpos in qtargets:
            target = np.asarray(qpos)[indices]
            count = max(1, math.ceil(np.max(np.abs(target-current))/.002))
            samples.extend(current+alpha*(target-current) for alpha in np.linspace(1/count, 1, count))
            current = target
        positions = np.vstack((samples, np.repeat(current[None], 180, axis=0)))
        control = {'left_arm': None, 'right_arm': None, 'left_gripper': None, 'right_gripper': None}
        control[arm+'_arm'] = {'position':positions, 'velocity':np.zeros_like(positions)}
        self.task.take_dense_action(control, save_freq=None)
        if self.attachment is not None and self.attachment['arm'] == arm:
            self.attachment.setdefault('return_path', []).extend(np.asarray(q).copy() for q in qtargets)
        self.log('joint_drive_completed', arm=arm, phase=label,
                 joint_tracking_error_rad=float(np.max(np.abs(np.asarray(entity.get_qpos())[indices]-current))))
        self.env._capture_motion_frame(label)

    def move(self, arm, xyz, label, closing_yaw=None, tool_axis=None):
        start = self.tip(arm)[0]
        entity = self.models[arm][0]
        qpos = np.asarray(entity.get_qpos()).copy()
        targets = []
        count = max(1, math.ceil(np.linalg.norm(np.asarray(xyz)-start)/.025))
        for alpha in np.linspace(1/count, 1, count):
            qpos = self.solve(arm, start+alpha*(np.asarray(xyz)-start), qpos, closing_yaw, tool_axis)
            targets.append(qpos)
        self.drive(arm, targets, label)
        error = float(np.linalg.norm(self.tip(arm)[0]-xyz))
        if error > .015:
            correction = self.solve(arm, np.asarray(xyz), entity.get_qpos(), closing_yaw, tool_axis)
            self.drive(arm, [correction], label+'_settle')
            error = float(np.linalg.norm(self.tip(arm)[0]-xyz))
        self.log('motion_reached' if error <= .02 else 'motion_failed', arm=arm, phase=label,
                 tip_error_m=error)
        if error > .02:
            raise self.env.motion_error_type(f'Measured fingertip failed to reach {label}: {error:.4f} m')

    def sync_attachment(self):
        if self.attachment is None:
            return
        import sapien
        attached = self.attachment
        point, rotation = self.tip(attached['arm'])
        yaw = math.atan2(rotation[1, 1], rotation[0, 1])
        q = self.task._yaw_quat(math.degrees(yaw-attached['wrist_yaw']), base_quat=attached['orientation'])
        actor = self.task.objects[attached['name']]
        getattr(actor, 'actor', actor).set_pose(sapien.Pose(point+attached['offset'], q))
        self.env._clear_attachment_velocity(attached['name'])

    def recover_held(self):
        """Reverse the executed carry with the gripper, then release at pickup."""
        if self.attachment is None:
            return
        arm, name = self.attachment['arm'], self.attachment['name']
        route = list(reversed(self.attachment['return_path']))
        self.drive(arm, route, 'failed_carry_return_to_pickup')
        self.env._gripper(arm, False)
        self.attachment = None
        self.env.held = None
        self.log('recovery_released_after_open', object=name, arm=arm)
        retreat = self.tip(arm)[0]; retreat[2] += .07
        self.move(arm, retreat, 'recovery_retreat')
        self.drive(arm, [self.home[arm]], 'recovery_home')

    def check_grasp_access(self, name):
        """Execution-only geometry check; feedback exposes IDs, never world poses."""
        task = self.task
        source = task.objects[name].get_pose().p
        source_top = source[2]+task.object_geometry[name]['height']
        blockers = []
        for key, actor in task.objects.items():
            if key == name:
                continue
            position = actor.get_pose().p
            geometry = task.object_geometry[key]
            footprint = geometry.get('footprint')
            if footprint is None or position[2]+geometry['height'] <= source_top+.003:
                continue
            angle = self.env._object_yaw(key)
            delta = np.asarray(source[:2])-position[:2]
            local = np.array([[math.cos(angle), math.sin(angle)],
                              [-math.sin(angle), math.cos(angle)]]) @ delta
            if np.all(np.abs(local) < np.asarray(footprint)/2+.008):
                blockers.append(key)
        if blockers:
            inverse = {value:key for key,value in self.env.object_map.items()}
            raise GraspBlockedError({'failure_category':'grasp_occluded',
                'source_object_id':inverse.get(name, name),
                'blocking_object_ids':[inverse.get(key, key) for key in blockers],
                'phase':'before_grasp', 'object_moved':False,
                'suggested_repair':'Observe blockers and plan removal before retrying the grasp'})

    def place(self, action, name):
        env, task = self.env, self.task
        if self.attachment is not None or env.held is not None:
            raise env.motion_error_type('New grasp requested while an object remains held')
        self.check_grasp_access(name)
        actor = task.objects[name]
        source = deepcopy(actor.get_pose())
        params = action.get('parameters', {})
        placement = params.get('pose') or params.get('relative_pose') or {}
        if not placement:
            placement = params.get('placement_target', {}).get('parameters', {}).get('pose', {})
        support_id = action.get('target')
        support = env._name(support_id) if support_id and support_id != 'table' else None
        bottom = (float(task.objects[support].get_pose().p[2]+task.object_geometry[support]['height']+.004)
                  if support else float(task._table_z()+.006))
        height = float(task.object_geometry[name]['height'])
        target_xy = (self.grounding.world_point(placement['xy'], bottom+height/2)[:2]
                     if placement.get('xy') is not None else np.asarray(source.p[:2]))
        if support and placement.get('xy') is None:
            target_xy = np.asarray(task.objects[support].get_pose().p[:2])
        target = np.r_[target_xy, bottom]
        contact = np.asarray(source.p).copy(); contact[2] += height+.003
        # Choose using the complete source/destination workload, not a thin
        # object's exposed overhang (which is irrelevant to ideal grasping).
        candidates = []
        for candidate in ('left', 'right'):
            root = self.arm_bases[candidate]
            candidates.append((np.linalg.norm(contact[:2]-root[:2])+
                               np.linalg.norm(target[:2]-root[:2]), candidate))
        arm = None
        failures = []
        tops = [float(obj.get_pose().p[2]+task.object_geometry[key]['height'])
                for key,obj in task.objects.items() if key != name]
        carry_bottom = max(float(source.p[2]), bottom, max(tops, default=bottom))+.025
        for _, candidate in sorted(candidates):
            qpos = self.models[candidate][0].get_qpos()
            try:
                for waypoint in (contact+[0, 0, .07], contact,
                                 np.r_[contact[:2], carry_bottom+height+.003],
                                 np.r_[target[:2], carry_bottom+height+.003],
                                 target+[0, 0, height+.003]):
                    qpos = self.solve(candidate, waypoint, qpos)
                arm = candidate
                break
            except env.motion_error_type as exc:
                failures.append(str(exc))
        if arm is None:
            raise env.motion_error_type('No arm can reach the complete grasp/carry route: '+str(failures))
        other = 'left' if arm == 'right' else 'right'
        self.drive(other, [self.home[other]], 'park_other_arm')
        env._gripper(arm, False)
        approach = contact.copy(); approach[2] += .07
        self.move(arm, approach, 'grasp_approach')
        self.move(arm, contact, 'grasp_contact')
        env._gripper(arm, True)
        point, rotation = self.tip(arm)
        if np.linalg.norm(point-contact) > .02:
            raise env.motion_error_type('Cannot attach before gripper reaches the object')
        current = deepcopy(actor.get_pose())
        self.attachment = {'name':name, 'arm':arm, 'offset':np.asarray(current.p)-point,
                           'orientation':np.asarray(current.q),
                           'wrist_yaw':math.atan2(rotation[1, 1], rotation[0, 1]),
                           'return_path':[np.asarray(self.models[arm][0].get_qpos()).copy()]}
        env.held = (name, arm)
        self.log('attached_after_close', object=name, arm=arm, grasp_tip_error_m=float(np.linalg.norm(point-contact)),
                 offset_m=self.attachment['offset'].tolist())
        desired_yaw = placement.get('yaw_degrees', params.get('target_yaw_degrees'))
        closing_yaw = self.attachment['wrist_yaw']
        if desired_yaw is not None:
            desired_world_yaw = math.radians(float(desired_yaw))
            if self.yaw_semantics == 'visual_delta_from_initial_reference_degrees':
                desired_world_yaw = self.initial_yaws[name]-desired_world_yaw
            closing_yaw += desired_world_yaw-env._object_yaw(name)
        release = target-self.attachment['offset']
        tops = [float(obj.get_pose().p[2]+task.object_geometry[key]['height'])
                for key,obj in task.objects.items() if key != name]
        # Keep the payload just above live clutter; excess height can make
        # otherwise reachable workspace points unreachable for short arms.
        carry_bottom = max(float(source.p[2]), bottom, max(tops, default=bottom))+.025
        lift = point.copy(); lift[2] = carry_bottom-self.attachment['offset'][2]
        high = release.copy(); high[2] = lift[2]
        self.move(arm, lift, 'carry_lift')
        self.move(arm, high, 'carry_transfer', closing_yaw)
        self.move(arm, release, 'carry_lower', closing_yaw)
        actual = np.asarray(actor.get_pose().p).copy()
        if np.linalg.norm(actual-target) > .025:
            raise env.motion_error_type('Attached object did not arrive at release position')
        env._gripper(arm, False)
        released = np.asarray(actor.get_pose().p).copy()
        self.attachment = None
        env.held = None
        self.log('released_after_open', object=name, arm=arm, actual_xyz=released.tolist(),
                 position_error_m=float(np.linalg.norm(released-target)))
        retreat = self.tip(arm)[0]; retreat[2] += .08
        self.move(arm, retreat, 'release_retreat')
        self.drive(arm, [self.home[arm]], 'return_empty_arm')
        env.last_placement = {'object':name, 'arm':arm, 'mode':'symbolic_gripper_follow',
                              'requested_xy':target_xy.tolist(), 'actual_xy':released[:2].tolist(),
                              'placement_error':float(np.linalg.norm(released[:2]-target_xy)),
                              'robot_motion_executed':True, 'contact_physics_verified':False}
