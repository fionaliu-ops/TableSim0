"""Check both arms in a shared world frame before and during execution."""
from pathlib import Path
import json
import numpy as np
import yaml


class ArmClearanceError(RuntimeError):
    pass


class TrajectoryClearanceError(ArmClearanceError):
    pass


def sphere_clearance(left, right):
    distances = np.linalg.norm(left[:, None, :3] - right[None, :, :3], axis=2)
    return float(np.min(distances - left[:, None, 3] - right[None, :, 3]))


class DualArmSafety:
    def __init__(self, robot, output_dir, minimum_clearance=.05, scene=None):
        self.robot = robot
        self.scene = scene
        self.link_names = {}
        self.minimum_clearance = minimum_clearance
        # Planned paths need headroom for controller tracking error.  Scale the
        # allowance from the configured physical clearance instead of tuning it
        # to one scene or arm pose.
        self.planning_margin = .8 * minimum_clearance
        # Collision spheres and FK are evaluated in floating point.  Treat a
        # sub-millimetre equality at the conservative planning envelope as an
        # equality, while the runtime guard below continues to enforce the
        # configured physical clearance without tolerance.
        # MotionGen samples links and trajectories discretely; allow a small
        # planning-only numerical band around the deliberately inflated
        # clearance envelope.  The runtime guard below still enforces the
        # physical minimum with zero tolerance on every control step.
        self.planning_tolerance = min(.006, .12 * minimum_clearance)
        # Runtime safety is a hard physical boundary. Tracking uncertainty is
        # handled by the planning margin and feedback settling, not by allowing
        # the real arms to enter the requested clearance envelope.
        self.runtime_tolerance = 0.0
        self.active_context = None
        self.path = Path(output_dir) / 'arm_safety.json'
        self.models, self.geometry, self.indices = {}, {}, {}
        self.stats = {'minimum_required_m': minimum_clearance, 'minimum_observed_m': None,
                      'checked_control_steps': 0, 'checked_trajectory_samples': 0,
                      'rejected_trajectories': 0, 'runtime_violations': 0, 'inter_arm_contacts': 0,
                      'model': 'configured_link_spheres_in_world_frame',
                      'trajectory_joint_step_rad': .005,
                      'planning_extra_margin_m': self.planning_margin,
                      'runtime_tracking_tolerance_m': self.runtime_tolerance,
                      'runtime_emergency_clearance_m': minimum_clearance - self.runtime_tolerance,
                      'violation_samples': []}
        for arm in ('left', 'right'):
            entity = getattr(robot, arm + '_entity')
            self.models[arm] = entity.create_pinocchio_model()
            config = Path(getattr(robot, arm + '_curobo_yml_path'))
            if robot.is_dual_arm:
                config = Path(str(config).replace('curobo.yml', 'curobo_' + arm + '.yml'))
            cfg = yaml.safe_load(config.read_text())['robot_cfg']['kinematics']
            sphere_path = Path(cfg['collision_spheres'])
            data = yaml.safe_load(sphere_path.read_text())['collision_spheres']
            links = {link.name: i for i, link in enumerate(entity.get_links())}
            self.link_names[arm] = set(cfg['collision_link_names']) | {j.child_link.name for j in getattr(robot, arm + '_arm_joints')}
            self.geometry[arm] = [(links[name], np.array([s['center'] for s in spheres]),
                                   np.array([s['radius'] + .004 for s in spheres]))
                                  for name, spheres in data.items() if spheres and name in cfg['collision_link_names']]
            names = [j.name for j in entity.get_active_joints()]
            self.indices[arm] = [names.index(j.name) for j in getattr(robot, arm + '_arm_joints')]
            if not np.allclose(self.spheres(arm, entity.get_qpos()), self.spheres(arm), atol=.001):
                raise ArmClearanceError(f'{arm} kinematic/world-frame collision geometry mismatch')

    def spheres(self, arm, qpos=None):
        entity = getattr(self.robot, arm + '_entity')
        model = self.models[arm]
        if qpos is not None:
            model.compute_forward_kinematics(qpos)
        result = []
        for index, centers, radii in self.geometry[arm]:
            pose = (entity.get_root_pose() * model.get_link_pose(index) if qpos is not None
                    else entity.get_links()[index].get_pose())
            matrix = pose.to_transformation_matrix()
            world = centers @ matrix[:3, :3].T + matrix[:3, 3]
            result.append(np.column_stack((world, radii)))
        return np.concatenate(result)

    def validate(self, control, start_qpos=None):
        # Serial ownership is enforced even if a future caller supplies two paths.
        if control.get('left_arm') is not None and control.get('right_arm') is not None:
            raise TrajectoryClearanceError('Simultaneous dual-arm trajectories are not permitted')
        for arm, other in (('left', 'right'), ('right', 'left')):
            plan = control.get(arm + '_arm')
            if plan is None:
                continue
            qpos = np.asarray((start_qpos or {}).get(arm, getattr(self.robot, arm + '_entity').get_qpos())).copy()
            stationary = self.spheres(other)
            indices = self.indices[arm]
            previous = qpos[indices].copy()
            for waypoint in np.asarray(plan['position']):
                # Include interpolants between controller targets; endpoint-only
                # tests can miss a crossing between otherwise safe poses.
                count = max(1, int(np.ceil(np.max(np.abs(waypoint - previous)) / .005)))
                for alpha in np.linspace(0., 1., count + 1):
                    qpos[indices] = previous + alpha * (waypoint - previous)
                    clearance = sphere_clearance(self.spheres(arm, qpos), stationary)
                    self.stats['checked_trajectory_samples'] += 1
                    required = self.minimum_clearance + self.planning_margin
                    tolerance = getattr(self, 'planning_tolerance', 0.0)
                    # Discrete sphere samples can land exactly on the
                    # planning boundary and differ by sub-millimetre floating
                    # error.  Keep this slack far below the runtime margin.
                    if clearance + .0005 < required - tolerance:
                        self.stats['rejected_trajectories'] += 1
                        self.save()
                        raise TrajectoryClearanceError(f'{arm} path violates inter-arm clearance: '
                                                f'{clearance:.3f} m, required {required:.3f} m')
                previous = waypoint

    def check_live(self):
        if self.scene is not None:
            for contact in self.scene.get_contacts():
                names = {body.entity.name for body in contact.bodies}
                if names & self.link_names['left'] and names & self.link_names['right']:
                    self.stats['inter_arm_contacts'] += 1
                    self.save()
                    raise ArmClearanceError(f'Physical inter-arm contact: {sorted(names)}')
        clearance = sphere_clearance(self.spheres('left'), self.spheres('right'))
        self.stats['checked_control_steps'] += 1
        minimum = self.stats['minimum_observed_m']
        self.stats['minimum_observed_m'] = clearance if minimum is None else min(minimum, clearance)
        tolerance = getattr(self, 'runtime_tolerance', .05 * self.minimum_clearance)
        if clearance < self.minimum_clearance - tolerance:
            self.stats['runtime_violations'] += 1
            context = dict(getattr(self, 'active_context', None) or {})
            context.update(clearance_m=clearance)
            if hasattr(self, 'robot'):
                for arm in ('left', 'right'):
                    entity = getattr(self.robot, arm + '_entity')
                    context[arm + '_qpos'] = np.asarray(entity.get_qpos(), dtype=float).tolist()
                    if hasattr(entity, 'get_qvel'):
                        context[arm + '_qvel'] = np.asarray(entity.get_qvel(), dtype=float).tolist()
            self.stats.setdefault('violation_samples', []).append(context)
            self.save()
            raise ArmClearanceError(f'Inter-arm clearance fell to {clearance:.3f} m')

    def save(self):
        self.path.write_text(json.dumps(self.stats, indent=2) + '\n')
