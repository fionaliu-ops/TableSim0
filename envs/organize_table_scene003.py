"""Scene 003 organization with the large book removed at the user's request.

Uses contact-driven manipulation, rejecting grasps without two-finger contact
and measured lift. The remaining stapler proxy uses a physical hinge.
Reward groups follow RoboDojo's organize_table design, without
importing its incompatible TaskEnv/RewardManager runtime.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien
import transforms3d as t3d

from .scene_003 import scene_003
from .utils import create_actor
from .utils.scene003_physical import PhysicalManipulation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = PROJECT_ROOT / 'planner_outputs/scene_003_kimi_api_unique_goal_20260919/action_flow.json'
OBJECT_IDS = {
    'book_sam3': 'book', 'book_sam3_2': 'folder', 'cup_sam3': 'cup',
    'keyboard_sam3': 'keyboard', 'mouse_sam3': 'mouse',
    'phone_sam3': 'phone', 'stapler_sam3': 'stapler', 'monitor_sam3': 'monitor',
}
EXPECTED_CHAIN = (
    ('move', 'cup_sam3', 'table'), ('move', 'book_sam3', 'table'),
    ('close', 'book_sam3_2', None), ('stack', 'book_sam3_2', 'book_sam3'),
    ('move', 'mouse_sam3', 'table'), ('move', 'keyboard_sam3', 'table'),
    ('place', 'cup_sam3', 'book_sam3_2'), ('close', 'stapler_sam3', None),
    ('move', 'phone_sam3', 'table'),
)


def load_action_chain(path=DEFAULT_PLAN):
    """Reject a different or malformed chain instead of silently substituting it."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    payload = json.loads(path.read_text())
    steps = payload['steps']
    chain = tuple((s['action_type'], s['source_object_id'], s['target']) for s in steps)
    if payload.get('scene') != 'scene_003' or not payload.get('unique_goal') or chain != EXPECTED_CHAIN:
        raise ValueError('Expected the scene_003 nine-step chain with one goal')
    for number, step in enumerate(steps, 1):
        if step['step'] != number:
            raise ValueError('Action steps must be consecutive')
        if step['action_type'] != 'close':
            pose = step['parameters'].get('pose') or step['parameters'].get('relative_pose')
            if not pose or pose.get('frame') != 'image_normalized' or pose.get('support') != step['target']:
                raise ValueError(f'Step {number}: invalid placement frame/support')
            xy = np.asarray(pose['xy'], dtype=float)
            if xy.shape != (2,) or not np.all(np.isfinite(xy)) or not np.all((0 <= xy) & (xy <= 1)):
                raise ValueError(f'Step {number}: invalid image position')
    return steps, hashlib.sha256(path.read_bytes()).hexdigest()


def image_region_to_workspace(xy):
    """Visual layout embedding into the existing simulator, NOT camera calibration.

    Image +v points towards the viewer, whereas simulator +y points backwards.
    These scales are simulator workspace choices, not recovered physical sizes.
    For stacking we separately resolve support height and footprint containment.
    """
    u, v = xy
    return np.array([(u - .5) * .95, .10 - (v - .38) * .80])


def without_large_book(steps):
    """Apply the user's removal to the validated source plan, preserving its IDs."""
    result=[]
    for step in deepcopy(steps):
        if step['source_object_id']=='book_sam3_2':
            continue
        if step['target']=='book_sam3_2':
            step['target']='book_sam3'
            pose=step['parameters'].get('pose') or step['parameters']['relative_pose']
            pose['support']='book_sam3'
        result.append(step)
    return result


class _HingedProp:
    """Existing two-panel geometry with a passive hinge and a free root."""
    def __init__(self, scene, name, pose, size, angle, color):
        self.size = np.asarray(size)
        width, depth, height = self.size
        builder = scene.create_articulation_builder()
        base = builder.create_link_builder()
        lid = builder.create_link_builder(base)
        for i, (link, c) in enumerate(zip((base, lid), (color, tuple(min(1., x+.1) for x in color)))):
            link.set_name(f'{name}_{i}')
            shape_pose = sapien.Pose([0, 0, height/4])
            link.add_box_collision(pose=shape_pose, half_size=[width/2, depth/2, height/4],
                                   material=scene.default_physical_material)
            link.add_box_visual(pose=shape_pose, half_size=[width/2, depth/2, height/4],
                                material=sapien.render.RenderMaterial(base_color=[*c, 1]))
        # SAPIEN hinge axis is joint-frame X. Rotate X onto local -Y.
        joint_q = t3d.quaternions.axangle2quat([0, 0, 1], -np.pi/2)
        lid.set_joint_name(f'{name}_hinge')
        lid.set_joint_properties('revolute', [[0, np.pi]],
                                sapien.Pose([-width/2, 0, height/2], joint_q),
                                sapien.Pose([-width/2, 0, 0], joint_q))
        self.actor = builder.build(fix_root_link=False)
        self.actor.set_name(name)
        # Same default mass as the existing Actor/create_box wrapper, per panel.
        for link in self.actor.get_links():
            link.mass = .01
        self.actor.set_root_pose(pose)
        self.actor.set_qpos([float(angle)])
        self.actor.set_qvel([0.])

    @property
    def angle(self):
        return float(self.actor.get_qpos()[0])

    def get_pose(self):
        return self.actor.get_root_pose()

    def get_qpos(self):
        return np.array([self.angle])

    def lid_tip(self, angle):
        width, _, height = self.size
        hinge = sapien.Pose([-width/2, 0, height/2], t3d.quaternions.axangle2quat([0, 1, 0], -angle))
        return (self.get_pose() * hinge * sapien.Pose([width * .85, 0, height/4])).p


class organize_table_scene003(PhysicalManipulation, scene_003):
    """Seven ordered stages, observed-state goal checks and four reward groups."""
    ORDERED_STAGES = tuple(f'{i:02d}_{kind}_{OBJECT_IDS[obj]}'
                           for i, (kind, obj, _) in enumerate(EXPECTED_CHAIN, 1)
                           if obj!='book_sam3_2')
    PROXY_SIZES = {'folder': (.19, .145, .016), 'stapler': (.105, .04, .024)}
    COMPACT_LAYOUT = {'book':[-.12,.06], 'keyboard':[.055,.055],
                      'mouse':[.16,.06], 'phone':[.16,-.05], 'stapler':[-.115,-.09]}

    def setup_demo(self, **kwargs):
        self.action_chain, self.plan_sha256 = load_action_chain(kwargs.pop('action_flow', DEFAULT_PLAN))
        self.action_chain = without_large_book(self.action_chain)
        self.OBJECT_SPECS = deepcopy(scene_003.OBJECT_SPECS)
        self.OBJECT_SPECS.pop('folder')
        self.stage_log = []
        self.physical_grasp_log = {}
        self.grasp_choices = {}
        self.failure_reason = None
        self.last_motion_error = None
        self._score_peak = 0
        self._reward_previous = 0
        super().setup_demo(**kwargs)
        self.arm_home = {arm: np.asarray(self._current_ee_pose(arm)) for arm in ('left', 'right')}

    def save_traj_data(self, idx):
        import pickle
        super().save_traj_data(idx)
        path = Path(self.save_dir)/'_traj_data'/f'episode{idx}.pkl'
        with path.open('rb') as file:
            data = pickle.load(file)
        data['physical_grasp_choices'] = self.grasp_choices
        with path.open('wb') as file:
            pickle.dump(data,file)

    def load_tran_data(self, idx):
        data = super().load_tran_data(idx)
        self.grasp_choices = deepcopy(data.get('physical_grasp_choices',{}))
        return data

    def load_actors(self):
        # Preserve geometry and layout; enable physical motion (user approved).
        self.object_geometry_by_spec = {}
        self.object_geometry = {}
        self.objects = {}
        self.initial_bottom_z = {}
        self.target_bottom_z = {}
        for name, spec in self.OBJECT_SPECS.items():
            spec.setdefault('arm', 'left' if spec['initial_xy'][0] < 0 else 'right')
            spec['target_xy'] = list(spec['initial_xy'])
            spec['target_yaw'] = spec['initial_yaw']
            if name in self.PROXY_SIZES:
                width, depth, height = self.PROXY_SIZES[name]
                geometry = {'height': height, 'bottom_offset': 0., 'footprint': np.array([width, depth])}
            else:
                geometry = self._asset_geometry(spec)
            self.object_geometry[name] = geometry
            self.object_geometry_by_spec[id(spec)] = geometry

        for name, spec in self.OBJECT_SPECS.items():
            bottom = self._table_z() + self.SURFACE_GAP
            if name == 'cup':
                bottom += self.object_geometry['book']['height'] + self.STACK_GAP
            self.initial_bottom_z[name] = self.target_bottom_z[name] = bottom
            if name in self.PROXY_SIZES:
                pose = sapien.Pose([*spec['initial_xy'], bottom], self._yaw_quat(spec['initial_yaw']))
                angle = np.pi if name == 'folder' else np.deg2rad(24)
                color = (.35, .70, .86) if name == 'folder' else (.75, .77, .78)
                actor = _HingedProp(self.scene, f'scene003_{name}', pose, self.PROXY_SIZES[name], angle, color)
            else:
                actor = create_actor(self, pose=self._rigid_pose(spec, spec['initial_xy'], spec['initial_yaw'], bottom),
                                     modelname=spec['source'], model_id=spec['model_id'],
                                     scale_multiplier=spec['scale'], convex=True, is_static=False)
                if actor is None:
                    raise RuntimeError(f'Failed loading {name}')
                actor.set_name(f'scene003_{name}')
            self.objects[name] = actor
        spec = self.MONITOR_SPEC
        self.object_geometry_by_spec[id(spec)] = self._asset_geometry(spec)
        self.monitor = create_actor(self, pose=self._rigid_pose(spec, spec['initial_xy'], spec['initial_yaw'], self._table_z()),
                                    modelname=spec['source'], model_id=spec['model_id'],
                                    scale_multiplier=spec['scale'], convex=True, is_static=True)
        if self.monitor is None:
            raise RuntimeError('Failed loading monitor')
        self.monitor_initial_pose = self.monitor.get_pose()
        self._final_targets = self._compile_targets()
        self.delay(1, save_freq=None)

    def _target_pose(self, name):
        if name in self.PROXY_SIZES:
            spec = self.OBJECT_SPECS[name]
            return sapien.Pose([*spec['target_xy'], self.target_bottom_z[name]], self._yaw_quat(spec['target_yaw']))
        return super()._target_pose(name)

    def _bottom(self, name):
        return float(self.objects[name].get_pose().p[2] + self.object_geometry[name]['bottom_offset'])

    def _compile_targets(self):
        targets = {name: {'xy': list(spec['initial_xy']), 'bottom': self.initial_bottom_z[name],
                          'yaw': spec['initial_yaw'], 'support': 'table'}
                   for name, spec in self.OBJECT_SPECS.items()}
        image_xy = {}
        self.stage_targets = {}
        for step in self.action_chain:
            if step['action_type'] == 'close':
                continue
            name = OBJECT_IDS[step['source_object_id']]
            pose = step['parameters'].get('pose') or step['parameters']['relative_pose']
            support = step['target']
            xy = image_region_to_workspace(pose['xy'])
            bottom = self._table_z() + self.SURFACE_GAP
            if support != 'table':
                support_name = OBJECT_IDS[support]
                base = targets[support_name]
                # Object centroids at different heights cannot be projected as
                # coplanar points. Align the books; preserve the cup's rightward
                # offset, clamped to the upper book's visible support footprint.
                offset = np.zeros(2)
                if name == 'cup':
                    offset = (np.asarray(pose['xy']) - image_xy[support_name]) * [.95, -.80]
                    room = (self.object_geometry[support_name]['footprint'] - self.object_geometry[name]['footprint'])/2 - .005
                    if np.any(room < 0):
                        raise ValueError('Cup does not fit on the proxy book support')
                    offset = np.clip(offset, -room, room)
                xy = np.asarray(base['xy']) + offset
                bottom = base['bottom'] + self.object_geometry[support_name]['height'] + self.STACK_GAP
            target = {'xy': xy.tolist(), 'bottom': bottom, 'yaw': 0., 'support': support}
            targets[name] = target
            image_xy[name] = np.asarray(pose['xy'])
            self.stage_targets[step['step']] = deepcopy(target)
        # Keep the temporary cup clear of the future lower book's footprint.
        # This grounding adjustment does not alter the recorded image target.
        buffer = self.stage_targets[1]
        book = targets['book']
        separation = (self.object_geometry['book']['footprint'] + self.object_geometry['cup']['footprint'])/2 + .02
        if np.all(np.abs(np.asarray(buffer['xy']) - book['xy']) < separation):
            buffer['xy'][0] = float(book['xy'][0] + separation[0])
        # The old buffer clears the book footprint but obstructs the wrist's
        # front approach. Replan this TEMPORARY action only; keep final goals,
        # object dimensions, materials and all initial transforms unchanged.
        source = self.OBJECT_SPECS['book']['initial_xy']
        if abs(buffer['xy'][0]-source[0]) < .09 and buffer['xy'][1] < source[1]:
            self.buffer_grounding = {'original_xy':list(buffer['xy']),
                                     'reason':'clear the horizontal book-grasp approach corridor'}
            buffer['xy'] = [-.18,-.25]
            self.buffer_grounding['selected_xy'] = list(buffer['xy'])
        for name,xy in self.COMPACT_LAYOUT.items():
            targets[name]['xy']=list(xy)
        targets['cup']['xy']=[-.12,.045]
        for step in self.action_chain:
            if step['step']!=1 and step['action_type']!='close':
                self.stage_targets[step['step']]=deepcopy(targets[OBJECT_IDS[step['source_object_id']]])
        return targets

    def _set_target(self, name, target):
        self.OBJECT_SPECS[name]['target_xy'] = list(target['xy'])
        self.OBJECT_SPECS[name]['target_yaw'] = target['yaw']
        self.target_bottom_z[name] = target['bottom']

    def _close_prop(self, name, stage):
        if name=='stapler':
            self._set_target(name,self._final_targets[name])
            self.initial_bottom_z[name]=self._bottom(name)
            if not self._pick_and_place(name,stage+'/position'):
                return False
            self.completed_stages.pop()
        if name == 'folder' and self.completed_stages != list(self.ORDERED_STAGES[:2]):
            raise RuntimeError('Clear cup and small book before closing the large book')
        return self._physical_close_prop(name, stage)

    def play_once(self):
        self.completed_stages, self.stage_log = [], []
        # The collector treats info['info'] as template placeholders only.
        # This task uses literal templates, so keep that mapping empty.
        self.info['info'] = {}
        self.info['instruction'] = self.get_instruction()
        self.info['execution'] = {'mode': 'contact_physics', 'physical_grasp_verified': False,
                                  'plan_sha256': self.plan_sha256, 'stages': self.stage_log,
                                  'physical_grasps': self.physical_grasp_log,
                                  'grasp_choices': self.grasp_choices,
                                  'buffer_grounding': getattr(self,'buffer_grounding',None),
                                  'grounded_stage_targets': self.stage_targets}
        self.info['execution']['plan_adaptation']={'removed_object':'book_sam3_2',
            'cup_final_support':'book_sam3','effective_steps':self.action_chain,
            'layout':'compact_in_front_of_monitor','final_targets':self._final_targets}
        for step, stage in zip(self.action_chain, self.ORDERED_STAGES):
            name = OBJECT_IDS[step['source_object_id']]
            if step['action_type'] == 'close':
                success = self._close_prop(name, stage)
            else:
                self._set_target(name, self.stage_targets[step['step']])
                # Cup is picked twice: always use its CURRENT height.
                self.initial_bottom_z[name] = self._bottom(name)
                try:
                    success = self._pick_and_place(name, stage)
                finally:
                    self._attached_actor = self._attached_arm = self._ee_to_actor = None
            self.stage_log.append({'step': step['step'], 'stage': stage, 'success': bool(success)})
            if not success:
                self.plan_success = False
                self.info['execution']['failure_reason'] = self.failure_reason
                return self.info
            if not self._run_move(stage+'/home', self.back_to_origin(self._arm_tag(self.OBJECT_SPECS[name]['arm']))):
                self.stage_log[-1]['return_home_failed'] = True
                return self.info
        self.delay(1)
        self.info['layout_metrics']=self.layout_metrics()
        self.info['final_object_poses']={name:[*map(float,obj.get_pose().p),*map(float,obj.get_pose().q)]
                                         for name,obj in self.objects.items()}
        self.info['goal_checks'] = self.goal_checks()
        self.info['execution']['physical_grasp_verified'] = bool(self.check_success())
        return self.info

    def _at_final(self, name):
        target = self._final_targets[name]
        base = None if name in self.PROXY_SIZES else self.RIGID_Y_UP_TO_Z_UP
        expected_q = self._yaw_quat(target['yaw'], base)
        alignment = abs(float(np.dot(self.objects[name].get_pose().q, expected_q)))
        bottom = target['bottom']
        if target['support'] != 'table':
            support = OBJECT_IDS[target['support']]
            bottom = self._bottom(support)+self.object_geometry[support]['height']
        return bool(np.linalg.norm(self.objects[name].get_pose().p[:2] - target['xy']) < .025
                    and abs(self._bottom(name) - bottom) < .012
                    and alignment > np.cos(np.deg2rad(12)/2))

    def _on(self, upper, lower):
        delta = np.abs(self.objects[upper].get_pose().p[:2] - self.objects[lower].get_pose().p[:2])
        # Require the upper center to lie inside the lower footprint, plus a
        # small vertical gap. Static props cannot establish physical contacts.
        inside = np.all(delta < self.object_geometry[lower]['footprint']/2)
        gap = self._bottom(upper) - self._bottom(lower) - self.object_geometry[lower]['height']
        return bool(inside and -.002 <= gap <= self.STACK_GAP + .006)

    def layout_metrics(self):
        """Measure occupied footprints; the cup shares its book's support."""
        low,high,distances=[],[],[]
        monitor=self.monitor.get_pose().p[:2]
        for name in self.COMPACT_LAYOUT:
            pose=self.objects[name].get_pose()
            matrix=pose.to_transformation_matrix()
            yaw=np.arctan2(matrix[1,0],matrix[0,0])
            rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
            half=np.abs(rotation)@(self.object_geometry[name]['footprint']/2)
            low.append(pose.p[:2]-half)
            high.append(pose.p[:2]+half)
            distances.append(np.linalg.norm(pose.p[:2]-monitor))
        lower=np.min(low,axis=0); upper=np.max(high,axis=0)
        table_center=np.asarray(getattr(self,'table_xy_bias',[0.,0.]))[:2]
        margin=np.min(np.r_[lower-(table_center-[.6,.35]),(table_center+[.6,.35])-upper])
        return {'minimum_table_edge_margin':float(margin),
                'occupied_area':float(np.prod(upper-lower)),
                'mean_distance_to_monitor':float(np.mean(distances)),
                'bounds':[lower.tolist(),upper.tolist()]}

    def goal_checks(self):
        mouse, keyboard = self.objects['mouse'].get_pose().p, self.objects['keyboard'].get_pose().p
        monitor_fixed = np.linalg.norm(self.monitor.get_pose().p - self.monitor_initial_pose.p) < .002
        monitor_fixed = monitor_fixed and abs(np.dot(self.monitor.get_pose().q, self.monitor_initial_pose.q)) > .999
        arms_home = all(np.linalg.norm(np.asarray(self._current_ee_pose(arm))[:3] - pose[:3]) < .03
                        for arm, pose in self.arm_home.items())
        checks = {
            'books': self._at_final('book'),
            'cup': self._at_final('cup') and self._on('cup', 'book'),
            'electronics': self._at_final('mouse') and self._at_final('keyboard')
                           and mouse[0] > keyboard[0] and keyboard[1] < self.monitor.get_pose().p[1],
            'stationery_phone': self._at_final('stapler') and self._at_final('phone')
                                and abs(self.objects['stapler'].angle) < .03,
            'monitor_fixed': bool(monitor_fixed),
            'arms_home': bool(arms_home),
            'grippers_open': bool(self.is_left_gripper_open() and self.is_right_gripper_open()),
        }
        layout=self.layout_metrics()
        checks['compact_layout']=(layout['minimum_table_edge_margin']>=.17
                                  and layout['occupied_area']<=.14
                                  and layout['mean_distance_to_monitor']<=.26)
        return {key: bool(value) for key, value in checks.items()}

    def get_score(self):
        checks = self.goal_checks()
        score = int(25 * sum(checks[name] for name in ('books', 'cup', 'electronics', 'stationery_phone')))
        return score if checks['grippers_open'] and checks['monitor_fixed'] and checks['arms_home'] else 0

    def run_reward(self):
        # Transition reward pays each achieved level only once per episode.
        score = self.get_score()
        self._score_peak = max(self._score_peak, score)
        reward = self._score_peak - self._reward_previous
        self._reward_previous = self._score_peak
        return reward

    def check_success(self):
        # Goal state, not expert history: a policy may use another valid order.
        self.eval_success = all(self.goal_checks().values())
        return self.eval_success

    def get_instruction(self, instruction=None):
        return instruction or getattr(self, 'instruction', None) or ('将物品紧凑地摆放在显示器前方并远离桌边：先移开杯子，'
                               '推动小书到位，摆好鼠标和键盘，将杯子放到小书上，'
                               '把闭合的订书机移入物品组，最后摆好手机。')

    def gen_instruction(self, env_idx=0):
        return [self.get_instruction()]
