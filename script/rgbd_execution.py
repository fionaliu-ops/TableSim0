"""Sensor-grounded execution: camera RGB/depth and robot encoders only.

No semantic simulator inventory, prop poses, dimensions or segmentation enters
this controller. Ideal follow uses contact-selected bodies and velocities;
it never queries a prop transform or sets a prop pose.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from script.dynamic_sim import TableSimAdapter, request_action, DynamicSimError
from script.symbolic_grasp_execution import SymbolicGraspExecutor, GraspBlockedError
from high_level_planner.perception.mask_codec import decode_binary_mask


class DepthGeometry:
    def __init__(self, depth_mm, intrinsic, extrinsic):
        depth = np.asarray(depth_mm, float) / 1000.
        self.height, self.width = depth.shape
        self.intrinsic = np.asarray(intrinsic, float)
        extrinsic = np.asarray(extrinsic, float)
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])
        self.camera = np.linalg.inv(extrinsic)
        yy, xx = np.indices(depth.shape)
        rays = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(self.intrinsic).T
        self.points = (rays * depth[..., None]) @ self.camera[:3, :3].T + self.camera[:3, 3]
        self.valid = np.isfinite(depth) & (depth > .02) & (depth < 5.)
        self.table_z = self.find_table()

    def find_table(self):
        # A broad horizontal surface, observed in camera depth. No preset height.
        p = self.points
        dx = p[1:-1, 2:] - p[1:-1, :-2]
        dy = p[2:, 1:-1] - p[:-2, 1:-1]
        n = np.cross(dx, dy)
        norm = np.linalg.norm(n, axis=-1)
        horizontal = (abs(n[..., 2]) > .97 * norm) & (norm > 1e-9) & self.valid[1:-1, 1:-1]
        z = p[1:-1, 1:-1, 2][horizontal]
        if len(z) < 100:
            raise DynamicSimError('Camera depth contains no reliable horizontal support plane')
        bins = np.round(z / .005).astype(int)
        values, counts = np.unique(bins, return_counts=True)
        mode = values[np.argmax(counts)] * .005
        return float(np.median(z[abs(z-mode) < .006]))

    def ray_plane(self, xy, z):
        ray = self.camera[:3, :3] @ np.linalg.solve(self.intrinsic, [xy[0]*self.width, xy[1]*self.height, 1.])
        if abs(ray[2]) < 1e-8:
            raise DynamicSimError('Placement ray is parallel to observed support')
        scale = (z-self.camera[2, 3])/ray[2]
        if scale <= 0:
            raise DynamicSimError('Placement is behind camera')
        return self.camera[:3, 3] + scale*ray

    def insertion_approach(self, release, gripper_length):
        """Clear an observed vertical wall with an outward-tilted wrist."""
        points = self.points[self.valid]
        distance = np.linalg.norm(points[:,:2]-release[:2],axis=1)
        nearby = points[(distance < .09) & (points[:,2] > release[2]+.02)
                       & (points[:,2] < release[2]+gripper_length+.05)]
        if len(nearby) < 30 or np.ptp(nearby[:,2]) < .08:
            return None
        center = np.median(nearby,axis=0)
        _, _, vectors = np.linalg.svd(nearby-center, full_matrices=False)
        normal = vectors[-1]
        if abs(normal[2]) > .25 or np.median(abs((nearby-center)@normal)) > .005:
            return None
        direction = normal[:2].copy()
        direction /= max(1e-8,np.linalg.norm(direction))
        if np.dot(center[:2]-release[:2],direction) < 0:
            direction *= -1.
        axis = np.r_[.5*direction,-math.sqrt(.75)]
        stage = release.copy()
        stage[:2] -= direction*(.5*gripper_length+.02)
        return stage, axis, direction

    def mask_points(self, detection):
        mask = decode_binary_mask(detection.mask, expected_image_size=[self.width, self.height])
        if mask is None:
            raise DynamicSimError('RGB-D execution requires a visual object mask')
        return self.points[mask & self.valid]

    def filter(self, result):
        from dataclasses import replace
        kept, rejected = [], []
        for d in result.detections:
            points = self.mask_points(d)
            # Reject only clearly below-table regions, not partially occluded items.
            if len(points) and np.mean(points[:, 2] < self.table_z-.025) > .8:
                rejected.append({'label':d.label, 'bbox':d.bbox, 'reason':'below_observed_table'})
            else:
                if len(points) >= 12:
                    span = np.diff(np.percentile(points, [3, 97], axis=0), axis=0)[0]
                    # Tall near-round objects have no reliable image major-axis yaw.
                    observable = not (span[2] > .7*max(span[0], span[1])
                        and max(span[0], span[1])/max(.001,min(span[0],span[1])) < 1.5)
                    d = replace(d, metadata={**d.metadata, 'rgbd_yaw_axis_observable':bool(observable),
                        'rgbd_observed_height_m':float(np.percentile(points[:,2],97)-self.table_z)})
                kept.append(d)
        result.detections = kept
        result.metadata['rgbd_workspace_filter'] = {'table_z':self.table_z, 'rejected':rejected,
            'source':'camera_depth_and_calibration'}
        return result

    def surface(self, detection):
        pts = self.mask_points(detection)
        pts = pts[pts[:, 2] > self.table_z-.012]
        if len(pts) < 12:
            raise DynamicSimError('Insufficient visible depth for grasp')
        top = float(np.percentile(pts[:, 2], 97))
        low = float(np.percentile(pts[:, 2], 3))
        surface = pts[pts[:, 2] >= top-.012]
        center = np.median(surface, axis=0)
        # Pick a measured surface pixel, not an unobserved mask/bbox centre.
        contact = surface[np.argmin(np.linalg.norm(surface[:, :2]-center[:2], axis=1))].copy()
        extent = np.percentile(pts[:, :2], [3, 97], axis=0)
        return {'contact':contact, 'top':top, 'low':low, 'extent':extent,
                'center':np.median(pts, axis=0), 'points':pts}


def destination_blockers(surfaces, source_id, support_id, xy, bottom):
    """Check the requested support against current camera depth footprints.

    A center inside another visible prop is occupied; boundary overlap alone
    remains undecided. No asset dimensions or recorded coordinates are used.
    """
    point=np.asarray(xy)
    return [key for key,surface in surfaces.items()
        if key not in {source_id,support_id} and surface['top'] > bottom+.002
        and np.all(point > surface['extent'][0]+.005)
        and np.all(point < surface['extent'][1]-.005)]


class ContactVelocityFollow:
    """Ideal grasp physics, selected by fingertip contact, without pose reads.

    The contacted body integrates velocities equal to measured gripper motion.
    Collision response is disabled only while carrying; no target-position write.
    """
    def __init__(self, task, tip):
        self.task, self.tip = task, tip
        self.body = None
        self.arm = None
        self.contacts = []
        self.previous = None
        self.groups = []

    def prepare_symbolic_contacts(self):
        import sapien
        # Keep resting props fixed until the symbolic grasp is closed. Native
        # collision contacts remain enabled for tactile attachment selection.
        # Enumerating rigid bodies configures physics, not planner identities.
        for entity in self.task.scene.get_entities():
            for component in entity.get_components():
                if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                    component.set_kinematic(True)

    def sample(self):
        import sapien
        self.contacts = list(self.task.scene.get_contacts())
        if self.body is None:
            return
        point, rotation = self.tip(self.arm)
        yaw = math.atan2(rotation[1, 1], rotation[0, 1])
        old_point, old_yaw = self.previous
        dt = self.task.scene.get_timestep()
        self.body.set_linear_velocity((point-old_point)/dt)
        delta_yaw = (yaw-old_yaw+math.pi) % (2*math.pi)-math.pi
        self.body.set_angular_velocity([0., 0., delta_yaw/dt])
        self.integrated_yaw = getattr(self, 'integrated_yaw', 0.) + delta_yaw
        self.previous = (point.copy(), yaw)

    def contact_body(self, arm, *, surface_points=None, minimum_z=None):
        import sapien
        point, rotation = self.tip(arm)
        links = getattr(self.task.robot, arm+'_entity').get_links()
        active_entities = {link.entity for link in links}
        robot_entities = {link.entity for a in ('left', 'right')
            for link in getattr(self.task.robot, a+'_entity').get_links()}
        candidates = []
        from scipy.spatial import cKDTree
        visible_surface = cKDTree(surface_points) if surface_points is not None else None
        aperture = getattr(self.task.robot, arm+'_gripper_scale', [-.01,.045])
        contact_radius = max(abs(float(x)) for x in aperture)+.03
        for c in self.contacts:
            bodies = c.bodies
            if not any(b.entity in active_entities for b in bodies):
                continue
            for cp in c.points:
                distance = float(np.linalg.norm(np.asarray(cp.position)-point))
                if distance > contact_radius:
                    continue
                if minimum_z is not None and cp.position[2] < minimum_z:
                    continue
                if visible_surface is not None and visible_surface.query(np.asarray(cp.position))[0] > .012:
                    bounds = np.percentile(surface_points,[1,99],axis=0)
                    if not np.all(np.asarray(cp.position) >= bounds[0]-.006) or not np.all(np.asarray(cp.position) <= bounds[1]+.006):
                        continue
                for body in bodies:
                    if (body.entity not in robot_entities and
                            not isinstance(body, sapien.physx.PhysxRigidStaticComponent)):
                        candidates.append((distance, body))
        if not candidates:
            distances = [float(np.linalg.norm(np.asarray(p.position)-point)) for c in self.contacts for p in c.points]
            raise DynamicSimError(f'No fingertip contact after closing; no object was attached (contacts={len(self.contacts)}, nearest={min(distances, default=None)})')
        return min(candidates, key=lambda pair:pair[0])[1]

    def attach(self, arm, *, surface_points=None, minimum_z=None):
        import sapien
        point, rotation = self.tip(arm)
        body = self.contact_body(arm, surface_points=surface_points, minimum_z=minimum_z)
        if isinstance(body, sapien.physx.PhysxRigidStaticComponent):
            raise DynamicSimError('Contacted asset is static; initialize symbolic grasp assets as kinematic')
        self.body, self.arm = body, arm
        self.integrated_yaw = 0.
        self.groups = [(s, s.get_collision_groups()) for s in body.get_collision_shapes()]
        for shape, _ in self.groups:
            shape.set_collision_groups([0, 0, 0, 0])
        body.set_kinematic(False)
        body.set_disable_gravity(True)
        body.set_linear_damping(0.)
        body.set_angular_damping(0.)
        body.set_linear_velocity([0., 0., 0.])
        body.set_angular_velocity([0., 0., 0.])
        self.previous = (point.copy(), math.atan2(rotation[1, 1], rotation[0, 1]))

    def release(self):
        if self.body is None:
            return
        self.body.set_linear_velocity([0., 0., 0.])
        self.body.set_angular_velocity([0., 0., 0.])
        # Symbolic support validation: preserve the physically integrated pose.
        self.body.set_kinematic(True)
        for shape, groups in self.groups:
            shape.set_collision_groups(groups)
        self.body = None


class ForbiddenObjectTruth:
    def __init__(self, audit, field):
        self.audit, self.field = audit, field
    def fail(self, *args, **kwargs):
        self.audit['blocked_reads'].append(self.field)
        raise DynamicSimError('Forbidden simulator object truth: '+self.field)
    __getitem__ = __iter__ = __len__ = __bool__ = fail
    values = items = keys = get = fail


class RGBDAdapter(TableSimAdapter):
    visual_only_execution = True

    def reset(self, seed=0):
        self.task.symbolic_grasp_assets = True
        observation = super().reset(seed)
        self.truth_audit = {'policy':'camera_rgb_depth_calibration_and_robot_encoders_only',
            'blocked_reads':[], 'guarded_fields':['objects','object_geometry','OBJECT_SPECS','initial_bottom_z','target_bottom_z','_table_z','check_success'],
            'attachment':'contact_selected_velocity_integration'}
        for field in ('objects', 'object_geometry', 'OBJECT_SPECS', 'initial_bottom_z', 'target_bottom_z'):
            setattr(self.task, field, ForbiddenObjectTruth(self.truth_audit, field))
        self.task._table_z = ForbiddenObjectTruth(self.truth_audit, '_table_z').fail
        self.task.check_success = ForbiddenObjectTruth(self.truth_audit, 'check_success').fail
        (self.output_dir/'truth_access_audit.json').write_text(json.dumps(self.truth_audit, indent=2))
        return observation

    def close(self):
        if hasattr(self, 'truth_audit'):
            (self.output_dir/'truth_access_audit.json').write_text(json.dumps(self.truth_audit, indent=2))
        super().close()

    def observe(self):
        frame = self.task.get_obs()
        camera = frame['observation']['head_camera']
        depth = self.task.cameras.get_depth()['head_camera']['depth']
        image = self.output_dir/f'observation_{self.step_index:03d}_{getattr(self, "sensor_index", 0):03d}.png'
        self.sensor_index = getattr(self, 'sensor_index', 0)+1
        Image.fromarray(np.asarray(camera['rgb'], dtype=np.uint8)).save(image)
        geometry = DepthGeometry(depth, camera['intrinsic_cv'], camera['extrinsic_cv'])
        self.sensor_packets = getattr(self, 'sensor_packets', {})
        self.sensor_packets[str(image.resolve())] = geometry
        self.geometry = geometry
        np.savez_compressed(image.with_suffix('.npz'), depth_mm=depth,
            intrinsic_cv=camera['intrinsic_cv'], extrinsic_cv=camera['extrinsic_cv'])
        self.last_observation_image = image
        return {'image':image, 'sensor_source':'head_camera_rgb_depth_calibration',
                'depth_path':image.with_suffix('.npz')}

    def bind_graph(self, graph, mapping=None):
        if mapping:
            raise DynamicSimError('Simulator object aliases are forbidden in RGB-D mode')
        self.current_scene = graph
        self.object_map = {o.id:o.id for o in graph.objects}
        return dict(self.object_map)

    def step(self, request):
        action = request_action(request)
        self.last_placement = None
        if action['kind'] not in {'move','place','stack','insert','align','reorient','attach'}:
            raise DynamicSimError('Unsupported visual execution action: '+action['kind'])
        controller = self.symbolic_grasp_controller
        controller.executing_arm = None
        controller.phase = 'before_grasp'
        try:
            controller.place(action, action['source'])
        except Exception as exc:
            from script.dual_arm_safety import ArmClearanceError, TrajectoryClearanceError
            fatal = isinstance(exc, ArmClearanceError) and not isinstance(exc, TrajectoryClearanceError)
            details = dict(getattr(exc, 'details', {}))
            details.setdefault('phase',getattr(controller,'phase','before_grasp'))
            details.setdefault('failure_category','destination_obstructed' if details['phase'] in
                {'carry_lower','insert_below_obstacle','lower_outside_obstacle'} else 'motion_execution_failed')
            details.setdefault('object_moved',self.held is not None)
            exc.details = details
            if not fatal and self.held is None and controller.executing_arm:
                arm = controller.executing_arm
                try:
                    self._gripper(arm, False)
                    controller.move(arm, controller.tip(arm)[0]+[0,0,.07], 'failed_grasp_withdraw')
                    controller.drive(arm, [controller.home[arm]], 'failed_grasp_home')
                except Exception as recovery:
                    controller.log('empty_grasp_recovery_failed', error=str(recovery))
            raise
        self.step_index += 1
        result = self.observe()
        result['action'] = action
        return result

    def evaluate(self):
        raise DynamicSimError('Use camera goal evaluation; simulator success is forbidden')


class RGBDGraspExecutor(SymbolicGraspExecutor):
    def __init__(self, env):
        super().__init__(env, visual_only=True)
        self.physics = ContactVelocityFollow(self.task, self.tip)

    def tip(self, arm):
        # Joint.global_pose is the joint anchor, before the final revolute
        # rotation. Use child-link FK with all encoder angles for wrist yaw.
        return self.fk(arm, self.models[arm][0].get_qpos())

    def install(self):
        self.physics.prepare_symbolic_contacts()
        original = self.task._sync_attached_actor
        def sync():
            self.physics.sample()
            original()
        self.task._sync_attached_actor = sync

    def clear_camera_view(self):
        # Empty-arm parking derived from current encoders and robot clearance.
        clearance = self.env.arm_safety.minimum_clearance + self.env.arm_safety.planning_margin
        for arm in ('left','right'):
            tip, rotation = self.tip(arm)
            direction = np.sign(self.arm_bases[arm][0]) or (-1. if arm=='left' else 1.)
            destination = tip + [direction*2*clearance, 0., clearance]
            try:
                self.log('camera_view_parking_request', arm=arm, tip=tip.tolist(), destination=destination.tolist())
                self.move(arm, tip+[0,0,clearance], 'camera_view_lift', tool_axis=rotation[:,0])
                self.move(arm, destination, 'clear_camera_view', tool_axis=rotation[:,0])
            except DynamicSimError as exc:
                self.log('camera_view_parking_failed', arm=arm, error=str(exc))
                self.drive(arm, [self.home[arm]], 'camera_view_parking_recovery')
            self.home[arm] = np.asarray(self.models[arm][0].get_qpos()).copy()

    def move(self, arm, xyz, label, closing_yaw=None, tool_axis=None):
        self.phase = label
        return super().move(arm, xyz, label, closing_yaw, tool_axis)

    def recover_held(self):
        if self.attachment is None:
            return
        arm = self.attachment['arm']
        self.drive(arm, list(reversed(self.attachment['return_path'])), 'failure_return')
        self.env._gripper(arm, False)
        self.physics.release()
        self.attachment = None
        self.env.held = None
        retreat = self.tip(arm)[0]+[0,0,.07]
        self.move(arm, retreat, 'failure_retreat')
        self.drive(arm, [self.home[arm]], 'failure_home')

    def place(self, action, name):
        env = self.env
        if env.held is not None:
            raise DynamicSimError('Previous payload still held')
        graph = env.current_scene
        obj = graph.get_object(name)
        if obj is None or not obj.metadata.get('observed_in_current_frame', True):
            raise DynamicSimError('Source not visible in current camera frame')
        if obj.metadata.get('visual_identity_confirmed') is False:
            raise GraspBlockedError({'failure_category':'visual_identity_unresolved',
                'source_object_id':name, 'blocking_object_ids':[], 'phase':'before_grasp',
                'object_moved':False})
        result = env.perception.detect(env.last_observation_image, instruction='detect every tabletop object')
        # Use current visual boxes to associate masks; never simulator names.
        def detection_for(node):
            box = np.asarray(node.attributes.bbox)
            if not result.detections:
                raise DynamicSimError('No current object masks')
            detection = min(result.detections, key=lambda d:np.linalg.norm(np.asarray(d.bbox)-box))
            if np.linalg.norm(np.asarray(detection.bbox)-box) > 35:
                raise DynamicSimError('Current mask does not match visual source')
            return detection
        g = env.geometry
        surfaces = {}
        for node in graph.objects:
            if node.attributes.movable and node.metadata.get('observed_in_current_frame', True):
                try:
                    surfaces[node.id] = g.surface(detection_for(node))
                except DynamicSimError:
                    pass
        if name not in surfaces:
            raise DynamicSimError('No current depth-supported source surface')
        source = surfaces[name]
        contact = source['contact'].copy()
        blockers = [r.source for r in graph.relations
            if r.target == name and r.relation in {'on','blocks'}]
        for key, other in surfaces.items():
            if key == name or other['top'] <= source['top']+.018:
                continue
            # A higher surface inside the source footprint can obstruct pickup.
            lo, hi = source['extent']
            if np.all(other['center'][:2] > lo) and np.all(other['center'][:2] < hi):
                if key not in blockers:
                    blockers.append(key)
        if blockers:
            raise GraspBlockedError({'failure_category':'grasp_occluded', 'source_object_id':name,
                'blocking_object_ids':blockers, 'phase':'before_grasp', 'object_moved':False})
        support_z = g.table_z
        for relation in graph.relations:
            if relation.source == name and relation.relation == 'on' and relation.target in surfaces:
                support_z = max(support_z, surfaces[relation.target]['top'])
        for key, other in surfaces.items():
            if key == name or other['top'] >= source['top']-.005:
                continue
            if np.all(contact[:2] >= other['extent'][0]) and np.all(contact[:2] <= other['extent'][1]):
                support_z = max(support_z, other['top'])
        height = max(.006, source['top']-support_z)
        params = action.get('parameters', {})
        placement = params.get('pose') or params.get('relative_pose') or params.get('placement_target', {}).get('parameters', {}).get('pose', {})
        target_id = action.get('target') or placement.get('support')
        if target_id is None and action['kind'] in {'align','reorient'}:
            from high_level_planner.reasoning_graph.geometry import pose
            target_id = (pose(graph,name) or {}).get('support','table')
        target_support = surfaces.get(target_id)
        if target_id not in (None, 'table') and target_support is None:
            raise DynamicSimError('Target support lacks current visual depth')
        bottom = target_support['top'] if target_support else g.table_z
        xy = placement.get('xy') or obj.attributes.xy
        # Bbox centre approximates the half-height plane, not the top surface.
        destination = g.ray_plane(xy, bottom+height/2)
        blockers=destination_blockers(surfaces,name,target_id,destination[:2],bottom)
        if blockers:
            raise GraspBlockedError({'failure_category':'destination_occupied',
                'source_object_id':name,'blocking_object_ids':blockers,
                'phase':'before_grasp','object_moved':False,
                'reason':'requested support is covered at the placement center in current camera depth'})
        penetration = min(.008, height*.4)
        contact[2] -= penetration
        source_center = g.ray_plane(obj.attributes.xy, support_z+height/2)
        release = np.r_[destination[:2]+contact[:2]-source_center[:2], contact[2]+bottom-support_z]
        clutter_top = max([s['top'] for key,s in surfaces.items() if key != name]+[bottom])
        high_z = max(contact[2], release[2], clutter_top+height) + .025
        lift = np.r_[contact[:2], high_z]
        insertion = g.insertion_approach(release, self.pad_offset)
        stage, tool_axis, wall_direction = insertion if insertion else (release, None, None)
        high = np.r_[stage[:2], high_z]
        approaches = [contact+[0,0,.07],contact,lift,high,stage,release] if insertion else [contact+[0,0,.07],contact,lift,high,release]
        arm = None
        errors = []
        for candidate in sorted(self.models, key=lambda a: np.linalg.norm(contact[:2]-self.arm_bases[a][:2])+np.linalg.norm(release[:2]-self.arm_bases[a][:2])):
            qpos = self.models[candidate][0].get_qpos()
            try:
                for waypoint_index, waypoint in enumerate(approaches):
                    qpos = self.solve(candidate, waypoint, qpos,
                        tool_axis=tool_axis if insertion and waypoint_index>=4 else None)
                arm = candidate
                break
            except DynamicSimError as exc:
                errors.append(str(exc))
        if arm is None:
            raise DynamicSimError('RGB-D route unreachable: '+str(errors))
        self.log('rgbd_grounding', object=name, arm=arm, table_z=g.table_z,
            contact=contact.tolist(), release=release.tolist(), estimated_height=height,
            source='rgb_sam_mask_camera_depth_calibration')
        other = 'left' if arm == 'right' else 'right'
        self.executing_arm = arm
        self.drive(other, [self.home[other]], 'park_other_arm')
        env._gripper(arm, False)
        self.move(arm, approaches[0], 'grasp_approach')
        try:
            self.move(arm, contact, 'grasp_contact')
        except DynamicSimError:
            # A wide/thin prop may stop the fingers before the nominal tip
            # waypoint. Actual contact with the observed surface is sufficient.
            self.physics.contact_body(arm, surface_points=source['points'], minimum_z=g.table_z+.002)
            self.log('surface_contact_before_nominal_tip', object=name, arm=arm)
        env._gripper(arm, True)
        self.physics.attach(arm, surface_points=source['points'], minimum_z=g.table_z+.002)
        self.attachment = {'arm':arm, 'name':name,
            'return_path':[np.asarray(self.models[arm][0].get_qpos()).copy()]}
        env.held = (name, arm)
        contact_offset = self.tip(arm)[0]-contact
        release = release+contact_offset
        stage = stage+contact_offset
        high = high+contact_offset
        lift = lift+contact_offset
        self.log('attached_after_contact_and_close', object=name, arm=arm)
        _, rotation = self.tip(arm)
        closing_yaw = math.atan2(rotation[1,1], rotation[0,1])
        desired = placement.get('yaw_degrees', params.get('target_yaw_degrees'))
        if desired is not None:
            from high_level_planner.reasoning_graph.geometry import pose
            observed_yaw = (pose(graph, name) or {}).get('yaw_degrees', 0.)
            closing_yaw -= math.radians(float(desired)-float(observed_yaw))
        if insertion and detection_for(obj).metadata.get('rgbd_yaw_axis_observable') is False:
            closing_yaw = math.atan2(-wall_direction[0], wall_direction[1])
        self.move(arm, lift, 'carry_lift')
        self.move(arm, high, 'carry_transfer', closing_yaw)
        if insertion:
            self.log('rgbd_vertical_obstacle_insertion', stage=stage.tolist(), tool_axis=tool_axis.tolist())
            self.move(arm, stage, 'lower_outside_obstacle', closing_yaw, tool_axis)
            self.move(arm, release, 'insert_below_obstacle', closing_yaw, tool_axis)
        else:
            self.move(arm, release, 'carry_lower', closing_yaw)
        env._gripper(arm, False)
        self.physics.release()
        self.attachment = None
        env.held = None
        self.log('released_after_open', object=name, arm=arm,
            integrated_wrist_yaw_deg=math.degrees(self.physics.integrated_yaw),
            desired_visual_yaw_deg=desired, commanded_wrist_yaw_deg=math.degrees(closing_yaw))
        if insertion:
            self.move(arm, stage, 'withdraw_from_obstacle', closing_yaw, tool_axis)
        self.move(arm, self.tip(arm)[0]+[0,0,.07], 'release_retreat')
        self.drive(arm, [self.home[arm]], 'return_empty_arm')
        env.last_placement = {'object':name,'arm':arm,'mode':'rgbd_contact_velocity_follow',
            'robot_motion_executed':True,'contact_physics_verified':False,
            'object_truth_read':False, 'requires_visual_verification':True}


from high_level_planner.scene_graph.constructor import SceneGraphConstructor
from high_level_planner.scene_graph.graph import Relation


class RGBDSceneGraphConstructor(SceneGraphConstructor):
    """Derive support direction from observed depth instead of bbox overlap."""
    def __init__(self, *, sensor_packets, **kwargs):
        super().__init__(**kwargs)
        self.sensor_packets = sensor_packets

    def construct(self, image, instruction=None, *, previous_graph=None):
        graph = super().construct(image, instruction, previous_graph=previous_graph)
        geometry = self.sensor_packets().get(str(Path(image).resolve()))
        if geometry is None:
            return graph
        result = self.perception_backend.detect(image, instruction=instruction)
        surfaces = {}
        masks = {}
        for obj in graph.objects:
            if obj.attributes.bbox is None:
                continue
            detection = min(result.detections, key=lambda d:np.linalg.norm(np.asarray(d.bbox)-obj.attributes.bbox))
            if np.linalg.norm(np.asarray(detection.bbox)-obj.attributes.bbox) > 3:
                continue
            try:
                surfaces[obj.id] = geometry.surface(detection)
                masks[obj.id] = decode_binary_mask(detection.mask,
                    expected_image_size=[geometry.width,geometry.height])
            except DynamicSimError:
                continue
        for source, a in surfaces.items():
            if not graph.get_object(source).attributes.movable:
                continue
            candidates = []
            for target, b in surfaces.items():
                if source == target or a['top'] <= b['top']+.004:
                    continue
                overlap = np.maximum(0., np.minimum(a['extent'][1], b['extent'][1])-np.maximum(a['extent'][0], b['extent'][0]))
                fraction = np.prod(overlap)/max(1e-8,np.prod(a['extent'][1]-a['extent'][0]))
                gap = a['low']-b['top']
                if fraction > .15 and -.01 <= gap <= .035:
                    candidates.append((b['top'],target))
            if candidates:
                _, target = max(candidates)
                # Semantic containment inferred from labels/bboxes is weaker
                # than a measured upper surface resting on this support.
                graph.relations = [r for r in graph.relations if not
                    (r.source == source and r.target == target and r.relation in {'in','inside'})]
                graph.add_relation(Relation(source,target,'on',confidence=.8,
                    metadata={'evidence':'rgbd_visible_surface_support','not_simulator_truth':True}))
        # Bbox intersection is symmetric; actual occlusion is not. A book
        # resting on a folder does not become hidden by its own support.
        for obj in graph.objects:
            if obj.id not in surfaces:
                continue
            own = surfaces[obj.id]
            occluders = []
            x1,y1,x2,y2 = np.round(obj.attributes.bbox).astype(int)
            x1,x2 = np.clip([x1,x2],0,geometry.width)
            y1,y2 = np.clip([y1,y2],0,geometry.height)
            area = max(1,(x2-x1)*(y2-y1))
            for other_id, other in surfaces.items():
                if other_id == obj.id or masks.get(other_id) is None:
                    continue
                # Occlusion is measured along camera rays, not by overlap
                # in tabletop XY (a screen above/behind a book is not
                # necessarily in front of that book in the image).
                upper = geometry.points[y1:y2,x1:x2,2] > own['top']+.004
                overlap = masks[other_id][y1:y2,x1:x2] & upper & geometry.valid[y1:y2,x1:x2]
                if np.count_nonzero(overlap) >= max(12,.02*area):
                    occluders.append(other_id)
            obj.attributes.visibility = 'partially_visible' if occluders else 'fully_visible'
            obj.attributes.extra['visibility_evidence'] = ['rgbd_occluded_by:'+key for key in occluders] or ['rgbd_no_visible_upper_occluder']
        from script.online_rearrangement import _mask_axis
        axes = {}
        for obj in graph.objects:
            if not obj.attributes.movable or obj.attributes.bbox is None:
                continue
            detection = min(result.detections,key=lambda d:np.linalg.norm(np.asarray(d.bbox)-obj.attributes.bbox))
            axis = _mask_axis(detection,result.image_size)
            if axis is not None and axis[1] >= 1.15:
                axes[obj.id] = axis[0]
        ids = sorted(axes)
        for index, left in enumerate(ids):
            for right in ids[index+1:]:
                a,b = graph.get_object(left), graph.get_object(right)
                delta = abs((axes[left]-axes[right]+90)%180-90)
                if delta <= 8. and np.linalg.norm(np.asarray(a.attributes.xy)-b.attributes.xy) <= .04:
                    graph.add_relation(Relation(left,right,'aligned_with',confidence=.8,
                        metadata={'evidence':'current_rgb_mask_axes_and_centers'}))
        graph.metadata['support_evidence_source'] = 'camera_depth_and_sam_masks'
        return graph
