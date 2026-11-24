"""Contact-driven actions; never attach, teleport, or animate an object pose.

The existing ALOHA finger tips are 3 mm thick. Book grasp frames point the tool
X axis horizontally into an edge and the jaw-closing Y axis vertically. Asset
geometry, mass, material and initial transforms are owned by the scene task.
"""
from __future__ import annotations

import numpy as np
import sapien
import transforms3d as t3d


def horizontal_book_frame(inward, wrist_pitch_degrees=0.):
    # A small downward pitch raises the wrist/palm above the table while the
    # fingertips still enter from the edge (never through the top face).
    slope = np.tan(np.deg2rad(wrist_pitch_degrees))
    x = np.asarray([inward[0], inward[1], -slope], dtype=float)
    x /= np.linalg.norm(x)
    y = np.array([0., 0., 1.])
    y -= np.dot(y,x)*x
    y /= np.linalg.norm(y)
    return np.column_stack((x, y, np.cross(x, y)))


def side_grasp_geometry(center, footprint, height, bottom, yaw, edge='front', wrist_pitch_degrees=12.):
    """Return an edge target, NOT the book center; all lengths are simulator units."""
    directions = {'front':(0,-1), 'back':(0,1), 'left':(-1,0), 'right':(1,0)}
    outward = np.asarray(directions[edge],dtype=float)
    rotation = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    # Keep the contact at the edge: the remaining tip extends another 8 mm in.
    local = outward * (np.asarray(footprint)/2 - .002)
    xy = np.asarray(center)[:2] + rotation @ local
    inward = -(rotation @ outward)
    return np.r_[xy, bottom + height/2], horizontal_book_frame(inward,wrist_pitch_degrees)


class PhysicalManipulation:
    FINGER_TIP_THICKNESS = .003  # measured from unchanged link7/8.STL distal tips
    FINGER_CONTACT_OFFSET = .147  # URDF joint x=.08457 + distal pad x≈.063
    LIFT_HEIGHT = .07

    def _run_move(self, stage_name, *actions):
        if not super()._run_move(stage_name,*actions):
            return False
        for arm, sequence in actions:
            if not sequence or sequence[-1].action != 'move':
                continue
            target=sequence[-1].target_pose
            if isinstance(target,sapien.Pose):
                target=np.r_[target.p,target.q]
            target=np.asarray(target,dtype=float)
            actual=np.asarray(self._current_ee_pose(str(arm)),dtype=float)
            error=float(np.linalg.norm(actual[:3]-target[:3]))
            if error>.02:
                self.last_motion_error={'stage':stage_name,'position_error':error,
                                        'requested':target.tolist(),'observed':actual.tolist()}
                self.plan_success=False
                print(f'[contact_physics] controller tracking error at {stage_name}: {error:.4f}',flush=True)
                return False
        return True

    def take_dense_action(self, control_seq, save_freq=-1):
        from .._base_task import Base_Task
        if getattr(self,'_slow_contact',False):
            control_seq=dict(control_seq)
            for key in ('left_arm','right_arm'):
                path=control_seq.get(key)
                if path is not None:
                    path=dict(path)
                    count=len(path['position'])
                    samples=np.linspace(0,count-1,max(2,count*3))
                    for field in ('position','velocity'):
                        values=np.asarray(path[field])
                        path[field]=np.column_stack([np.interp(samples,np.arange(count),values[:,j]) for j in range(values.shape[1])])
                    path['velocity']/=3
                    control_seq[key]=path
        return Base_Task.take_dense_action(self, control_seq, save_freq=save_freq)

    def _attach_actor(self, *args):
        raise RuntimeError('Kinematic object attachment is disabled')

    def _detach_actor_at(self, *args):
        raise RuntimeError('Teleport placement is disabled')

    def _reject(self, stage, reason):
        self.failure_reason = {'stage':stage, 'reason':reason}
        if getattr(self,'last_motion_error',None):
            self.failure_reason['controller_tracking_error']=self.last_motion_error
        for arm in ('left','right'):
            paths = getattr(self,arm+'_joint_path',[])
            if paths and paths[-1].get('failure_status'):
                self.failure_reason[arm+'_planner_status'] = paths[-1]['failure_status']
        self.plan_success = False
        print(f'[contact_physics] {stage}: {reason}', flush=True)
        return False

    def _entity_names(self, name):
        obj = self.objects[name].actor
        if hasattr(obj, 'get_links'):
            return {link.entity.get_name() for link in obj.get_links()}
        return {obj.get_name()}

    def _contacts(self, name, arm_name):
        joints = self.robot.left_gripper if arm_name == 'left' else self.robot.right_gripper
        fingers = {j[0].child_link.get_name() for j in joints}
        objects = self._entity_names(name)
        touched = set()
        for contact in self.scene.get_contacts():
            first, second = (b.entity.get_name() for b in contact.bodies)
            if first in objects and second in fingers:
                touched.add(second)
            if second in objects and first in fingers:
                touched.add(first)
        return touched, fingers

    def _held(self, name, arm_name):
        touched, fingers = self._contacts(name, arm_name)
        return len(fingers) == 2 and touched == fingers

    @staticmethod
    def _tool_pose(contact, rotation, offset=.147):
        return np.r_[np.asarray(contact)-rotation[:,0]*offset,
                     t3d.quaternions.mat2quat(rotation)].tolist()

    def _jaw_command(self, aperture, arm_name):
        # Symmetric ±q prismatic fingers. Use the existing robot scale.
        scale = self.robot.left_gripper_scale if arm_name == 'left' else self.robot.right_gripper_scale
        return float(np.clip((aperture/2-scale[0])/(scale[1]-scale[0]),0,1))

    def _side_candidate(self, name, edge_order=('front','right','back','left'), pitch=12.):
        obj = self.objects[name]
        pose = obj.get_pose()
        yaw = np.arctan2(pose.to_transformation_matrix()[1,0],pose.to_transformation_matrix()[0,0])
        geo = self.object_geometry[name]
        aperture = geo['height'] + .006
        if aperture > .09:
            return None, 'Book thickness exceeds the unchanged gripper opening'
        bottom = self._bottom(name)
        clearance = bottom - self._table_z()
        required = self.FINGER_TIP_THICKNESS + .003
        front_edge = float(getattr(self, 'table_xy_bias', [0,0])[1]) - .35
        front_overhang = float(pose.p[1]-geo['footprint'][1]/2) < front_edge-.012
        if clearance < required and not front_overhang:
            return None, ('No space for the lower finger: book/table clearance '
                          f'{clearance:.4f}, required {required:.4f}. A physical edge-slide is required first.')
        for edge in edge_order:
            point, rotation = side_grasp_geometry(pose.p,geo['footprint'],geo['height'],bottom,yaw,edge,pitch)
            # An overhang elsewhere on the book does not clear this finger.
            # Check the actual insertion region in world coordinates.
            probe = point[:2]+rotation[:2,0]*.008
            if clearance < required and probe[1] >= front_edge-.004:
                continue
            # Check the lower tip's insertion region against nearby supports.
            blocked = False
            for other in self.objects:
                if other == name:
                    continue
                lower_top = self._bottom(other)+self.object_geometry[other]['height']
                other_pose = self.objects[other].get_pose()
                local = other_pose.to_transformation_matrix()[:2,:2]
                # Rigid meshes are Y-up; recover the planar yaw explicitly.
                angle = np.arctan2(local[1,0],local[0,0])
                r = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                delta = np.abs(r.T@(probe-other_pose.p[:2]))
                if (np.all(delta < self.object_geometry[other]['footprint']/2)
                        and lower_top > bottom-required and self._bottom(other) < point[2]+aperture/2):
                    blocked = True
                    break
            if not blocked:
                return (point, rotation, aperture, edge), None
        return None, 'Every side approach intersects another object/support; clear or physically slide the book first'

    def _select_side_grasp(self,name,arm_name,stage):
        choices = getattr(self,'grasp_choices',{})
        if not self.need_plan:
            if stage not in choices:
                return None,'Replay lacks the physically planned grasp choice'
            choice = choices[stage]
            return self._side_candidate(name,(choice['edge'],),choice['pitch'])
        planner = self.robot.left_plan_path if arm_name=='left' else self.robot.right_plan_path
        entity = self.robot.left_entity if arm_name=='left' else self.robot.right_entity
        arm_joints = self.robot.left_arm_joints if arm_name=='left' else self.robot.right_arm_joints
        names = [joint.get_name() for joint in entity.get_active_joints()]
        indices = [names.index(joint.get_name()) for joint in arm_joints]
        failures=[]
        for edge in ('front','right','back','left'):
            for pitch in (12.,18.,24.):
                candidate, reason = self._side_candidate(name,(edge,),pitch)
                if candidate is None:
                    failures.append(f'{edge}/{pitch}: {reason}')
                    continue
                point,rotation,_,_=candidate
                approach = self._tool_pose(point-rotation[:,0]*.065,rotation)
                high = np.asarray(approach).copy(); high[2] += .12
                qpos = entity.get_qpos().copy()
                for pose in (high.tolist(),approach,self._tool_pose(point,rotation)):
                    result = planner(pose,last_qpos=qpos)
                    if result['status'] != 'Success':
                        failures.append(f'{edge}/{pitch}: no collision-free IK/path')
                        break
                    qpos[indices] = result['position'][-1]
                else:
                    self.grasp_choices[stage]={'edge':edge,'pitch':pitch}
                    return candidate,None
        return None,'No reachable collision-free side grasp: '+'; '.join(failures)

    def _pick_and_place(self, name, stage):
        if name in ('book','keyboard'):
            return self._push_to_xy(name,self.OBJECT_SPECS[name]['target_xy'],stage,complete_stage=True)
        arm_name = self.OBJECT_SPECS[name]['arm']
        arm = self._arm_tag(arm_name)
        actor = self.objects[name]
        before_z = float(actor.get_pose().p[2])
        is_book = name in ('book','folder')
        if is_book:
            candidate, reason = self._select_side_grasp(name,arm_name,stage)
            if candidate is None:
                return self._reject(stage, reason)
            point, rotation, aperture, edge = candidate
            grasp = self._tool_pose(point,rotation)
            approach = self._tool_pose(point-rotation[:,0]*.065,rotation)
            high = np.asarray(approach).copy()
            high[2] += .12
            opening = self._jaw_command(aperture,arm_name)
            mode = 'horizontal_side_pinch_'+edge
        else:
            from .._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
            rotation = t3d.quaternions.quat2mat(GRASP_DIRECTION_DIC['top_down'])
            height = self.object_geometry[name]['height']
            contact_z = max(self._table_z()+.040,self._bottom(name)+height-.018)
            point = np.r_[actor.get_pose().p[:2],contact_z]
            grasp = self._tool_pose(point,rotation,offset=.12)
            approach = np.asarray(grasp).copy()
            approach[2] += .075
            high = approach
            opening, mode = 1., 'top_down_sidewall_pinch'
        log = self.physical_grasp_log[stage] = {'object':name,'mode':mode,'two_finger_contact':False,
                                              'lift_verified':False,'transfer_verified':False}
        log['grasp_pose'] = list(grasp)
        log['approach_pose'] = list(approach)
        log['object_pose_before'] = [*map(float,actor.get_pose().p),*map(float,actor.get_pose().q)]
        log['motion_log'] = []
        for suffix, action in [('open',self.open_gripper(arm,pos=opening)),
                               ('orient_clear_of_table',self.move_to_pose(arm,list(high))),
                               ('approach_edge',self.move_to_pose(arm,list(approach))),
                               ('insert',self.move_to_pose(arm,list(grasp))),
                               ('pinch',self.close_gripper(arm,pos=0.))]:
            if not self._run_move(stage+'/'+suffix,action):
                return self._reject(stage,'Motion planning failed at '+suffix)
            log['motion_log'].append({'substep':suffix,'actual_ee':list(map(float,self._current_ee_pose(arm_name)))})
        touched, expected = self._contacts(name,arm_name)
        log['contact_links'] = sorted(touched)
        log['expected_links'] = sorted(expected)
        log['two_finger_contact'] = self._held(name,arm_name)
        if not log['two_finger_contact']:
            return self._reject(stage,'No verified contact with both fingers; refusing to attach or lift the object')
        # Lift vertically; a side grasp must not withdraw sideways under load.
        lifted = np.asarray(self._current_ee_pose(arm_name),dtype=float)
        lifted[2] += self.LIFT_HEIGHT
        if not self._run_move(stage+'/lift',self.move_to_pose(arm,lifted.tolist())):
            return self._reject(stage,'Lift trajectory infeasible')
        log['lift_delta'] = float(actor.get_pose().p[2])-before_z
        log['lift_verified'] = log['lift_delta'] > .035 and self._held(name,arm_name)
        if not log['lift_verified']:
            return self._reject(stage,'Object did not lift with the fingers or slipped; no forced pose correction')
        # Derive the desired EE transform from the ACTUALLY grasped relation.
        held_pose = actor.get_pose().to_transformation_matrix()
        ee_pose = self._pose_matrix(self._current_ee_pose(arm_name))
        object_to_ee = np.linalg.inv(held_pose)@ee_pose
        target_pose = self._target_pose(name).to_transformation_matrix()
        # Release above the support so a horizontal lower finger can withdraw.
        target_pose[2,3] += .008 if is_book else .004
        target_ee = target_pose@object_to_ee
        above = target_ee.copy()
        above[2,3] += .025 if name=='cup' and stage.startswith('07_') else .04
        start = np.asarray(self._current_ee_pose(arm_name))[:3]
        for i, position in enumerate(np.linspace(start,above[:3,3],6)[1:]):
            waypoint = above.copy()
            waypoint[:3,3] = position
            pose = self._matrix_pose(waypoint)
            if not self._run_move(f'{stage}/transfer_{i}',self.move_to_pose(arm,[*pose.p,*pose.q])):
                return self._reject(stage,'Transfer trajectory infeasible')
            if not self._held(name,arm_name):
                return self._reject(stage,'Lost finger contact during transfer')
        log['transfer_verified'] = True
        pose = self._matrix_pose(target_ee)
        if not self._run_move(stage+'/lower',self.move_to_pose(arm,[*pose.p,*pose.q])):
            return self._reject(stage,'Placement trajectory infeasible')
        if not self._run_move(stage+'/release',self.open_gripper(arm,pos=opening)):
            return self._reject(stage,'Release failed')
        retreat = np.asarray(self._current_ee_pose(arm_name))
        if is_book:
            retreat[:3] -= target_ee[:3,0]*.08
        else:
            retreat[2] += .045
        if not self._run_move(stage+'/withdraw',self.move_to_pose(arm,retreat.tolist())):
            return self._reject(stage,'Finger withdrawal failed')
        self.delay(1)
        log['placement_error_xy'] = float(np.linalg.norm(actor.get_pose().p[:2]-self._target_pose(name).p[:2]))
        if log['placement_error_xy'] > .03:
            return self._reject(stage,'Released object missed the target')
        self.completed_stages.append(stage)
        return True

    def _push_to_xy(self,name,target_xy,stage,complete_stage=False):
        """Closed fingers push a side wall, using measured position feedback.

        Prefer X then Y segments to avoid sweeping a diagonal corridor across
        unrelated desk objects. No object transform is assigned.
        """
        actor = self.objects[name]
        arm_name = self.OBJECT_SPECS[name]['arm']
        arm = self._arm_tag(arm_name)
        start = actor.get_pose().p.copy()
        log = self.physical_grasp_log[stage] = {'object':name,'mode':'contact_push',
                                              'position_before':start.tolist(),'segments':[]}
        # Revisit both axes: a real object may rotate/drift during a push.
        # Feedback corrects that drift instead of snapping the final pose.
        for axis in ((0,1)*20 if name=='folder' else (0,1,0,1,0,1)):
            selected_arm = False
            previous_direction = None
            for iteration in range(8 if name=='folder' else 60):
                pose = actor.get_pose()
                error = float(target_xy[axis]-pose.p[axis])
                if abs(error) < (.022 if name=='stapler' else .012):
                    break
                direction = np.zeros(2); direction[axis]=np.sign(error)
                yaw = np.arctan2(pose.to_transformation_matrix()[1,0],pose.to_transformation_matrix()[0,0])
                r = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
                radius = float(np.sum(np.abs(r.T@direction)*self.object_geometry[name]['footprint']/2))
                if (name=='stapler' and axis==0) or (name=='book' and '/clear_' in stage):
                    # Contact the side intersected by the center-line ray,
                    # not the far AABB corner of a rotated narrow rectangle.
                    ray=np.abs(r.T@direction)
                    radius=float(np.min((self.object_geometry[name]['footprint']/2)/np.maximum(ray,1e-9)))
                push_center=pose.p[:2].copy()
                if name=='folder' and abs(actor.angle-np.pi)<.1:
                    width,depth=self.object_geometry[name]['footprint']
                    push_center-=r@np.array([width/2,0.])
                    radius=float(np.min(np.array([width,depth/2])/np.maximum(np.abs(r.T@direction),1e-9)))
                    if axis==1:
                        push_center[0]-=np.clip(.025+.20*yaw,-.07,.07)*(-direction[1])
                        local_offset=r.T@(push_center-(pose.p[:2]-r@np.array([width/2,0.])))
                        outward=r.T@(-direction)
                        radius=float(np.min((np.array([width,depth/2])-np.sign(outward)*local_offset)/np.maximum(np.abs(outward),1e-9)))
                # Tool X points down; the closed finger pair presents a side
                # surface normal to the intended push direction.
                from .._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
                rotation=t3d.quaternions.quat2mat(GRASP_DIRECTION_DIC['top_down'])
                z = self._bottom(name)+min(self.object_geometry[name]['height']*.5,.012)+.031
                point=np.r_[push_center-direction*(radius+.009),z]
                preferred_arm = 'left' if point[0]<0 else 'right'
                if name=='folder':
                    rotation=t3d.axangles.axangle2mat([1,0,0],np.deg2rad(35))@rotation
                    point[2]=self._bottom(name)+.004+.031*abs(rotation[2,0])
                    point[:2]-=rotation[:2,0]*.031
                if preferred_arm == 'left' and name!='folder' and (axis==1 or point[1]>-.20):
                    rotation=t3d.quaternions.quat2mat(GRASP_DIRECTION_DIC['left_arm_perf'])
                    point[2]=self._bottom(name)+min(self.object_geometry[name]['height']*.5,.012)+.031*abs(rotation[2,0])
                    if '/clear_' in stage or name=='folder':
                        point[:2]-=rotation[:2,0]*.031
                switch_arm = selected_arm and preferred_arm != arm_name
                reverse_direction = selected_arm and np.dot(direction,previous_direction)<0
                if switch_arm or reverse_direction:
                    retreat=np.asarray(self._current_ee_pose(arm_name))
                    retreat[:2]-=previous_direction*.03
                    retreat[2]+=.025
                    if not self._run_move(stage+'/switch_arm_retreat',self.move_to_pose(arm,retreat.tolist())):
                        return self._reject(stage,'Cannot clear the first pushing arm')
                    if not self._run_move(stage+'/switch_arm_open',self.open_gripper(arm)):
                        return False
                    if not self._run_move(stage+'/switch_arm_home',self.back_to_origin(arm)):
                        return False
                if iteration == 0 or switch_arm or reverse_direction:
                    outside=point.copy(); outside[:2]-=direction*.035
                    arm_name = preferred_arm
                    arm = self._arm_tag(arm_name)
                    selected_arm = True
                    approach=self._tool_pose(outside,rotation,offset=.12)
                    above=np.asarray(approach).copy(); above[2]+=.045
                    for suffix,action in [('above',self.move_to_pose(arm,above.tolist())),
                                          ('close',self.close_gripper(arm,pos=0.)),
                                          ('side',self.move_to_pose(arm,approach))]:
                        if not self._run_move(stage+f'/push_{axis}_{suffix}',action):
                            return self._reject(stage,'Cannot reach a collision-free pushing pose')
                point[:2]+=direction*min(.012 if name=='stapler' else .016,abs(error)+.009)
                self._slow_contact=True
                try:
                    moved=self._run_move(stage+f'/push_{axis}_{iteration}',
                                         self.move_to_pose(arm,self._tool_pose(point,rotation,offset=.12)))
                finally:
                    self._slow_contact=False
                if not moved:
                    return self._reject(stage,'Contact push trajectory infeasible')
                previous_direction=direction.copy()
                log['segments'].append({'axis':axis,'position':actor.get_pose().p.tolist()})
                if iteration>3 and abs(float(actor.get_pose().p[axis]-pose.p[axis]))<.0005:
                    return self._reject(stage,'No measured sliding under contact; refusing to move the object by code')
            else:
                if name!='folder':
                    return self._reject(stage,'Contact push did not converge within the bounded stroke count')
            retreat=np.asarray(self._current_ee_pose(arm_name))
            if selected_arm:
                retreat[:2]-=direction*.03
                retreat[2]+=.025
            if selected_arm:
                if not self._run_move(stage+f'/push_{axis}_retreat',self.move_to_pose(arm,retreat.tolist())):
                    return self._reject(stage,'Cannot retreat after pushing')
                if not self._run_move(stage+f'/push_{axis}_open',self.open_gripper(arm)):
                    return False
                if not self._run_move(stage+f'/push_{axis}_home',self.back_to_origin(arm)):
                    return self._reject(stage,'Cannot return the pushing arm to its home pose')
            if name=='folder' and np.linalg.norm(actor.get_pose().p[:2]-np.asarray(target_xy))<.012:
                break
            if name=='keyboard' and np.linalg.norm(actor.get_pose().p[:2]-np.asarray(target_xy))<.018:
                break
        if not self._run_move(stage+'/push_open',self.open_gripper(arm)):
            return False
        self.delay(1)
        log['position_after']=actor.get_pose().p.tolist()
        log['error_xy']=float(np.linalg.norm(actor.get_pose().p[:2]-np.asarray(target_xy)))
        if log['error_xy']>.025:
            return self._reject(stage,'Pushed object did not remain near target after settling')
        if complete_stage:
            self.completed_stages.append(stage)
        return True

    def _physical_close_prop(self, name, stage):
        # Do not assign qpos during execution. A passive book hinge must be
        # grasped and rotated with contact; a stapler can be pressed from above.
        if name == 'folder':
            return self._close_book_by_edge_contact(stage)
        prop = self.objects[name]
        if abs(prop.angle)<.03:
            self.completed_stages.append(stage)
            return True
        arm_name = self.OBJECT_SPECS[name]['arm']
        arm = self._arm_tag(arm_name)
        tip = prop.lid_tip(prop.angle)
        if not self._run_move(stage+'/open',self.open_gripper(arm)):
            return False
        for height in np.linspace(tip[2]+.05,self._bottom(name)+self.object_geometry[name]['height'],10):
            pose = self._top_down_pose(tip[:2],height,arm_name)
            if not self._run_move(stage+'/press',self.move_to_pose(arm,pose)):
                return self._reject(stage,'Cannot reach the stapler lever')
        self.delay(1)
        if abs(prop.angle) > .05:
            return self._reject(stage,'Stapler did not close under contact')
        self.completed_stages.append(stage)
        return True

    def _plan_edge_corridor_clearance(self, lower, upper):
        """Find short, unobstructed lateral buffer moves before executing any.

        These conservative swept boxes are a geometric filter. Robot IK and
        measured contact still determine whether a proposed move can execute.
        """
        boxes = {}
        for name, actor in self.objects.items():
            pose = actor.get_pose()
            yaw = np.arctan2(pose.to_transformation_matrix()[1,0],pose.to_transformation_matrix()[0,0])
            r = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
            half = np.abs(r) @ (self.object_geometry[name]['footprint']/2)
            boxes[name] = [pose.p[:2].copy(),half]
        pending = [name for name,(center,half) in boxes.items() if name != 'folder'
                   and np.all(center+half>lower) and np.all(center-half<upper)]
        plan=[]
        while pending:
            candidates=[]
            for name in pending:
                center,half=boxes[name]
                for side in (-1,1):
                    target=center.copy()
                    target[0] = lower[0]-half[0]-.025 if side<0 else upper[0]+half[0]+.025
                    if np.any(np.abs(target)+half > np.array([.56,.33])):
                        continue
                    sweep_min=np.minimum(center,target)-half-.006
                    sweep_max=np.maximum(center,target)+half+.006
                    blocked=False
                    for other,(other_center,other_half) in boxes.items():
                        if other==name:
                            continue
                        if other=='folder':
                            # Include both halves of the open book, with yaw.
                            root=self.objects['folder'].get_pose()
                            matrix=root.to_transformation_matrix()[:2,:2]
                            w,d=self.object_geometry['folder']['footprint']
                            other_center=root.p[:2]+matrix@np.array([-w/2,0.])
                            other_half=np.abs(matrix)@np.array([w,d/2])
                        if np.all(other_center+other_half>sweep_min) and np.all(other_center-other_half<sweep_max):
                            blocked=True
                            break
                    if not blocked:
                        candidates.append((float(np.linalg.norm(target-center)),name,target))
            if not candidates:
                return None,pending
            _,name,target=min(candidates,key=lambda item:item[0])
            plan.append((name,target.tolist()))
            boxes[name][0]=target
            pending.remove(name)
        return plan,[]

    def _clear_edge_corridor(self, lower, upper, stage):
        plan,blocked=self._plan_edge_corridor_clearance(lower,upper)
        log=self.physical_grasp_log[stage+'/clearance_plan']={
            'mode':'geometric_buffer_planning','corridor':[lower.tolist(),upper.tolist()],
            'moves':plan,'unresolved':blocked}
        if plan is None:
            if not {'book','stapler','cup','keyboard'}.issubset(self.objects):
                return self._reject(stage,'No unobstructed lateral buffer route for '+', '.join(blocked))
            # A book still partly supported by the open leaf must first slide
            # off it. Move the narrow stapler out of that lateral corridor.
            # These are action waypoints, never changes to initial geometry.
            book_half=self.object_geometry['book']['footprint']/2
            plan=[('cup',[.11,-.25]),('stapler',[-.305,-.285]),('stapler',[-.49,-.285]),
                  ('book',[float(self.objects['book'].get_pose().p[0]),-.155]),
                  ('book',[float(lower[0]-book_half[0]-.025),-.155]),
                  ('keyboard',[.145,-.08])]
            log['moves']=plan
            log['mode']='multi_segment_contact_clearance'
        restore=[]
        original={name:self.objects[name].get_pose().p[:2].copy() for name,_ in plan}
        for index,(name,target) in enumerate(plan):
            if not self._move_buffer_object(name,target,stage+f'/clear_{index}_{name}'):
                return False
            if name not in [entry[0] for entry in restore]:
                restore.append((name,original[name].tolist()))
        self._edge_corridor_restore=restore
        log['executed']=True
        return True

    def _move_buffer_object(self,name,target,stage):
        if name != 'cup':
            return self._push_to_xy(name,target,stage)
        # Keep the proven upright cup grasp rather than pushing a tall vessel.
        spec=self.OBJECT_SPECS[name]
        old_xy,old_yaw=list(spec['target_xy']),spec['target_yaw']
        old_bottom=self.target_bottom_z[name]
        old_stages=list(self.completed_stages)
        try:
            self._set_target(name,{'xy':target,'yaw':0.,'bottom':self._table_z()+self.SURFACE_GAP})
            self.initial_bottom_z[name]=self._bottom(name)
            if not self._pick_and_place(name,stage):
                return False
            return self._run_move(stage+'/home',self.back_to_origin(self._arm_tag(spec['arm'])))
        finally:
            spec['target_xy'],spec['target_yaw']=old_xy,old_yaw
            self.target_bottom_z[name]=old_bottom
            self.completed_stages[:]=old_stages

    def _slide_book_to_front_edge(self, stage):
        """Push the base through actual contact; never translate the prop itself."""
        name = 'folder'
        prop = self.objects[name]
        arm = self._arm_tag('left')
        table_front = float(getattr(self,'table_xy_bias',[0,0])[1])-.35
        depth = self.object_geometry[name]['footprint'][1]
        target_y = table_front + depth/2 - .022
        # The open cover also sweeps space: checking only the closed root box
        # would push it through the already placed small book or buffer cup.
        root=prop.get_pose().p
        width=float(prop.size[0])
        lower=np.array([root[0]-1.5*width,target_y-depth/2])
        upper=np.array([root[0]+width/2,root[1]+depth/2])
        blockers=[]
        for other in self.objects:
            if other==name:
                continue
            center=self.objects[other].get_pose().p[:2]
            half=self.object_geometry[other]['footprint']/2
            if np.all(center+half>lower) and np.all(center-half<upper):
                blockers.append(other)
        if blockers:
            if not self._clear_edge_corridor(lower,upper,stage):
                return False
        return self._push_to_xy('folder',[float(prop.get_pose().p[0]),target_y],stage+'/edge_slide')

    def _close_book_by_edge_contact(self, stage):
        prop = self.objects['folder']
        if abs(prop.angle) < .05:
            self.completed_stages.append(stage)
            return True
        if not self._slide_book_to_front_edge(stage):
            return False
        arm = self._arm_tag('left')
        width,depth,height = prop.size
        root = prop.get_pose()
        start_angle = prop.angle
        # Pinch the open cover at its overhanging FRONT edge. The other cover
        # remains on the table. The fingers rotate with the cover during closure.
        joint = sapien.Pose([-width/2,0,height/2],t3d.quaternions.axangle2quat([0,1,0],-start_angle))
        point = (root*joint*sapien.Pose([width*.8,-depth/2+.002,height/4])).p
        root_rotation=root.to_transformation_matrix()[:3,:3]
        local_rotation=horizontal_book_frame([0,1])
        initial_rotation = root_rotation@local_rotation
        opening = self._jaw_command(height/2+.006,'left')
        approach = self._tool_pose(point-initial_rotation[:,0]*.05,initial_rotation)
        high = np.asarray(approach).copy(); high[2]+=.12
        for suffix, action in [('open',self.open_gripper(arm,pos=opening)),
                               ('above_cover_edge',self.move_to_pose(arm,high.tolist())),
                               ('approach_cover_edge',self.move_to_pose(arm,approach)),
                               ('insert_cover_edge',self.move_to_pose(arm,self._tool_pose(point,initial_rotation))),
                               ('pinch_cover',self.close_gripper(arm,pos=0.))]:
            if not self._run_move(stage+'/'+suffix,action):
                return self._reject(stage,'Cannot grasp the open cover at its edge')
        if not self._held('folder','left'):
            return self._reject(stage,'Both fingers did not contact the cover')
        for i, angle in enumerate(np.linspace(start_angle,0.,25)[1:]):
            joint = sapien.Pose([-width/2,0,height/2],t3d.quaternions.axangle2quat([0,1,0],-angle))
            point = (root*joint*sapien.Pose([width*.8,-depth/2+.002,height/4])).p
            rotation = root_rotation@t3d.axangles.axangle2mat([0,1,0],start_angle-angle)@local_rotation
            if not self._run_move(stage+f'/rotate_cover_{i}',self.move_to_pose(arm,self._tool_pose(point,rotation))):
                return self._reject(stage,'Cover rotation trajectory infeasible')
            if np.linalg.norm(prop.get_pose().p[:2]-root.p[:2]) > .025:
                return self._reject(stage,'Book base moved during closure; needs additional stabilization')
            if not self._held('folder','left'):
                return self._reject(stage,'Lost cover contact while closing the passive hinge')
        if not self._run_move(stage+'/release_cover',self.open_gripper(arm,pos=opening)):
            return False
        retreat = np.asarray(self._current_ee_pose('left')); retreat[1]-=.07
        if not self._run_move(stage+'/withdraw_cover',self.move_to_pose(arm,retreat.tolist())):
            return False
        self.delay(1)
        if abs(prop.angle) > .05:
            return self._reject(stage,'Physical hinge did not reach the closed state')
        if not self._run_move(stage+'/cover_home',self.back_to_origin(arm)):
            return False
        for index,(name,target) in enumerate(reversed(getattr(self,'_edge_corridor_restore',[]))):
            if not self._move_buffer_object(name,target,stage+f'/restore_{index}_{name}'):
                return False
        self.completed_stages.append(stage)
        return True
