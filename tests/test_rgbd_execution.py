import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from script.rgbd_execution import DepthGeometry, ContactVelocityFollow, RGBDAdapter
from script.dynamic_sim import DynamicSimError
from high_level_planner.perception.base import ObjectDetection, PerceptionResult
from high_level_planner.perception.mask_codec import encode_binary_mask


class RGBDTests(unittest.TestCase):
    def geometry(self, table=.61):
        depth=np.full((80,100),1000.)
        depth[30:40,30:45]=960.
        depth[60:75,5:20]=1200.
        camera=np.diag([1.,-1.,-1.,1.]);camera[2,3]=table+1.
        return DepthGeometry(depth,[[100,0,50],[0,100,40],[0,0,1]],np.linalg.inv(camera))

    def detection(self, bbox):
        mask=np.zeros((80,100),bool)
        x1,y1,x2,y2=bbox;mask[y1:y2,x1:x2]=True
        return ObjectDetection('item',.9,bbox,mask=encode_binary_mask(mask))

    def test_unknown_table_height_and_backprojection(self):
        for table in (.42,.79,1.15):
            g=self.geometry(table)
            self.assertAlmostEqual(g.table_z,table)
            np.testing.assert_allclose(g.ray_plane([.5,.5],table),[0,0,table])
            self.assertAlmostEqual(g.surface(self.detection([30,30,45,40]))['top'],table+.04)

    def test_background_filter_keeps_tabletop_object(self):
        g=self.geometry()
        foreground=self.detection([30,30,45,40]); background=self.detection([5,60,20,75])
        result=g.filter(PerceptionResult('test',[foreground,background],[100,80]))
        self.assertEqual([d.bbox for d in result.detections],[foreground.bbox])
        self.assertEqual(result.metadata['rgbd_workspace_filter']['rejected'][0]['reason'],'below_observed_table')

    def test_follow_only_when_attached_and_does_not_read_or_set_pose(self):
        task=SimpleNamespace(scene=SimpleNamespace(get_contacts=lambda:[],get_timestep=lambda:.01))
        tip=Mock(return_value=(np.array([.1,0,.8]),np.eye(3)))
        bridge=ContactVelocityFollow(task,tip)
        bridge.sample();tip.assert_not_called()
        body=Mock();body.get_pose.side_effect=AssertionError('truth read');body.set_pose.side_effect=AssertionError('teleport')
        bridge.body=body;bridge.arm='left';bridge.previous=(np.array([.09,0,.8]),np.pi/2)
        bridge.sample()
        np.testing.assert_allclose(body.set_linear_velocity.call_args.args[0],[1,0,0],atol=1e-9)
        bridge.release();self.assertIsNone(bridge.body)
        body.get_pose.assert_not_called();body.set_pose.assert_not_called()
        bridge.sample();self.assertEqual(tip.call_count,1)

    def test_no_contact_cannot_start_follow(self):
        task=SimpleNamespace(robot=SimpleNamespace(left_entity=SimpleNamespace(get_links=lambda:[]),right_entity=SimpleNamespace(get_links=lambda:[])))
        bridge=ContactVelocityFollow(task,lambda arm:(np.zeros(3),np.eye(3)))
        with self.assertRaisesRegex(DynamicSimError,'No fingertip contact'):
            bridge.attach('left')
        self.assertIsNone(bridge.body)

    def test_static_contact_is_rejected_without_live_component_replacement(self):
        import sapien
        bridge=ContactVelocityFollow(SimpleNamespace(),lambda arm:(np.zeros(3),np.eye(3)))
        bridge.contact_body=Mock(return_value=sapien.physx.PhysxRigidStaticComponent())
        with self.assertRaisesRegex(DynamicSimError,'initialize symbolic grasp assets'):
            bridge.attach('left')
        self.assertIsNone(bridge.body)

    def test_nearer_table_contact_does_not_hide_contacted_movable_body(self):
        import sapien
        finger=sapien.Entity();table=sapien.Entity();prop=sapien.Entity()
        static=sapien.physx.PhysxRigidStaticComponent();table.add_component(static)
        dynamic=sapien.physx.PhysxRigidDynamicComponent();prop.add_component(dynamic)
        link=SimpleNamespace(entity=finger)
        robot=SimpleNamespace(left_entity=SimpleNamespace(get_links=lambda:[link]),
                              right_entity=SimpleNamespace(get_links=lambda:[]))
        bridge=ContactVelocityFollow(SimpleNamespace(robot=robot),lambda arm:(np.zeros(3),np.eye(3)))
        bridge.contacts=[SimpleNamespace(bodies=[link,body],points=[SimpleNamespace(position=[distance,0,0])])
                         for body,distance in [(static,.001),(dynamic,.005)]]
        self.assertIs(bridge.contact_body('left'),dynamic)
        bridge.contacts=bridge.contacts[:1]
        with self.assertRaisesRegex(DynamicSimError,'No fingertip contact'):
            bridge.contact_body('left')

    def test_wrist_yaw_uses_final_joint_encoder_not_joint_anchor(self):
        from script.rgbd_execution import RGBDGraspExecutor
        controller=RGBDGraspExecutor.__new__(RGBDGraspExecutor)
        entity=Mock(); entity.get_qpos.return_value=np.array([0.,0.,0.,0.,0.,.7])
        controller.models={'left':(entity,)}
        controller.task=SimpleNamespace(_current_ee_pose=Mock(side_effect=AssertionError('joint anchor omits wrist rotation')))
        controller.fk=Mock(return_value=(np.ones(3),np.eye(3)))
        controller.tip('left')
        controller.fk.assert_called_once()
        self.assertAlmostEqual(controller.fk.call_args.args[1][-1],.7)

    def test_visual_adapter_refuses_truth_evaluation(self):
        with self.assertRaisesRegex(DynamicSimError,'simulator success is forbidden'):
            RGBDAdapter.evaluate(SimpleNamespace())


if __name__=='__main__': unittest.main()

class VisualRotationIdentityTests(unittest.TestCase):
    def test_unobservable_axis_does_not_leave_an_unverifiable_yaw_goal(self):
        from script.online_rearrangement import refine_visual_rotations
        from high_level_planner.scene_graph.graph import SceneGraph, SceneObject
        from high_level_planner.scene_graph.attributes import ObjectAttributes
        obj=SceneObject('round_item','cup',attributes=ObjectAttributes(
            xy=[.5,.5],bbox=[40,30,60,70],yaw_degrees=0.))
        obj.metadata['planning_pose']={'frame':'image_normalized','xy':[.5,.5],'yaw_degrees':0.}
        initial=SceneGraph(objects=[obj]);target=initial.copy()
        detection=ObjectDetection('cup',.9,[40,30,60,70],
            metadata={'rgbd_yaw_axis_observable':False})
        perception=Mock();perception.detect.return_value=PerceptionResult('image',[detection],[100,100])
        refine_visual_rotations(initial,target,'initial','target',perception)
        self.assertIsNone(target.objects[0].attributes.yaw_degrees)
        self.assertNotIn('yaw_degrees',target.objects[0].metadata['planning_pose'])
        self.assertEqual(initial.objects[0].attributes.yaw_degrees,0.)

    def test_raw_sam_label_drift_does_not_lose_reviewed_folder_mask(self):
        from script.online_rearrangement import _assign_detections
        from high_level_planner.scene_graph.graph import SceneObject
        from high_level_planner.scene_graph.attributes import ObjectAttributes
        folder=SceneObject('folder_id','folder',attributes=ObjectAttributes(xy=[.3,.4],bbox=[10,10,50,70]))
        book=SceneObject('book_id','book',attributes=ObjectAttributes(xy=[.6,.5],bbox=[40,30,80,70]))
        a=ObjectDetection('book',.9,[10,10,50,70]);b=ObjectDetection('book',.9,[40,30,80,70])
        matches=_assign_detections([folder,book],[b,a],[100,100])
        self.assertIs(matches['folder_id'],a)
        self.assertIs(matches['book_id'],b)

    def test_tall_round_shape_does_not_supply_planar_yaw(self):
        from script.online_rearrangement import _mask_axis
        d=ObjectDetection('item',.9,[0,0,20,40],metadata={'rgbd_yaw_axis_observable':False})
        self.assertIsNone(_mask_axis(d,[100,100]))

class ObservedObstacleTests(unittest.TestCase):
    def test_occupied_destination_adapts_to_observed_supports(self):
        from script.rgbd_execution import destination_blockers
        for offset in (np.zeros(2),np.array([1.2,-.5])):
            surfaces={'cover':{'top':.8,'extent':np.array([[-.1,-.1],[.1,.1]])+offset}}
            self.assertEqual(destination_blockers(surfaces,'payload','table',offset,.74),['cover'])
            self.assertEqual(destination_blockers(surfaces,'payload','cover',offset,.8),[])
            self.assertEqual(destination_blockers(surfaces,'payload','table',offset+[.2,0],.74),[])

    def test_insertion_adapts_to_translated_observed_wall(self):
        x,z=np.meshgrid(np.linspace(-.06,.06,50),np.linspace(.83,.98,50))
        points=np.stack([x,np.full_like(x,.04),z],-1)
        first=None
        for offset in (np.zeros(3),np.array([1.3,-.4,.2])):
            g=DepthGeometry.__new__(DepthGeometry)
            g.points=points+offset;g.valid=np.ones(x.shape,bool)
            release=np.array([0.,0.,.8])+offset
            stage,axis,_=g.insertion_approach(release,.147)
            self.assertLess(stage[1],release[1])
            self.assertGreater(axis[1],0.)
            if first is not None:
                np.testing.assert_allclose(stage-offset,first)
            first=stage-offset

    def test_empty_observed_space_does_not_request_wall_insertion(self):
        g=DepthGeometry.__new__(DepthGeometry)
        g.points=np.zeros((10,10,3));g.valid=np.ones((10,10),bool)
        self.assertIsNone(g.insertion_approach(np.array([0.,0.,.8]),.147))

    def test_wall_edge_does_not_skew_insertion_direction(self):
        # The goal lies beyond the observed plane edge. The plane normal,
        # rather than the bearing to that edge, defines wrist clearance.
        x,z=np.meshgrid(np.linspace(-.06,-.02,40),np.linspace(.83,.98,40))
        g=DepthGeometry.__new__(DepthGeometry)
        g.points=np.stack([x,np.full_like(x,.04),z],-1)
        g.valid=np.ones(x.shape,bool)
        _,_,direction=g.insertion_approach(np.array([.02,0.,.8]),.147)
        np.testing.assert_allclose(direction,[0.,1.],atol=1e-6)


class RGBDSceneGraphTests(unittest.TestCase):
    def test_depth_support_replaces_containment_and_uses_camera_occlusion(self):
        from pathlib import Path
        from script.rgbd_execution import RGBDSceneGraphConstructor
        from high_level_planner.scene_graph.graph import SceneGraph, SceneObject, Relation
        from high_level_planner.scene_graph.attributes import ObjectAttributes
        g=RGBDTests().geometry()
        masks=[]
        for box in ([20,20,60,60],[30,30,45,40],[65,20,85,60]):
            mask=np.zeros((80,100),bool);x1,y1,x2,y2=box;mask[y1:y2,x1:x2]=True;masks.append(mask)
        masks[0] &= ~masks[1]
        g.points[masks[0],2]=.63
        g.points[masks[1],2]=.65
        # A high screen has the same tabletop footprint as the book but
        # occupies different camera pixels: it cannot obscure that book.
        g.points[masks[2],:2]=np.median(g.points[masks[1],:2],axis=0)
        g.points[masks[2],2]=.95
        specs=[('support',[20,20,60,60]),('payload',[30,30,45,40]),('screen',[65,20,85,60])]
        objects=[SceneObject(key,key,attributes=ObjectAttributes(bbox=box,movable=key!='screen')) for key,box in specs]
        graph=SceneGraph(objects=objects,relations=[Relation('payload','support','inside')])
        detections=[ObjectDetection(key,.9,box,mask=encode_binary_mask(mask)) for (key,box),mask in zip(specs,masks)]
        backend=Mock();backend.detect.return_value=PerceptionResult('frame',detections,[100,80])
        path=Path('/tmp/rgbd_support_unit.png')
        constructor=RGBDSceneGraphConstructor(sensor_packets=lambda:{str(path):g},perception_backend=backend)
        with patch('high_level_planner.scene_graph.constructor.SceneGraphConstructor.construct',return_value=graph):
            observed=constructor.construct(path)
        relations={(r.source,r.relation,r.target) for r in observed.relations}
        self.assertIn(('payload','on','support'),relations)
        self.assertNotIn(('payload','inside','support'),relations)
        self.assertEqual(observed.get_object('payload').attributes.visibility,'fully_visible')
        self.assertEqual(observed.get_object('support').attributes.visibility,'partially_visible')
