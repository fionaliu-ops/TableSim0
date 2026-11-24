from types import SimpleNamespace
from dataclasses import replace
import unittest

import numpy as np

from high_level_planner.scene_graph.attributes import ObjectAttributes
from high_level_planner.scene_graph.graph import SceneGraph, SceneObject
from script.dynamic_sim import DynamicSimError
from script.online_rearrangement import OnlineRearrangementPlanner


class VisualRotationReferenceTest(unittest.TestCase):
    def test_pixel_noise_axis_swap_uses_visual_prior_without_changing_symmetry(self):
        from unittest.mock import Mock, patch
        from high_level_planner.perception.base import ObjectDetection, PerceptionResult
        from script.online_rearrangement import refine_visual_rotations
        obj=SceneObject('item','folder',attributes=ObjectAttributes(xy=[.5,.5],bbox=[20,20,80,80]))
        initial=SceneGraph(objects=[obj]);target=initial.copy()
        target.objects[0].attributes.yaw_degrees=-15.
        backend=Mock();backend.detect.return_value=PerceptionResult('test',[
            ObjectDetection('folder',.9,[20,20,80,80])],[100,100])
        with patch('script.online_rearrangement._mask_axis',side_effect=[(99.,1.3),(0.,1.004)]):
            refine_visual_rotations(initial,target,'initial.png','goal.png',backend)
        self.assertEqual(target.objects[0].attributes.yaw_degrees,-9.)
        self.assertEqual(target.metadata['visual_rotation_refinement']['records'][0]['axis_period_degrees'],180.)

    def test_nearly_square_target_projection_does_not_erase_quarter_turn(self):
        from unittest.mock import Mock, patch
        from high_level_planner.perception.base import ObjectDetection, PerceptionResult
        from script.online_rearrangement import refine_visual_rotations
        obj=SceneObject('item','book',attributes=ObjectAttributes(xy=[.5,.5],bbox=[20,20,80,80]))
        initial=SceneGraph(objects=[obj]);target=initial.copy()
        detection=ObjectDetection('book',.9,[20,20,80,80])
        backend=Mock();backend.detect.return_value=PerceptionResult('test',[detection],[100,100])
        with patch('script.online_rearrangement._mask_axis',side_effect=[(0.,2.),(90.,1.08)]):
            refine_visual_rotations(initial,target,'initial.png','goal.png',backend)
        self.assertEqual(abs(target.objects[0].attributes.yaw_degrees),90.)
        self.assertEqual(target.metadata['visual_rotation_refinement']['records'][0]['axis_period_degrees'],180.)


class Pose:
    def __init__(self, xy):
        self.p = np.array([xy[0], xy[1], .74], dtype=float)
        self.q = np.array([1., 0., 0., 0.])

    def to_transformation_matrix(self):
        matrix = np.eye(4)
        matrix[:3, 3] = self.p
        return matrix


class Actor:
    def __init__(self, xy):
        self.pose = Pose(xy)

    def get_pose(self):
        return self.pose


class FakeEnvironment:
    def __init__(self, initial):
        self.task = SimpleNamespace(
            objects={name: Actor(xy) for name, xy in initial.items()},
            object_geometry={name: {"footprint": np.array([.075, .06]), "height": .025}
                             for name in initial},
        )
        self.object_map={f"{name}_id": name for name in initial}
        self.transform=np.array([[.6, 0.], [0., .5], [-.3, -.25]])
        self.held=None

    @staticmethod
    def _object_yaw(_name):
        return 0.0

    @staticmethod
    def _arm(name):
        return "left" if name < "n" else "right"

    @staticmethod
    def _rectangles_overlap(a_xy, a_size, _a_yaw, b_xy, b_size, _b_yaw, margin):
        return bool(abs(float(a_xy[0]-b_xy[0])) < (a_size[0]+b_size[0])/2+margin
                    and abs(float(a_xy[1]-b_xy[1])) < (a_size[1]+b_size[1])/2+margin)

    def _clear_table_placement(self, name, xy, support=None, margin=.006):
        if not (-.34 <= xy[0] <= .34 and -.24 <= xy[1] <= .24):
            return False
        own=self.task.object_geometry[name]["footprint"]
        return not any(other != name and other != support and self._rectangles_overlap(
            xy, own, 0., actor.get_pose().p[:2],
            self.task.object_geometry[other]["footprint"], 0., margin)
            for other,actor in self.task.objects.items())

    def _placement_blockers(self, name, xy):
        own=self.task.object_geometry[name]["footprint"]
        return [other for other,actor in self.task.objects.items() if other != name
                and self._rectangles_overlap(xy,own,0.,actor.get_pose().p[:2],
                    self.task.object_geometry[other]["footprint"],0.,.006)]


def target_graph(targets):
    objects=[]
    for name,xy in targets.items():
        attrs=ObjectAttributes(xy=list(xy),yaw_degrees=0.0,movable=True,portable=True)
        objects.append(SceneObject(f"{name}_id",name,attributes=attrs,
            metadata={"planning_pose":{"frame":"image_normalized","xy":list(xy),
                                        "support":"table","yaw_degrees":0.0}}))
    return SceneGraph(objects=objects)


class OnlineRearrangementTest(unittest.TestCase):
    def _run_cycle(self, names):
        targets={name:(.30+.20*i,.50) for i,name in enumerate(names)}
        transform=np.array([[.6,0.],[0.,.5],[-.3,-.25]])
        world={name:np.r_[targets[name],1.]@transform for name in names}
        initial={name:world[names[(i+1)%len(names)]] for i,name in enumerate(names)}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        decisions=[]
        for _ in range(10):
            if planner.goal.evaluate()["passed"]:
                break
            action,decision=planner.next_action(); decisions.append(decision)
            name=env.object_map[action["source_object_id"]]
            xy=np.r_[action["pose"]["xy"],1.]@env.transform
            env.task.objects[name].pose.p[:2]=xy
        self.assertTrue(planner.goal.evaluate()["passed"])
        self.assertEqual(decisions[0]["kind"],"buffer")
        self.assertLessEqual(len(decisions),2*len(names))
        return [item["object"] for item in decisions]

    def test_cyclic_occupancy_is_solved_from_live_geometry(self):
        first=self._run_cycle(["alpha","beta","gamma"])
        second=self._run_cycle(["gamma","alpha","beta"])
        self.assertLessEqual(len(first),6)
        self.assertLessEqual(len(second),6)

    def test_buffer_is_sampled_and_not_a_reserved_target(self):
        names=["one","two","three"]
        targets={name:(.30+.20*i,.50) for i,name in enumerate(names)}
        transform=np.array([[.6,0.],[0.,.5],[-.3,-.25]])
        initial={name:np.r_[targets[names[(i+1)%3]],1.]@transform for i,name in enumerate(names)}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        action,decision=planner.next_action()
        self.assertEqual(decision["kind"],"buffer")
        self.assertNotIn(tuple(action["pose"]["xy"]),set(targets.values()))

    def test_dense_fallback_finds_gap_between_coarse_samples(self):
        targets={"alpha":(.70,.70),"beta":(.80,.80)}
        initial={"alpha":np.array([-.1,0.]),"beta":np.array([.25,.15])}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        expected=np.r_[[3/32,3/32],1.]@env.transform
        env._clear_table_placement=lambda name,xy,support=None,margin=.006: bool(
            np.linalg.norm(np.asarray(xy)-expected)<1e-9)
        planner._buffer_neighbour_clear=lambda name,xy:True
        candidates=planner._buffer_candidates("alpha")
        self.assertEqual(len(candidates),1)
        np.testing.assert_allclose(candidates[0][1],[3/32,3/32])

    def test_buffer_reserves_regrasp_clearance_from_neighbours(self):
        targets={"alpha":(.30,.50),"beta":(.70,.50)}
        env=FakeEnvironment({"alpha":np.array([-.1,0.]),"beta":np.array([.1,0.])})
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        self.assertFalse(planner._buffer_neighbour_clear("alpha",np.array([.01,0.])))
        self.assertTrue(planner._buffer_neighbour_clear("alpha",np.array([-.2,.2])))

    def test_buffer_attempts_are_scoped_to_arm(self):
        targets={"alpha":(.70,.70),"beta":(.80,.80)}
        env=FakeEnvironment({"alpha":np.array([-.1,0.]),"beta":np.array([.25,.15])})
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        image=np.array([.5,.5]); world=np.r_[image,1.]@env.transform
        planner._buffer_candidates=lambda name:[(0.,image,world)]
        planner.cycle_arm_overrides["alpha"]="left"
        planner._buffer_action("alpha")
        planner.cycle_arm_overrides["alpha"]="right"
        planner._buffer_action("alpha")
        self.assertEqual(planner.used_buffers,
                         {("alpha","0.500,0.500","left"),
                          ("alpha","0.500,0.500","right")})

    def test_ready_object_blocking_another_goal_moves_directly_to_its_goal(self):
        targets={"alpha":(.30,.50),"beta":(.65,.50)}
        transform=np.array([[.6,0.],[0.,.5],[-.3,-.25]])
        world={name:np.r_[xy,1.]@transform for name,xy in targets.items()}
        initial={"alpha":world["alpha"].copy()+[0.,-.10],
                 "beta":world["alpha"].copy()+[.01,0.]}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        action,decision=planner.next_action()
        self.assertEqual(decision,{"kind":"final","object":"beta","dependency_pressure":1})
        self.assertEqual(action["pose"]["xy"],list(targets["beta"]))

    def test_successful_geometric_buffer_is_held_until_source_moves(self):
        targets={"alpha":(.30,.50),"beta":(.70,.50)}
        initial={"alpha":np.array([-.12,-.08]),"beta":np.array([.12,-.08])}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        blocking={"alpha":["beta"],"beta":[]}
        planner._blocking_objects=lambda name:list(blocking[name])
        planner._above_blockers=lambda name:[]
        planner._access_blockers=lambda name:[]
        planner._final_ready=lambda name:False
        action,decision=planner.next_action()
        self.assertEqual(decision["object"],"beta")
        planner.record_success(action)
        self.assertEqual(planner.release_holds,{"alpha":{"beta"}})
        blocking["alpha"]=[]
        planner._final_ready=lambda name:True
        _,decision=planner.next_action()
        self.assertEqual(decision["object"],"alpha")
        planner.record_success(planner._action("alpha",final=True))
        self.assertEqual(planner.release_holds,{})

    def test_runtime_contact_becomes_live_dependency_until_blocker_moves(self):
        targets={"alpha":(.30,.50),"beta":(.65,.50)}
        transform=np.array([[.6,0.],[0.,.5],[-.3,-.25]])
        initial={name:np.r_[xy,1.]@transform for name,xy in targets.items()}
        initial["alpha"] = initial["alpha"] + np.array([0., -.08])
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        action=planner._action("alpha",final=True)
        planner.record_failure(action,"Robot touched non-target objects: ['beta']")
        self.assertEqual(planner._access_blockers("alpha"),["beta"])
        origin=planner.access_dependencies["alpha"]["beta"].copy()
        env.task.objects["beta"].pose.p[0] += .02
        planner.record_failure(action,"Robot touched non-target objects: ['beta']")
        np.testing.assert_allclose(planner.access_dependencies["alpha"]["beta"],origin)
        self.assertEqual(len(planner.access_collision_regions["alpha"]["beta"]),1)
        _,decision=planner.next_action()
        self.assertEqual(decision["object"],"beta")
        env.task.objects["beta"].pose.p[0] += .06
        self.assertEqual(planner._access_blockers("alpha"),[])
        _,decision=planner.next_action()
        self.assertEqual(decision["object"],"alpha")

    def test_new_separated_collision_pose_becomes_an_additional_region(self):
        targets={"alpha":(.30,.50),"beta":(.65,.50)}
        transform=np.array([[.6,0.],[0.,.5],[-.3,-.25]])
        initial={name:np.r_[xy,1.]@transform for name,xy in targets.items()}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        action=planner._action("alpha",final=True)
        planner.record_failure(action,"Robot touched non-target objects during grasp: ['beta']")
        first=env.task.objects["beta"].pose.p[:2].copy()
        env.task.objects["beta"].pose.p[0] += .08
        planner.record_failure(action,"Robot touched non-target objects during grasp: ['beta']")
        regions=planner.access_collision_regions["alpha"]["beta"]
        self.assertEqual(len(regions),2)
        np.testing.assert_allclose(regions[0],first)
        self.assertEqual(planner._access_blockers("alpha"),["beta"])
        self.assertFalse(planner._access_origin_clear("beta",regions[0]))
        self.assertFalse(planner._access_origin_clear("beta",regions[1]))

    def test_object_name_is_not_inferred_from_part_of_an_error_word(self):
        targets={"cup":(.30,.50),"keyboard":(.65,.50)}
        transform=np.array([[.6,0.],[0.,.5],[-.3,-.25]])
        initial={name:np.r_[xy,1.]@transform for name,xy in targets.items()}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        action=planner._action("keyboard",final=False,image_xy=(.5,.25))
        planner.record_failure(action,
            "No collision-free IK placement; occupied candidates were rejected")
        self.assertNotIn("cup",planner.access_dependencies.get("keyboard",{}))

    def test_structured_blocker_excludes_incidental_audit_names(self):
        targets={"book":(.25,.50),"cup":(.45,.50),"keyboard":(.70,.50)}
        transform=np.array([[.6,0.],[0.,.5],[-.3,-.25]])
        initial={name:np.r_[xy,1.]@transform for name,xy in targets.items()}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        action=planner._action("keyboard",final=False,image_xy=(.5,.25))
        error=RuntimeError("No live-IK top grasp; candidate near book failed; trajectory intersects cup")
        error.blockers=("cup",)
        planner.record_failure(action,error)
        self.assertEqual(planner._access_blockers("keyboard"),["cup"])
        self.assertNotIn("left",planner.failed_arms.get("keyboard",set()))

    def test_empty_structured_blockers_do_not_scrape_candidate_audit(self):
        targets={"book":(.25,.50),"keyboard":(.70,.50)}
        transform=np.array([[.6,0.],[0.,.5],[-.3,-.25]])
        initial={name:np.r_[xy,1.]@transform for name,xy in targets.items()}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        action=planner._action("keyboard",final=False,image_xy=(.5,.25))
        error=DynamicSimError(
            "No live-geometry side grasp for keyboard; nearest audit entry was book",
            blockers=())
        planner.record_failure(action,error)
        self.assertEqual(planner.access_dependencies.get("keyboard",{}),{})
        self.assertIn("left",planner.failed_arms.get("keyboard",set()))

    def test_observed_grasp_corridor_rejects_buffer_motion_along_corridor(self):
        targets={"alpha":(.30,.50),"beta":(.65,.50)}
        initial={"alpha":np.array([-.1,0.]),"beta":np.array([.05,0.])}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        action=planner._action("alpha",final=True)
        error=DynamicSimError(
            "Measured side-grasp insertion corridor intersects beta",
            blockers=("beta",),
            blocker_corridors={"beta": [([-.1,0.],[.25,0.])]})
        planner.record_failure(action,error)

        # Moving far along the insertion axis remains obstructive even though
        # it leaves the original radial collision basin.
        env.task.objects["beta"].pose.p[:2]=np.array([.22,0.])
        self.assertEqual(planner._access_blockers("alpha"),["beta"])
        self.assertFalse(planner._access_origin_clear("beta",[.22,0.]))
        self.assertTrue(planner._access_origin_clear("beta",[.22,.25]))
        self.assertTrue(planner.diagnostics()["access_corridors"]["alpha"]["beta"])

    def test_source_arm_handoff_survives_blocker_relocation(self):
        targets={"alpha":(.30,.50),"beta":(.65,.50)}
        transform=np.array([[.6,0.],[0.,.5],[-.3,-.25]])
        initial={name:np.r_[xy,1.]@transform for name,xy in targets.items()}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        action=planner._action("alpha",final=False,image_xy=(.5,.25))
        planner.record_failure(
            action,"No live-geometry side grasp for alpha; blocker: beta")
        self.assertEqual(planner._action_arm("alpha"),"right")

        # Moving the blocker out of its recorded collision basin must not
        # discard the fact that the natural arm has already been attempted.
        env.task.objects["beta"].pose.p[:2] += np.array([.25,0.])
        self.assertEqual(planner._access_blockers("alpha"),[])
        self.assertEqual(planner._action_arm("alpha"),"right")

    def test_two_failed_final_arm_routes_trigger_dynamic_staging(self):
        targets={"alpha":(.70,.70)}
        env=FakeEnvironment({"alpha":np.array([-.2,-.15])})
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        planner.failed_arms["alpha"]={"left","right"}

        action,decision=planner.next_action()
        self.assertEqual(decision["kind"],"buffer")
        self.assertNotEqual(action["pose"]["xy"],list(targets["alpha"]))

    def test_failed_placement_basin_drives_nonlocal_exploration(self):
        targets={"alpha":(.70,.70)}
        env=FakeEnvironment({"alpha":np.array([-.2,-.15])})
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        initial=planner._buffer_candidates("alpha")[0][2]
        planner.failed_placement_regions[("alpha","left")]=[initial.copy()]

        diversified=planner._buffer_candidates("alpha")[0][2]
        self.assertGreater(np.linalg.norm(diversified-initial),.20)

    def test_successful_buffer_remains_consumed_until_object_reaches_goal(self):
        targets={"alpha":(.70,.70)}
        env=FakeEnvironment({"alpha":np.array([-.2,-.15])})
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        action=planner._buffer_action("alpha")
        key=",".join(f"{float(v):.3f}" for v in action["pose"]["xy"])
        arm=action["pose"]["arm"]
        self.assertIn(("alpha",key,arm),planner.used_buffers)

        planner.record_success(action)
        self.assertIn(("alpha",key,arm),planner.used_buffers)

        planner.record_success(planner._action("alpha",final=True))
        self.assertNotIn(("alpha",key,arm),planner.used_buffers)

    def test_upstream_success_releases_its_buffer_blocker_nodes(self):
        targets={"alpha":(.70,.70),"beta":(.30,.30)}
        env=FakeEnvironment({"alpha":np.array([-.2,-.15]),
                             "beta":np.array([.2,.15])})
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        entry=("beta","0.250,0.250","right")
        planner.used_buffers.add(entry)
        planner.release_holds["alpha"]={"beta"}

        planner.record_success(planner._action("alpha",final=True))

        self.assertNotIn(entry,planner.used_buffers)

    def test_completed_stack_is_not_dismantled_for_predicted_overlap(self):
        targets={"book":(.25,.25),"keyboard":(.55,.45),"mouse":(.75,.45)}
        transform=np.array([[.6,0.],[0.,.5],[-.3,-.25]])
        initial={name:np.r_[xy,1.]@transform for name,xy in targets.items()}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        planner.goal.layout["book"]=replace(
            planner.goal.layout["book"],support_id="mouse_id",
            support_name="mouse")
        planner.goal.object_passed=lambda name: name != "keyboard"
        planner._blocking_objects=lambda name: (
            ["book","mouse"] if name == "keyboard" else [])
        planner._access_blockers=lambda name: (
            ["mouse"] if name == "keyboard" else [])

        _,decision=planner.next_action()

        self.assertEqual(decision["kind"],"buffer")
        self.assertEqual(decision["object"],"mouse")

    def test_failed_ik_region_skips_nearby_buffer_samples(self):
        targets={"alpha":(.30,.50),"beta":(.65,.50)}
        transform=np.array([[.6,0.],[0.,.5],[-.3,-.25]])
        initial={name:np.r_[xy,1.]@transform for name,xy in targets.items()}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        action=planner._action("alpha",final=False,image_xy=(.5,.25))
        planner.record_failure(action,
            "No collision-free IK placement within requested target")
        failed=np.r_[[.5,.25],1.]@env.transform
        self.assertTrue(planner._placement_region_failed("alpha","left",failed))
        self.assertFalse(planner._placement_region_failed(
            "alpha","left",failed+np.array([.2,0.])))
        second=planner._action("alpha",final=False,image_xy=(.8,.25))
        planner.record_failure(second,
            "No collision-free IK placement within requested target")
        self.assertEqual(planner._action_arm("alpha"),"right")

    def test_movable_blocker_does_not_poison_placement_region(self):
        targets={"alpha":(.30,.50),"beta":(.65,.50)}
        initial={name:np.r_[xy,1.]@np.array([[.6,0.],[0.,.5],[-.3,-.25]])
                 for name,xy in targets.items()}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        action=planner._action("alpha",final=False,image_xy=(.5,.25))
        planner.record_failure(action,
            "No collision-free IK placement; trajectory intersects beta")
        self.assertEqual(planner._access_blockers("alpha"),[])
        self.assertTrue(planner.failed_placement_regions)
        self.assertTrue(planner.failed_destinations)

    def test_temporary_transfer_collision_rejects_region_not_grasp_arm(self):
        targets={"alpha":(.30,.50),"beta":(.65,.50)}
        initial={name:np.r_[xy,1.]@np.array([[.6,0.],[0.,.5],[-.3,-.25]])
                 for name,xy in targets.items()}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        action=planner._action("alpha",final=False,image_xy=(.5,.25))
        planner.record_failure(
            action,"Robot or carried alpha touched other objects during transfer_route_0: ['beta']")
        self.assertNotIn("beta",planner.access_dependencies.get("alpha",{}))
        self.assertNotIn("left",planner.failed_arms.get("alpha",set()))
        failed=np.r_[[.5,.25],1.]@env.transform
        self.assertTrue(planner._placement_region_failed("alpha","left",failed))

    def test_held_object_correction_failure_does_not_reject_grasp_arm(self):
        targets={"alpha":(.30,.50),"beta":(.65,.50)}
        initial={name:np.r_[xy,1.]@np.array([[.6,0.],[0.,.5],[-.3,-.25]])
                 for name,xy in targets.items()}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        action=planner._action("alpha",final=True)
        planner.record_failure(action,
            "No strictly separated correction for held alpha: trajectory optimisation failed")
        self.assertNotIn("left",planner.failed_arms.get("alpha",set()))
        self.assertNotIn(("alpha","0.300,0.500","left"),planner.failed_destinations)

    def test_repeated_intrinsic_final_route_failure_switches_arm(self):
        targets={"alpha":(.30,.50),"beta":(.65,.50)}
        initial={name:np.r_[xy,1.]@np.array([[.6,0.],[0.,.5],[-.3,-.25]])
                 for name,xy in targets.items()}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        action=planner._action("alpha",final=True)
        error="Robot or carried alpha touched other objects during transfer: ['monitor']"
        planner.record_failure(action,error)
        self.assertEqual(planner._action_arm("alpha"),"left")
        planner.record_failure(action,error)
        self.assertEqual(planner._action_arm("alpha"),"right")

    def test_grasp_collision_creates_access_edge_and_records_failed_arm(self):
        targets={"alpha":(.30,.50),"beta":(.65,.50)}
        initial={name:np.r_[xy,1.]@np.array([[.6,0.],[0.,.5],[-.3,-.25]])
                 for name,xy in targets.items()}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        action=planner._action("alpha",final=False,image_xy=(.5,.25))
        planner.record_failure(action,"Robot touched non-target objects during grasp: ['beta']")
        self.assertEqual(planner._access_blockers("alpha"),["beta"])
        self.assertEqual(planner.access_failure_arms["alpha"]["beta"],{"left"})
        self.assertEqual(planner.failed_placement_regions,{})

    def test_blocker_chain_moves_leaf_before_blocked_parent(self):
        targets={"alpha":(.25,.50),"beta":(.50,.50),"gamma":(.75,.50)}
        initial={name:np.r_[xy,1.]@np.array([[.6,0.],[0.,.5],[-.3,-.25]])
                 for name,xy in targets.items()}
        initial["alpha"] = initial["alpha"] + np.array([0., -.10])
        initial["beta"] = initial["beta"] + np.array([0., -.10])
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        planner.record_failure(planner._action("alpha",final=True),
                               "trajectory intersects beta")
        planner.record_failure(planner._action("beta",final=True),
                               "trajectory intersects gamma")
        _,decision=planner.next_action()
        self.assertEqual(decision["object"],"gamma")

    def test_access_chain_tries_unfailed_arm_before_moving_blocker_chain(self):
        targets={"alpha":(.25,.50),"beta":(.50,.50),"gamma":(.75,.50)}
        initial={name:np.r_[xy,1.]@np.array([[.6,0.],[0.,.5],[-.3,-.25]])
                 for name,xy in targets.items()}
        initial["alpha"] += np.array([0.,-.10])
        initial["beta"] += np.array([0.,-.10])
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        planner.record_failure(planner._action("alpha",final=True),
            "Robot touched non-target objects during grasp: ['beta']")
        original=planner._above_blockers
        planner._above_blockers=lambda name: ["gamma"] if name == "beta" else original(name)
        action,decision=planner.next_action()
        self.assertEqual(decision["object"],"alpha")
        self.assertEqual(action["pose"]["arm"],"right")

    def test_access_cycle_breaks_inside_strongly_connected_component(self):
        targets={"alpha":(.25,.50),"beta":(.50,.50),"gamma":(.75,.50)}
        initial={name:np.r_[xy,1.]@np.array([[.6,0.],[0.,.5],[-.3,-.25]])
                 for name,xy in targets.items()}
        initial["alpha"] += np.array([0.,-.10])
        initial["beta"] += np.array([0.,-.10])
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        planner.record_failure(planner._action("alpha",final=True),
                               "trajectory intersects beta")
        planner.record_failure(planner._action("beta",final=True),
                               "trajectory intersects alpha")
        planner.motion_successes["gamma"]=99
        _,decision=planner.next_action()
        self.assertIn(decision["object"],{"alpha","beta"})
        self.assertEqual(set(decision["cycle_component"]),{"alpha","beta"})
        self.assertNotEqual(decision["object"],"gamma")

    def test_access_cycle_tries_other_arm_before_buffer_escape(self):
        targets={"alpha":(.25,.50),"beta":(.50,.50)}
        initial={name:np.r_[xy,1.]@np.array([[.6,0.],[0.,.5],[-.3,-.25]])
                 for name,xy in targets.items()}
        initial["alpha"] += np.array([0.,-.10])
        initial["beta"] += np.array([0.,-.10])
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        alpha=planner._action("alpha",final=True)
        beta=planner._action("beta",final=True)
        planner.record_failure(alpha,"Robot touched non-target objects during grasp: ['beta']")
        planner.record_failure(beta,"Robot touched non-target objects during grasp: ['alpha']")
        action,decision=planner.next_action()
        selected=decision["object"]
        natural=env._arm(selected)
        self.assertNotEqual(action["pose"]["arm"],natural)
        self.assertIn(decision["kind"],{"final","buffer"})

    def test_cycle_escape_includes_unnamed_arm_failure(self):
        targets={"alpha":(.25,.50),"beta":(.50,.50)}
        initial={name:np.r_[xy,1.]@np.array([[.6,0.],[0.,.5],[-.3,-.25]])
                 for name,xy in targets.items()}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        planner.failed_arms["alpha"]={"left"}
        self.assertEqual(planner._set_cycle_escape_arm("alpha"),"right")

    def test_failed_cycle_override_returns_to_available_arm(self):
        targets={"alpha":(.25,.50),"beta":(.50,.50)}
        initial={name:np.r_[xy,1.]@np.array([[.6,0.],[0.,.5],[-.3,-.25]])
                 for name,xy in targets.items()}
        env=FakeEnvironment(initial)
        planner=OnlineRearrangementPlanner(env,target_graph(targets))
        planner.cycle_arm_overrides["alpha"]="right"
        planner.failed_arms["alpha"]={"right"}
        self.assertEqual(planner._action_arm("alpha"),"left")
        self.assertNotIn("alpha",planner.cycle_arm_overrides)


if __name__ == "__main__":
    unittest.main()
