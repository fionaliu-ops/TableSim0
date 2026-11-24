import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from script.symbolic_feedback import run_feedback_loop
from script.symbolic_grasp_execution import GraspBlockedError


class SymbolicFeedbackTest(unittest.TestCase):
    def test_occlusion_replans_before_next_action_and_uses_new_edge(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Mock(); scene.metadata = {}
            graph = Mock(); graph.metadata = {'closed_loop':{'latest':{'mode':'full_rebuild'}}}
            pick = NS(from_state='S0',to_state='S1',source_object_id='book')
            unblock = NS(from_state='R0',to_state='R1',source_object_id='cup')
            search = NS(next_subtask=pick)
            request = NS(to_dict=lambda:{'action':'test'})
            planner = NS(_target_reasoning_generator=object(), memory=Mock(),
                vla_interface=NS(format_subtask=Mock(return_value=request)),
                scene_updater=NS(update_scene_graph=Mock(return_value=scene)),
                record_execution_result=Mock(return_value={'success':False}),
                graph_planner=Mock(), update_target_conditioned_scene=Mock(return_value=NS(
                    scene_graph=scene, reasoning_graph=graph, search_result=NS(next_subtask=unblock))))
            env = NS(output_dir=Path(directory), held=None, last_placement={},
                step=Mock(side_effect=[GraspBlockedError({'blocking_object_ids':['cup'],
                    'failure_category':'grasp_occluded'}), {'image':'after_unblock.png'}]),
                observe=Mock(return_value={'image':'failed_grasp.png'}))
            evaluator = Mock(); evaluator.evaluate.side_effect = [NS(terminal=False),NS(terminal=False),NS(terminal=True)]
            localized = NS(node_id='R1',conclusive=True)
            with patch('high_level_planner.reasoning_graph.target_conditioned.TargetGraphEvaluator',return_value=evaluator), \
                 patch('high_level_planner.reasoning_graph.node_localization.locate_reasoning_node',return_value=localized), \
                 patch('high_level_planner.reasoning_graph.geometry.pose',return_value={'frame':'image_normalized','xy':[.3,.4]}):
                result = run_feedback_loop(env,planner,scene,scene,graph,search,'goal',max_steps=2)
            self.assertEqual(result.status,'symbolic_grasp_visually_verified')
            feedback = planner.record_execution_result.call_args_list[0].kwargs
            self.assertFalse(feedback['success'])
            self.assertEqual(feedback['details']['blocking_object_ids'],['cup'])
            planner.update_target_conditioned_scene.assert_called_once()
            selected = [call.args[0] for call in planner.vla_interface.format_subtask.call_args_list]
            self.assertEqual(selected,[pick,pick,unblock,unblock])
            trace=json.loads(Path(directory,'dynamic_trace.json').read_text())
            self.assertEqual(trace['steps'][0]['observation'],'failed_grasp.png')
            self.assertEqual(trace['steps'][1]['reasoning_from'],'R0')

    def test_model_feedback_retains_occluders_but_excludes_simulation_poses(self):
        from high_level_planner.evidence import EvidenceStore
        from high_level_planner.reasoning_graph.state import Subtask
        store=EvidenceStore()
        edge=Subtask('S0','S1','grasp book',action_type='move',object_target={'source':'book','target':'table'})
        store.record_execution(edge,success=False,details={'failure_category':'grasp_occluded',
            'blocking_object_ids':['cup'],'phase':'before_grasp','object_moved':False,
            'world_pose':[.1,.2,.7],'blocking_visual_poses':{'cup':{'xy':[.3,.4]}}})
        details=store.execution_context()['recent_events'][0]['details']
        self.assertEqual(details['blocking_object_ids'],['cup'])
        self.assertEqual(details['failure_category'],'grasp_occluded')
        self.assertFalse(details['object_moved'])
        self.assertNotIn('world_pose',details)
        self.assertNotIn('blocking_visual_poses',details)

    def test_blocked_action_can_be_retried_after_blocker_moves(self):
        from high_level_planner.planner.planner import HighLevelPlanner
        from high_level_planner.reasoning_graph.graph import ReasoningGraph
        from high_level_planner.reasoning_graph.state import Subtask, TaskState
        from high_level_planner.tests.test_evidence_closed_loop import scene
        graph=ReasoningGraph()
        before, after=scene('cup','book'),scene('cup','book')
        for state, xy in [(before,[.3,.4]),(after,[.7,.4])]:
            state.get_object('cup').metadata['planning_pose']={'frame':'image_normalized','xy':xy}
        graph.add_node(TaskState('S0','blocked',scene_graph=before))
        graph.add_node(TaskState('S1','unblocked',scene_graph=after))
        for origin in ['S0','S1']:
            graph.add_edge(Subtask(origin,'S1','move book',action_type='move',
                object_target={'source':'book','target':'table'},metadata={'dependencies':{'validity':'valid'}}))
        event={'action_type':'move','source_object_id':'book','target_object_id':'table','success':False,
               'details':{'blocking_visual_poses':{'cup':{'frame':'image_normalized','xy':[.3,.4]}}}}
        planner=NS(evidence_store=NS(execution_events=[event]))
        HighLevelPlanner._apply_failed_execution_evidence(planner,graph)
        self.assertEqual(graph.edges[0].metadata['dependencies']['validity'],'invalid')
        self.assertEqual(graph.edges[1].metadata['dependencies']['validity'],'valid')

    def test_rebuild_with_ready_action_survives_missing_complete_goal_path(self):
        from high_level_planner.planner.planner import HighLevelPlanner
        from high_level_planner.evidence import EvidenceStore
        from high_level_planner.reasoning_graph.graph import ReasoningGraph
        from high_level_planner.reasoning_graph.state import Subtask, TaskState
        from high_level_planner.tests.test_evidence_closed_loop import scene
        observed=scene('cup','book')
        old=ReasoningGraph(); old.add_node(TaskState('S0','observed',scene_graph=observed))
        rebuilt=ReasoningGraph(start_id='R0'); rebuilt.add_node(TaskState('R0','observed',scene_graph=observed))
        rebuilt.add_node(TaskState('R1','cup removed',scene_graph=observed))
        rebuilt.add_edge(Subtask('R0','R1','move cup',action_type='move',object_target={'source':'cup','target':'table'}))
        result=lambda start:NS(goal_state=None,next_subtask=None,start_state=start)
        generator=NS(update_reasoning_graph=Mock(return_value=old))
        planner=NS(_target_reasoning_generator=generator,_target_scene_graph=observed,
            memory=NS(reasoning_graph=old,current_node_id='S0',traversed_node_ids=[],update=Mock()),
            evidence_store=EvidenceStore(),_contract_complete=lambda graph:True,
            repair_scheduler=NS(decide=Mock(return_value=NS(mode='incremental_repair',to_dict=lambda:{}))),
            _apply_failed_execution_evidence=Mock(),_full_target_rebuild=Mock(return_value=rebuilt),
            graph_planner=NS(plan=Mock(side_effect=[result('S0'),result('R0')])),
            vla_interface=NS(format_subtask=Mock()),_closed_loop_decisions=[])
        output=HighLevelPlanner.update_target_conditioned_scene(planner,observed,'goal',allow_partial_execution=True)
        self.assertIs(output.reasoning_graph,rebuilt)
        self.assertTrue(rebuilt.metadata['closed_loop']['latest']['used_full_rebuild_after_empty_repair'])

    def test_completed_move_reassociates_visual_identity_without_world_coordinates(self):
        from high_level_planner.scene_graph.updater import SceneGraphUpdater
        from high_level_planner.tests.test_evidence_closed_loop import scene
        old,new=scene('cup'),scene('new_cup')
        for graph in [old,new]:
            obj=graph.objects[0];obj.name='cup';obj.attributes.bbox=[0,0,30,40]
            obj.attributes.extra['detection_metadata']={'visual_appearance':[1.,0.]}
        new.objects[0].attributes.xy=[.8,.2]
        new.objects[0].metadata['visual_identity_new']=True
        SceneGraphUpdater.associate_moved_object(old,new,'cup')
        self.assertEqual(new.objects[0].metadata['stable_scene_object_id'],'cup')
        self.assertEqual(new.objects[0].attributes.xy,[.8,.2])
        self.assertFalse(new.objects[0].metadata['visual_identity_new'])
        new.objects[0].metadata.clear()
        new.objects[0].attributes.extra['detection_metadata']['visual_appearance']=[0.,1.]
        SceneGraphUpdater.associate_moved_object(old,new,'cup')
        self.assertNotIn('stable_scene_object_id',new.objects[0].metadata)

    def test_unreadable_failure_frame_does_not_continue_old_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            scene=Mock(); scene.metadata={}; graph=Mock(); edge=NS(from_state='S0',to_state='S1')
            planner=NS(_target_reasoning_generator=object(),memory=Mock(),
                vla_interface=NS(format_subtask=Mock(return_value=NS(to_dict=lambda:{}))),
                scene_updater=NS(update_scene_graph=Mock(side_effect=ValueError('unreadable image'))),
                record_execution_result=Mock(return_value={}),update_target_conditioned_scene=Mock())
            env=NS(output_dir=Path(directory),held=None,step=Mock(side_effect=RuntimeError('blocked')),
                   observe=Mock(return_value={'image':'failure.png'}))
            evaluator=Mock();evaluator.evaluate.return_value=NS(terminal=False)
            with patch('high_level_planner.reasoning_graph.target_conditioned.TargetGraphEvaluator',return_value=evaluator):
                result=run_feedback_loop(env,planner,scene,scene,graph,NS(next_subtask=edge),'goal')
            self.assertEqual(result.status,'feedback_replan_failed')
            env.step.assert_called_once()
            planner.update_target_conditioned_scene.assert_not_called()

    def test_unconfirmed_same_label_does_not_block_moved_object_identity(self):
        from high_level_planner.tests.test_evidence_closed_loop import scene
        from high_level_planner.scene_graph.updater import SceneGraphUpdater
        old,new=scene('cup'),scene('new_cup')
        for graph in (old,new):
            obj=graph.objects[0];obj.name='cup';obj.attributes.bbox=[0,0,30,40]
            obj.attributes.extra['detection_metadata']={'visual_appearance':[1.,0.]}
            obj.metadata['visual_identity_confirmed']=True
            ghost=__import__('copy').deepcopy(obj);ghost.id='unconfirmed_shadow'
            ghost.metadata['visual_identity_confirmed']=False
            graph.objects.append(ghost)
        SceneGraphUpdater.associate_moved_object(old,new,'cup')
        self.assertEqual(new.objects[0].metadata['stable_scene_object_id'],'cup')


if __name__ == '__main__': unittest.main()
