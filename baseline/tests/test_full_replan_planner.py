from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from baseline import FullReplanPlanner
from high_level_planner.goal_generator import GoalGenerator
from high_level_planner.reasoning_graph import ReasoningGraphGenerator
from high_level_planner.scene_graph.constructor import SceneGraphConstructor
from high_level_planner.tests.test_goal_generator import (
    FakePerception, FakeRenderer, FakeLayoutExtractor, draw_scene,
)
from high_level_planner.tests.test_target_conditioned import TargetActionModel
from script.dynamic_sim import request_action


class FullReplanTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.image = root / "initial.png"
        draw_scene(self.image)
        backend = FakePerception()
        constructor = SceneGraphConstructor(perception_backend=backend)
        self.initial = constructor.construct(self.image)
        self.goals = GoalGenerator(image_generator=FakeRenderer(root / "images"),
                                   perception_backend=backend,
                                   layout_graph_extractor=FakeLayoutExtractor(),
                                   output_dir=root / "goals")
        self.target = self.goals.generate(self.image, initial_scene_graph=self.initial).scene_graph
        self.model = TargetActionModel()
        self.planner = FullReplanPlanner(scene_constructor=constructor,
            reasoning_generator=ReasoningGraphGenerator(model=self.model, max_depth=4))

    def start(self):
        return self.planner.plan_initial_scene(self.initial, self.target, "Organize")

    def test_unchanged_observation_still_rebuilds_without_scheduler_or_repair(self):
        first = self.start()
        previous_prompts = len(self.model.prompts)
        generator = self.planner._target_reasoning_generator
        with patch.object(generator, "update_reasoning_graph", side_effect=AssertionError("No repair")), \
             patch.object(self.planner.repair_scheduler, "decide", side_effect=AssertionError("No scheduling")), \
             patch.object(generator, "generate", wraps=generator.generate) as generate:
            second = self.planner.update_target_conditioned_scene(self.initial, "Organize")
        generate.assert_called_once()
        self.assertGreater(len(self.model.prompts), previous_prompts)
        self.assertIsNot(first.reasoning_graph, second.reasoning_graph)
        self.assertIsNotNone(second.search_result.goal_state)
        self.assertEqual(self.planner.metrics["full_replans"], 1)
        self.assertFalse(second.reasoning_graph.metadata["baseline"]["incremental_repair_used"])

    def test_new_observation_starts_at_new_root_and_drops_old_cursor(self):
        first = self.start()
        edge = first.next_subtask
        self.planner.record_execution_result(edge, success=True)
        self.planner.mark_subtask_completed(edge)
        observed = deepcopy(first.reasoning_graph.get_node(edge.to_state).scene_graph)
        observed.version += 1
        second = self.planner.update_target_conditioned_scene(observed, "Organize")
        self.assertEqual(second.search_result.start_state, second.reasoning_graph.start_id)
        self.assertEqual(self.planner.memory.current_node_id, second.reasoning_graph.start_id)
        self.assertEqual(self.planner.memory.traversed_node_ids, [])
        self.assertEqual(len(second.search_result.path), 1)
        self.assertEqual(request_action(second.vla_request)["source"], second.next_subtask.source_object_id)

    def test_target_remains_fixed_and_inputs_are_not_mutated(self):
        original_initial, original_target = deepcopy(self.initial.to_dict()), deepcopy(self.target.to_dict())
        self.start()
        target = deepcopy(self.planner._target_scene_graph.to_dict())
        observed = deepcopy(self.initial)
        before = deepcopy(observed.to_dict())
        self.planner.update_target_conditioned_scene(observed, "Organize")
        self.assertEqual(self.planner._target_scene_graph.to_dict(), target)
        self.assertEqual(observed.to_dict(), before)
        self.assertEqual(self.initial.to_dict(), original_initial)
        self.assertEqual(self.target.to_dict(), original_target)

    def test_failed_execution_is_not_resurrected(self):
        first = self.start()
        self.planner.record_execution_result(first.next_subtask, success=False,
                                              details={"error": "grasp failed"})
        second = self.planner.update_target_conditioned_scene(self.initial, "Organize")
        self.assertTrue(second.reasoning_graph.metadata["suppressed_failed_edges"])
        self.assertIsNone(second.search_result.goal_state)
        self.assertIsNone(second.next_subtask)
        self.assertTrue(second.vla_request.metadata["empty"])

    def test_fatal_failure_stops_before_regeneration(self):
        first = self.start()
        self.planner.record_execution_result(first.next_subtask, success=False, fatal=True)
        with patch.object(self.planner._target_reasoning_generator, "generate") as generate:
            with self.assertRaisesRegex(RuntimeError, "fatal_execution_failure"):
                self.planner.update_target_conditioned_scene(self.initial, "Organize")
        generate.assert_not_called()

    def test_destination_failure_does_not_ban_other_destinations(self):
        first = self.start()
        edge = first.next_subtask
        failed_pose = {'frame': 'image_normalized', 'xy': [.7, .3]}
        self.planner.record_execution_result(edge, success=False, details={
            'failure_category': 'destination_obstructed',
            'requested_pose': failed_pose})
        for xy, suppressed in [([.7, .3], True), ([.3, .7], False)]:
            graph = deepcopy(first.reasoning_graph)
            candidate = next(e for e in graph.edges
                             if e.metadata['edge_id'] == edge.metadata['edge_id'])
            candidate.parameters['pose'] = {'frame': 'image_normalized', 'xy': xy}
            candidate.metadata.pop('physical_execution', None)
            self.planner._apply_failed_execution_evidence(graph)
            self.assertEqual(candidate.metadata.get('physical_execution', {}).get('status')
                             == 'failed_in_prior_attempt', suppressed)

    def test_moving_payload_does_not_clear_an_unchanged_destination_blocker(self):
        from high_level_planner.reasoning_graph.geometry import pose
        first=self.start();edge=first.next_subtask;graph=deepcopy(first.reasoning_graph)
        current=graph.get_node(edge.from_state).scene_graph
        blocker=next(o.id for o in current.objects if o.id!=edge.source_object_id)
        requested={'frame':'image_normalized','xy':[.7,.3]}
        self.planner.record_execution_result(edge,success=False,details={
            'failure_category':'destination_occupied','requested_pose':requested,
            'source_visual_pose':{'frame':'image_normalized','xy':[-1.,-1.]},
            'blocking_object_ids':[blocker],'blocking_visual_poses':{blocker:pose(current,blocker)}})
        candidate=next(e for e in graph.edges if e.metadata['edge_id']==edge.metadata['edge_id'])
        candidate.parameters['pose']=requested
        self.planner._apply_failed_execution_evidence(graph)
        self.assertEqual(candidate.metadata['physical_execution']['status'],'failed_in_prior_attempt')

    def test_image_routes_replan_without_regenerating_goal(self):
        self.planner.plan_initial_image(self.image, "Organize", goal_generator=self.goals)
        with patch.object(self.goals, "generate", side_effect=AssertionError("Fixed goal")), \
             patch.object(self.planner.scene_updater, "update_scene_graph", return_value=deepcopy(self.initial)) as updater:
            self.planner.process_observation(self.image, "Organize")
            self.planner.update_from_image(self.image, "Organize")
            self.planner.update_target_conditioned_image(self.image, "Organize")
        self.assertEqual(updater.call_count, 3)
        self.assertEqual(self.planner.metrics["full_replans"], 3)

    def test_goal_observation_yields_empty_path_at_new_root(self):
        first = self.start()
        observed = deepcopy(first.reasoning_graph.get_node(first.search_result.goal_state).scene_graph)
        second = self.planner.update_target_conditioned_scene(observed, "Organize")
        self.assertEqual(second.search_result.goal_state, second.search_result.start_state)
        self.assertEqual(second.search_result.path, [])
        self.assertIsNone(second.next_subtask)

    def test_real_updater_keeps_new_execution_context_in_rebuild(self):
        first = self.planner.plan_initial_image(self.image, "Organize", goal_generator=self.goals)
        self.planner.record_execution_result(first.next_subtask, success=False,
                                              details={"error_type": "test_execution_failure",
                                                       "object_poses": {"hidden_truth": [1, 2, 3]}})
        before_prompts = len(self.model.prompts)
        result = self.planner.update_target_conditioned_image(self.image, "Organize")
        self.assertEqual(self.planner.metrics["full_replans"], 1)
        self.assertGreater(result.scene_graph.version, first.scene_graph.version)
        prompts = self.model.prompts[before_prompts:]
        self.assertTrue(prompts)
        payload = json.loads(prompts[0].split("INPUT:\n", 1)[1])
        self.assertIn("test_execution_failure", json.dumps(payload["recent_execution_evidence"]))
        self.assertNotIn("hidden_truth", json.dumps(payload["recent_execution_evidence"]))
        self.assertFalse(self.planner._target_reasoning_generator.scene_graph_update_pending)

    def test_new_task_resets_round_statistics(self):
        self.start()
        self.planner.update_target_conditioned_scene(self.initial, "Organize")
        self.start()
        self.assertEqual(self.planner.metrics["planning_rounds"], 1)
        self.assertEqual(self.planner.metrics["full_replans"], 0)

    def test_update_requires_fixed_goal_session(self):
        with self.assertRaisesRegex(ValueError, "session"):
            self.planner.update_target_conditioned_scene(self.initial, "Organize")


if __name__ == "__main__":
    unittest.main()
