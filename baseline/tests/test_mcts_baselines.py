from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

from baseline import (MCTSConfig, GoalConditionedMCTSPlanner, BudgetedUCSPlanner,
                      TidinessMCTSPlanner)
from baseline.mcts import search
from baseline.mcts_runner import run_tidiness_episode
from baseline.tidiness import TidyingState, GridProposal, HeuristicTidiness, clean_observation, move_action, scene_key
from high_level_planner.interface.vla_interface import Pi05VLAInterface
from high_level_planner.planner.ucs import uniform_cost_search
from high_level_planner.reasoning_graph.graph import ReasoningGraph
from high_level_planner.reasoning_graph.state import Subtask, TaskState
from high_level_planner.reasoning_graph.transition import SceneGraphTransitionModel
from high_level_planner.scene_graph.attributes import ObjectAttributes
from high_level_planner.scene_graph.graph import SceneGraph, SceneObject


def scene_fixture():
    return SceneGraph(objects=[SceneObject(key, "book", attributes=ObjectAttributes(
        xy=xy, footprint=[.08, .06], yaw_degrees=yaw, category="document",
        extra={"geometry_frame": "image_normalized"}))
        for key, xy, yaw in [("a", [.2, .3], 35.), ("b", [.8, .7], 40.)]],
        metadata={"planning_context": {"workspace": {"frame": "image_normalized", "bounds": [.05, .05, .95, .95]}}})


def graph_fixture():
    graph = ReasoningGraph()
    for key in ("S0", "S1", "S2", "G"):
        graph.add_node(TaskState(key, key, metadata={"goal_report": {"goal_checks": {"p": key == "G"}}}), goal=key == "G")
    for source, target, cost in [("S0", "S1", 1.), ("S1", "G", 1.), ("S0", "S2", 4.), ("S2", "G", 4.)]:
        graph.add_edge(Subtask(source, target, "move book", cost=cost, action_type="move",
                              object_target={"source": "a", "target": "table"}))
    return graph


class GraphMCTSTest(unittest.TestCase):
    def test_injected_highlevel_image_pipeline(self):
        from high_level_planner import HighLevelPlanner
        from high_level_planner.goal_generator import GoalGenerator
        from high_level_planner.reasoning_graph import ReasoningGraphGenerator
        from high_level_planner.scene_graph.constructor import SceneGraphConstructor
        from high_level_planner.tests.test_goal_generator import FakePerception, FakeRenderer, FakeLayoutExtractor, draw_scene
        from high_level_planner.tests.test_target_conditioned import TargetActionModel
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "initial.png"
            draw_scene(image)
            perception = FakePerception()
            constructor = SceneGraphConstructor(perception_backend=perception)
            goals = GoalGenerator(image_generator=FakeRenderer(root / "images"),
                perception_backend=perception, layout_graph_extractor=FakeLayoutExtractor(), output_dir=root / "goals")
            searcher = GoalConditionedMCTSPlanner(config=MCTSConfig(iterations=32))
            planner = HighLevelPlanner(scene_constructor=constructor, graph_planner=searcher,
                reasoning_generator=ReasoningGraphGenerator(model=TargetActionModel(), max_depth=4))
            output = planner.plan_initial_image(image, "Organize", goal_generator=goals)
            self.assertIsNotNone(output.search_result.goal_state)
            self.assertIsNotNone(output.next_subtask)
            self.assertEqual(output.vla_request.object_target["source"], output.next_subtask.source_object_id)

    def test_same_graph_cost_and_no_mutation(self):
        graph = graph_fixture()
        before = graph.to_dict()
        result = GoalConditionedMCTSPlanner(config=MCTSConfig(iterations=64)).plan(graph)
        ucs = uniform_cost_search(graph)
        self.assertEqual((result.goal_state, result.total_cost), (ucs.goal_state, ucs.total_cost))
        self.assertEqual(graph.to_dict(), before)

    def test_invalid_dependencies_never_used(self):
        graph = graph_fixture()
        graph.edges[0].metadata["dependencies"] = {"validity": "stale"}
        for planner in (GoalConditionedMCTSPlanner(), BudgetedUCSPlanner()):
            result = planner.plan(graph)
            self.assertEqual(result.total_cost, 8.)
            self.assertNotIn(graph.edges[0], result.path)

    def test_non_root_start_and_current_terminal(self):
        planner = GoalConditionedMCTSPlanner()
        self.assertEqual(planner.plan(graph_fixture(), start_id="S1").total_cost, 1.)
        result = planner.plan(graph_fixture(), start_id="G")
        self.assertEqual((result.goal_state, result.total_cost, result.path), ("G", 0., []))

    def test_no_path_is_failure(self):
        graph = graph_fixture()
        graph.edges = graph.edges[:1]
        for planner in (GoalConditionedMCTSPlanner(), BudgetedUCSPlanner()):
            result = planner.plan(graph)
            self.assertIsNone(result.goal_state)
            self.assertEqual(result.path, [])
            self.assertTrue(math.isinf(result.total_cost))

    def test_cycles_are_bounded(self):
        graph = graph_fixture()
        graph.add_edge(Subtask("S1", "S0", "move book", action_type="move", object_target={"source": "a", "target": "table"}))
        planner = GoalConditionedMCTSPlanner(config=MCTSConfig(iterations=16, max_transitions=20))
        planner.plan(graph)
        self.assertLessEqual(planner.metrics["transitions"], 20)

    def test_budget_and_repeatability(self):
        for seed in range(3):
            config = MCTSConfig(iterations=10, max_transitions=1, rollout_depth=0, seed=seed)
            left, right = GoalConditionedMCTSPlanner(config=config), GoalConditionedMCTSPlanner(config=config)
            a, b = left.plan(graph_fixture()), right.plan(graph_fixture())
            self.assertEqual(a.path, b.path)
            self.assertLessEqual(left.metrics["transitions"], 1)
            self.assertIsNone(a.goal_state)
        ucs = BudgetedUCSPlanner(max_transitions=1)
        self.assertIsNone(ucs.plan(graph_fixture()).goal_state)
        self.assertEqual(ucs.metrics["transitions"], 1)

    def test_negative_cost_is_rejected(self):
        graph = graph_fixture()
        graph.edges[0].metadata["motion_cost"] = -1
        for planner in (GoalConditionedMCTSPlanner(), BudgetedUCSPlanner()):
            with self.assertRaises(ValueError):
                planner.plan(graph)

    def test_invalid_configuration(self):
        for kwargs in ({"iterations": 0}, {"max_depth": True}, {"rollout_mix": 2}, {"time_limit_seconds": -1}):
            with self.assertRaises(ValueError):
                MCTSConfig(**kwargs)


class TidinessMCTSTest(unittest.TestCase):
    def planner(self):
        return TidinessMCTSPlanner(config=MCTSConfig(iterations=12, max_transitions=32, max_depth=4, rollout_depth=1),
                                  proposal=GridProposal(grid_size=3, max_candidates=16))

    def test_real_transition_and_executor_schema(self):
        scene = scene_fixture()
        before = scene.to_dict()
        planner = self.planner()
        output = planner.plan_initial_scene(scene)
        self.assertEqual(output.status, "action")
        self.assertEqual(scene.to_dict(), before)
        output.next_subtask.validate_atomicity(allow_legacy_generic=False)
        from script.dynamic_sim import request_action
        decoded = request_action(output.vla_request)
        self.assertEqual(decoded["kind"], "move")
        self.assertEqual(decoded["parameters"]["pose"]["frame"], "image_normalized")
        from high_level_planner.reasoning_graph.state import AtomicAction
        edge = output.next_subtask
        action = AtomicAction(edge.action_type, edge.source_object_id, "table", parameters=edge.parameters)
        successor = SceneGraphTransitionModel().apply(clean_observation(scene), action)
        self.assertNotEqual(scene_key(scene), scene_key(successor))
        with self.assertRaises(RuntimeError):
            planner.update_from_scene(scene)

    def test_failed_action_not_repeated_in_unchanged_observation(self):
        scene, planner = scene_fixture(), self.planner()
        first = planner.plan_initial_scene(scene)
        planner.record_execution_result(first.next_subtask, success=False)
        second = planner.update_from_scene(scene)
        self.assertEqual(second.status, "action")
        self.assertNotEqual(first.next_subtask.parameters, second.next_subtask.parameters)
        self.assertEqual(planner.metrics["execution_failures"], 1)

    def test_fatal_feedback_and_new_session(self):
        planner, scene = self.planner(), scene_fixture()
        first = planner.plan_initial_scene(scene)
        planner.record_execution_result(first.next_subtask, success=False, details={"fatal": True})
        self.assertEqual(planner.update_from_scene(scene).reason, "fatal_execution_failure")
        self.assertEqual(planner.plan_initial_scene(scene).status, "action")

    def test_bad_feedback_does_not_clear_pending(self):
        planner = self.planner()
        first = planner.plan_initial_scene(scene_fixture())
        with self.assertRaises(ValueError):
            planner.record_execution_result(replace(first.next_subtask, to_state="other"), success=True)
        with self.assertRaises(RuntimeError):
            planner.update_from_scene(scene_fixture())

    def test_goal_metadata_does_not_leak(self):
        scene = scene_fixture()
        scene.metadata.update(task_goals=[{"kind": "discard"}], task_policy={"secret": True})
        scene.metadata["planning_context"]["task_goals"] = [{"kind": "discard"}]
        clean = clean_observation(scene)
        self.assertNotIn("task_goals", clean.metadata)
        self.assertNotIn("task_goals", clean.metadata["planning_context"])

    def test_missing_geometry_or_workspace_fails_closed(self):
        for field in ("footprint", "xy"):
            scene = scene_fixture()
            setattr(scene.objects[0].attributes, field, None)
            with self.assertRaises(ValueError):
                self.planner().plan_initial_scene(scene)
        scene = scene_fixture()
        scene.metadata.clear()
        with self.assertRaises(ValueError):
            self.planner().plan_initial_scene(scene)

    def test_done_is_current_scorer_judgment(self):
        class ConstantScore:
            def score(self, state):
                return 1.
        planner = TidinessMCTSPlanner(scorer=ConstantScore())
        output = planner.plan_initial_scene(scene_fixture())
        self.assertEqual(output.status, "done")
        self.assertFalse(output.metadata["external_success_verified"])
        report = run_tidiness_episode(planner, observe=lambda: TidyingState(scene_fixture()),
            execute=lambda _: self.fail("Scorer-done must not execute"), evaluate=lambda _: {"success": False})
        self.assertEqual(report["status"], "scorer_false_positive")

    def test_high_score_does_not_complete_overlapping_layout(self):
        class ConstantScore:
            def score(self, state):
                return 1.
        scene = scene_fixture()
        scene.objects[1].attributes.xy = list(scene.objects[0].attributes.xy)
        planner = TidinessMCTSPlanner(scorer=ConstantScore(), config=MCTSConfig(
            iterations=2, max_transitions=2, rollout_depth=0))
        output = planner.plan_initial_scene(scene)
        self.assertNotEqual(output.status, "done")
        self.assertFalse(output.metadata["current_geometry_valid"])

    def test_shared_runner_executes_and_uses_external_success(self):
        observations = iter([TidyingState(scene_fixture()), TidyingState(scene_fixture())])
        checks = iter([False, True])
        report = run_tidiness_episode(self.planner(), observe=lambda: next(observations),
            execute=lambda _: {"success": True}, evaluate=lambda _: {"success": next(checks)})
        self.assertEqual(report["status"], "external_success")
        self.assertEqual(report["metrics"]["executions"], 1)


if __name__ == "__main__":
    unittest.main()
