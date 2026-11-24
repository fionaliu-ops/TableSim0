"""Regression tests for externally measured completion and episode accounting."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

from high_level_planner.evaluation import EpisodeMetrics, aggregate_reports
from high_level_planner.scene_graph.attributes import ObjectAttributes
from high_level_planner.scene_graph.graph import SceneGraph, SceneObject, Relation
from script.episode_evaluation import evaluated_closed_loop
from script.summarize_episode_metrics import main as summarize


def scene(a=.2, b=.7):
    return SceneGraph(objects=[SceneObject(key, key, attributes=ObjectAttributes(xy=[x, .5]),
        metadata={"planning_pose": {"frame": "image_normalized", "xy": [x, .5], "support": "table"}})
        for key, x in (("a", a), ("b", b))])


class EpisodeEvaluationTests(unittest.TestCase):
    def collector(self, initial=None, target=None, **kwargs):
        metrics = EpisodeMetrics(method="test", **kwargs)
        metrics.configure(initial or scene(), target or scene(.4, .7))
        return metrics

    def test_progress_uses_frozen_goals_and_initial_credit_is_separate(self):
        metrics = self.collector()
        metrics.execution(success=True, seconds=.2)
        metrics.observe(scene(.4, .7), payload_released=True)
        result = metrics.report("completed")
        self.assertTrue(result["task_success"])
        self.assertEqual(result["goal_progress"]["predicate_count"], 2)
        self.assertEqual(result["goal_progress"]["initial_fraction"], .5)
        self.assertEqual(result["goal_progress"]["final_fraction"], 1.)
        self.assertEqual(result["goal_progress"]["net_change"], .5)
        self.assertEqual(result["goal_progress"]["initially_unsatisfied_completion"], 1.)

    def test_newly_misplaced_object_is_not_hidden_by_equal_net_progress(self):
        metrics = self.collector()
        metrics.observe(scene(.4, .9), payload_released=True)
        result = metrics.report("step_limit")
        self.assertFalse(result["task_success"])
        self.assertEqual(result["goal_progress"]["net_change"], 0.)
        self.assertEqual(result["preservation"]["newly_misplaced_object_ids"], ["b"])
        self.assertEqual(result["preservation"]["newly_misplaced_object_rate"], 1.)

    def test_success_requires_fresh_observation_and_release(self):
        metrics = self.collector(initial=scene(.4, .7))
        metrics.execution(success=True, seconds=.1)
        self.assertFalse(metrics.report("completed")["task_success"])
        self.assertIsNone(metrics.report("completed")["goal_progress"]["final_fraction"])
        metrics.observe(scene(.4, .7), payload_released=False)
        self.assertFalse(metrics.report("completed")["task_success"])
        metrics.observe(scene(.4, .7), payload_released=True)
        self.assertTrue(metrics.report("completed")["task_success"])
        metrics.invalidate_observation("camera disconnected")
        self.assertFalse(metrics.report("completed")["evaluation_available"])

    def test_missing_or_stale_object_is_not_physical_loss_or_success(self):
        metrics = self.collector()
        missing = scene(.4, .7)
        missing.objects[0].metadata["observed_in_current_frame"] = False
        metrics.observe(missing, payload_released=True)
        result = metrics.report("completed")
        self.assertFalse(result["task_success"])
        self.assertEqual(result["layout"]["missing_or_unobserved_ids"], ["a"])
        self.assertIsNone(result["preservation"]["lost_object_count"])
        self.assertEqual(result["layout"]["object_pose_match_rate"], .5)

    def test_target_revision_cannot_move_evaluation_goalposts(self):
        target = scene(.4, .7)
        metrics = self.collector(target=target)
        target.objects[0].metadata["planning_pose"]["xy"] = [.2, .5]
        metrics.target_revision_count += 1
        metrics.observe(scene(), payload_released=True)
        self.assertFalse(metrics.report("completed")["task_success"])
        with self.assertRaises(ValueError):
            metrics.configure(scene(), scene())

    def test_fatal_event_overrides_final_goal_match(self):
        metrics = self.collector()
        metrics.execution(success=False, seconds=.2, fatal=True)
        metrics.observe(scene(.4, .7), payload_released=True)
        self.assertFalse(metrics.report("completed")["task_success"])

    def test_no_disturbance_is_na_and_failure_is_not_a_disturbance(self):
        metrics = self.collector()
        metrics.execution(success=False, seconds=.2)
        metrics.observe(scene(.4, .7), payload_released=True)
        result = metrics.report("completed")
        self.assertIsNone(result["disturbance_recovery"]["eventual_task_success_rate"])
        self.assertTrue(result["disturbance_recovery"]["failure_followed_by_task_success"])
        metrics.disturbance({"kind": "move_object"})
        self.assertFalse(metrics.report("completed")["task_success"])
        metrics.observe(scene(.4, .7), payload_released=True)
        self.assertEqual(metrics.report("completed")["disturbance_recovery"]["eventual_task_success_rate"], 1.)

    def test_callback_denominator_changes_are_rejected(self):
        metrics = EpisodeMetrics(method="callback")
        metrics.observe(evaluation={"success": False, "goal_checks": {"a": False}})
        with self.assertRaises(ValueError):
            metrics.observe(evaluation={"success": True, "goal_checks": {"b": True}})

    def test_inconsistent_callback_success_cannot_override_failed_goals(self):
        metrics = EpisodeMetrics(method="callback")
        metrics.observe(evaluation={"success": True, "goal_checks": {"a": False}})
        self.assertFalse(metrics.report("completed")["task_success"])

    def test_spatial_relations_are_in_progress_even_if_not_compiled_as_predicates(self):
        target = scene(.4, .7)
        target.relations.append(Relation("a", "b", "left_of"))
        metrics = self.collector(target=target)
        metrics.observe(scene(.4, .7), payload_released=True)
        self.assertEqual(metrics.report("completed")["goal_progress"]["predicate_count"], 3)
        self.assertEqual(metrics.report("completed")["layout"]["target_relation_satisfaction"], 1.)

    def test_audits_count_retries_and_report_partial_token_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory, "audit_0000.json")
            old.write_text(json.dumps({"model": "m", "request_index": 0, "usage": {"prompt_tokens": 999}}))
            metrics = self.collector(output_dir=directory)
            for index, extra in enumerate([{"complete": False}, {"complete": True,
                "usage": {"prompt_tokens": 20, "completion_tokens": 5}}], 1):
                Path(directory, f"audit_{index:04d}.json").write_text(json.dumps({"request_index": index,
                    "model": "m", "elapsed_seconds": .5, **extra}))
            result = metrics.write("step_limit")
            usage = result["efficiency"]["model_usage"]
            self.assertEqual(usage["request_attempts"], 2)
            self.assertEqual(usage["total_tokens"], 25)
            self.assertFalse(usage["token_usage_complete"])
            self.assertTrue(Path(directory, "evaluation_metrics.md").is_file())
            self.assertEqual(json.loads(Path(directory, "evaluation_metrics.json").read_text()), result)

    def test_aggregate_counts_episodes_not_goal_predicates(self):
        metrics = self.collector(initial=scene(.4, .7))
        metrics.observe(scene(.4, .7), payload_released=True)
        complete = metrics.report("completed")
        failed = self.collector().report("step_limit")
        report = aggregate_reports([complete, failed])
        self.assertEqual(report["episodes"], 2)
        self.assertEqual(report["success_rate"], .5)
        self.assertLess(report["success_rate_wilson_95"][0], .5)
        self.assertGreater(report["success_rate_wilson_95"][1], .5)
        self.assertIsNone(aggregate_reports([])["success_rate"])

    def test_dynamic_boundary_writes_on_error_and_restores_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            env = NS(output_dir=Path(directory), held=None, step=Mock(return_value={}), observe=Mock())
            planner = NS(scene_updater=NS(update_scene_graph=Mock(return_value=scene(.4, .7))))
            original_step = env.step

            @evaluated_closed_loop
            def run(env, planner, instruction):
                env._episode_metrics.configure(scene(), scene(.4, .7))
                env.step({})
                planner.scene_updater.update_scene_graph(None)
                raise RuntimeError("replanning failed")

            with self.assertRaisesRegex(RuntimeError, "replanning failed"):
                run(env, planner, "tidy")
            report = json.loads(Path(directory, "evaluation_metrics.json").read_text())
            self.assertEqual(report["efficiency"]["attempted_actions"], 1)
            self.assertFalse(report["task_success"])
            self.assertIs(env.step, original_step)
            self.assertIsNone(env._episode_metrics)

    def test_summary_cli_deduplicates_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = self.collector(output_dir=directory)
            metrics.write("step_limit")
            output = Path(directory, "summary.json")
            self.assertEqual(summarize([directory, directory, "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text())["test"]["episodes"], 1)

    def test_live_closed_loop_zero_action_success_writes_report(self):
        from unittest.mock import patch
        from script.dynamic_sim import run_closed_loop
        from high_level_planner.reasoning_graph.graph import ReasoningGraph
        from high_level_planner.reasoning_graph.state import TaskState
        from high_level_planner.planner.ucs import SearchResult
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory, "frame.png")
            image.write_bytes(b"camera frame used only for provenance")
            initial = scene(.4, .7)
            graph = ReasoningGraph()
            graph.add_node(TaskState("S0", "already organized", scene_graph=initial), goal=True)
            env = NS(output_dir=Path(directory), held=None, image_goal=None,
                observe=lambda: {"image": image}, bind_graph=lambda *args: None,
                evaluate=lambda: True, step=Mock())
            evidence = NS(update_from_scene=lambda *args, **kw: None,
                attach_execution_context=lambda *args: None, to_dict=lambda: {})
            planner = NS(scene_constructor=NS(construct=lambda *args, **kw: initial),
                reasoning_generator=None, scene_updater=None, evidence_store=evidence,
                memory=NS(update=lambda *args, **kw: None, current_node_id="S0", traversed_node_ids=[]),
                graph_planner=NS(plan=lambda graph: SearchResult(goal_state="S0", start_state="S0")))
            with patch("high_level_planner.reasoning_graph.target_conditioned.generate_between_scenes", return_value=graph):
                result = run_closed_loop(env, planner, "tidy", target_graph=initial, graph_validation=True)
            self.assertEqual(result.status, "certified_terminal")
            env.step.assert_not_called()
            report = json.loads(Path(directory, "evaluation_metrics.json").read_text())
            self.assertTrue(report["task_success"])
            self.assertEqual(report["efficiency"]["attempted_actions"], 0)

    def test_callback_runners_return_and_write_the_same_report(self):
        from baseline.mcts_runner import run_tidiness_episode
        from baseline.paper_runner import run_paper_episode
        with tempfile.TemporaryDirectory() as directory:
            mcts = NS(reset=lambda: None, metrics={})
            result = run_tidiness_episode(mcts, observe=lambda: NS(scene=scene(), visual=None),
                execute=lambda _: self.fail("Already-complete task must not execute"),
                evaluate=lambda _: {"success": True, "goal_checks": {"placed": True}},
                output_dir=directory)
            self.assertTrue(result["evaluation_metrics"]["task_success"])
            self.assertEqual(result["evaluation_metrics"], json.loads(Path(directory, "evaluation_metrics.json").read_text()))
            paper = NS(method="test_paper", reset=lambda: None, metrics={})
            result = run_paper_episode(paper, observe=lambda: NS(scene=scene(), image=None),
                execute=lambda _: self.fail("Already-complete task must not execute"),
                evaluate=lambda _: {"success": True, "goal_checks": {"placed": True}},
                instruction="tidy", output_path=Path(directory, "paper", "episode.json"))
            self.assertTrue(result["evaluation_metrics"]["task_success"])
            self.assertTrue(Path(directory, "paper", "evaluation_metrics.json").is_file())

    def test_legacy_direct_evaluator_preserves_simulator_metric_scope(self):
        from unittest.mock import patch
        from script.direct_dynamic_sim import automatic_evaluation
        measured = {"objects": {"a": {"passed": True, "position_error_m": .01, "image_error": .02}},
                    "orientations": {}, "supports": {"book_on_folder": {"passed": True}}}
        env = NS(held=None, image_goal=NS(evaluate=lambda env: measured))
        with patch("script.direct_dynamic_sim.aligned_goal", return_value=True):
            result = automatic_evaluation(env)
        self.assertTrue(result["success"])
        self.assertEqual(result["layout"]["object_errors"]["a"]["units"], "m")
        self.assertEqual(result["layout"]["target_relation_satisfaction"], 1.)


if __name__ == "__main__":
    unittest.main()
