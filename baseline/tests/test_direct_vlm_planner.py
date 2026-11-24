import json
import unittest
from unittest.mock import patch

import numpy as np

from baseline import DirectVLMPlanner
from high_level_planner.reasoning_graph.language_models.base import LanguageModelError
from script.dynamic_sim import request_action


ACTION = {"status": "action", "reason": "Cup obstructs the working area",
          "action_type": "move", "instruction": "Move the cup to the left area.",
          "object_target": {"source": "cup", "target": "left_area"},
          "parameters": {}, "constraints": []}


class FakeModel:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def generate(self, prompt, *, num_responses, images):
        self.calls.append((prompt, images))
        return [json.dumps(self.answers.pop(0))]


class DirectVLMTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((16, 16, 3), dtype=np.uint8)

    def test_execution_boundary_and_feedback(self):
        model = FakeModel(ACTION, {"status": "done", "reason": "Observed clear table"})
        planner = DirectVLMPlanner(model=model)
        output = planner.plan_initial_image(self.image, "Clear the working area")
        self.assertEqual(request_action(output.vla_request)["source"], "cup")
        self.assertIsNone(output.to_dict()["reasoning_graph"])
        with self.assertRaises(RuntimeError):
            planner.update_from_image(self.image, "Clear the working area")
        planner.record_execution_result(output.next_subtask, success=True)
        planner.mark_subtask_completed(output.next_subtask)
        done = planner.update_from_image(self.image, "Clear the working area")
        self.assertTrue(done.task_completed)
        self.assertTrue(done.vla_request.metadata["empty"])
        self.assertIn('"success": true', model.calls[-1][0])
        self.assertEqual(len(model.calls), 2)

    def test_target_and_id_catalog(self):
        model = FakeModel(ACTION)
        planner = DirectVLMPlanner(model=model, target_image=self.image,
                                   object_catalog={"cup": "red cup"})
        planner.plan_initial_image(self.image, "Tidy")
        self.assertEqual(len(model.calls[0][1]), 2)
        self.assertIn("red cup", model.calls[0][0])

    def test_invalid_answers_fail_without_action(self):
        for answer in ({"status": "done", "action_type": "move"},
                       {**ACTION, "action_type": "generic"},
                       {**ACTION, "object_target": {"source": ["cup", "book"]}},
                       {**ACTION, "parameters": {"source_object_id": "book"}}):
            with self.subTest(answer=answer):
                planner = DirectVLMPlanner(model=FakeModel(answer))
                with self.assertRaises(LanguageModelError):
                    planner.plan_initial_image(self.image, "Tidy")

    def test_failure_feedback_and_fatal_stop(self):
        model = FakeModel(ACTION)
        planner = DirectVLMPlanner(model=model)
        output = planner.plan_initial_image(self.image, "Tidy")
        planner.record_execution_result(output.next_subtask, success=False, fatal=True)
        with self.assertRaises(ValueError):
            planner.mark_subtask_completed(output.next_subtask)
        self.assertEqual(planner.update_from_image(self.image, "Tidy").status, "blocked")
        self.assertEqual(len(model.calls), 1)

    def test_budget_is_not_completion(self):
        planner = DirectVLMPlanner(model=FakeModel(ACTION), max_steps=1)
        output = planner.plan_initial_image(self.image, "Tidy")
        planner.record_execution_result(output.next_subtask, success=False)
        blocked = planner.update_from_image(self.image, "Tidy")
        self.assertFalse(blocked.task_completed)
        self.assertEqual(blocked.reason, "step_limit")

    def test_reset_clears_history(self):
        planner = DirectVLMPlanner(model=FakeModel(ACTION, ACTION))
        output = planner.plan_initial_image(self.image, "Tidy")
        planner.record_execution_result(output.next_subtask, success=True)
        planner.plan_initial_image(self.image, "New task")
        self.assertEqual(planner.history, [])
        self.assertEqual(planner.decision_count, 1)

    def test_no_graph_planning_is_invoked(self):
        with patch("high_level_planner.scene_graph.constructor.SceneGraphConstructor.construct",
                   side_effect=AssertionError("Perception must not be invoked")), \
             patch("high_level_planner.reasoning_graph.generator.ReasoningGraphGenerator.generate",
                   side_effect=AssertionError("Graph generation must not be invoked")), \
             patch("high_level_planner.planner.planner.GraphSearchPlanner.plan",
                   side_effect=AssertionError("Search must not be invoked")):
            result = DirectVLMPlanner(model=FakeModel(ACTION)).plan_initial_image(self.image, "Tidy")
            self.assertEqual(result.status, "action")

    def test_nonfatal_failure_is_given_to_next_decision(self):
        model = FakeModel(ACTION, ACTION)
        planner = DirectVLMPlanner(model=model)
        output = planner.plan_initial_image(self.image, "Tidy")
        planner.record_execution_result(output.next_subtask, success=False,
                                        details={"error": "grasp failed"})
        planner.update_from_image(self.image, "Tidy")
        self.assertIn("grasp failed", model.calls[-1][0])
        self.assertEqual(planner.metrics["execution_failures"], 1)

    def test_image_types_and_invalid_rgb(self):
        from PIL import Image
        from baseline.direct_vlm_planner import image_state
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            Image.fromarray(self.image).save(path)
            self.assertTrue(image_state(path).as_image_url().startswith("data:image/png;base64,"))
            self.assertEqual(image_state(Image.fromarray(self.image)).as_image_url(),
                             image_state(self.image).as_image_url())
        with self.assertRaises(ValueError):
            image_state(self.image.astype(float))


if __name__ == "__main__":
    unittest.main()
