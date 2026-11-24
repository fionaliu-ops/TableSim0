import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from script.object_motion_validation import interpolated_poses, run_video_path


class ObjectMotionValidationTest(unittest.TestCase):
    def test_continuous_translation_and_quaternion_endpoints(self):
        start = np.array([0., 0., .7, 1, 0, 0, 0])
        end = np.array([.3, .2, .8, 0, 0, 0, 1.])
        samples = np.array(list(interpolated_poses(start, end, .004)))
        positions = np.vstack((start[:3], samples[:, :3]))
        self.assertLessEqual(np.linalg.norm(np.diff(positions, axis=0), axis=1).max(), .12*.004+1e-8)
        np.testing.assert_allclose(samples[-1], end)
        np.testing.assert_allclose(np.linalg.norm(samples[:, 3:], axis=1), 1)

    def test_quaternion_sign_does_not_cause_rotation_jump(self):
        start = [0, 0, .7, 1, 0, 0, 0]
        finish = [0, 0, .7, -1, 0, 0, 0]
        samples = np.array(list(interpolated_poses(start, finish, .004)))
        np.testing.assert_allclose(samples[:, 3], 1)

    def test_visual_failure_does_not_truncate_or_certify_complete_video(self):
        with tempfile.TemporaryDirectory() as directory:
            env = SimpleNamespace(output_dir=Path(directory), last_placement={'mode':'object_motion'},
                                  step=Mock(return_value={'image':'actual_camera.png'}))
            edges = [SimpleNamespace(from_state='S0', to_state='S1'),
                     SimpleNamespace(from_state='S1', to_state='GOAL')]
            request = SimpleNamespace(to_dict=lambda: {'instruction':'model command'})
            planner = SimpleNamespace(vla_interface=SimpleNamespace(format_subtask=Mock(return_value=request)),
                scene_updater=SimpleNamespace(update_scene_graph=Mock(side_effect=ValueError('ambiguous visual identity'))))
            report = SimpleNamespace(terminal=False, to_dict=lambda: {'terminal':False})
            evaluator = Mock(); evaluator.evaluate.return_value = report
            with patch('high_level_planner.reasoning_graph.target_conditioned.TargetGraphEvaluator', return_value=evaluator):
                result = run_video_path(env, planner, object(), object(), SimpleNamespace(metadata={}),
                                        SimpleNamespace(path=edges,goal_state='GOAL'), 'match image')
            self.assertEqual(env.step.call_count, 2)
            self.assertEqual(result.status, 'object_motion_video_complete_needs_visual_review')
            self.assertIn('ambiguous visual identity', (Path(directory)/'dynamic_trace.json').read_text())
            saved = json.loads((Path(directory)/'dynamic_trace.json').read_text())
            self.assertFalse(saved['final_visual_observation_available'])
            self.assertIsNone(saved['final_visual_goal_report'])
            evaluator.evaluate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
