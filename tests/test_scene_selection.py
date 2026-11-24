from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from batch_generate_scenes import _complete_batch, derive_seed, discover_manifests  # noqa: E402
import scence_generator as generator  # noqa: E402
from scence_generator import LoadedObject, collides, rectangles_overlap, sample_scene_labels  # noqa: E402
from scene_selection import SceneFilterConfig, evaluate_scene, select_diverse  # noqa: E402


def make_object(label: str, x: float, y: float) -> dict:
    return {
        "label": label,
        "asset": f"asset_{label}",
        "source": f"source_{label}",
        "kind": "rigid",
        "model_id": "0",
        "xy": [x, y],
        "z_offset": 0.0,
        "yaw_degrees": 0.0,
        "footprint": [0.02, 0.02],
        "height": 0.02,
        "relation": None,
    }


def make_scene() -> dict:
    positions = [
        ("mouse", 0.0, -0.10),
        ("notebook", 0.20, 0.10),
        ("pen_support", 0.30, -0.10),
        ("cup", -0.20, 0.10),
        ("folder", -0.30, -0.10),
        ("keyboard", 0.10, 0.10),
        ("phone", -0.10, 0.10),
        ("plant", 0.20, -0.10),
        ("lamp", -0.20, -0.10),
    ]
    return {
        "image": "scene_000.png",
        "placed_objects": [make_object(*item) for item in positions],
        "on_relationships": [
            {"subject": "pen_support", "predicate": "on", "object": "notebook"}
        ],
        "occlusion": {"occluder": "cup", "target": "folder"},
        "layout_policy": {"valid": True, "violations": []},
        "skipped": [],
        "resolution": [64, 64],
    }


def write_test_image(path: Path) -> None:
    image = Image.new("L", (64, 64))
    image.putdata([18 + ((x * 17 + y * 31) % 220) for y in range(64) for x in range(64)])
    image.convert("RGB").save(path)


def loaded_object(label: str, footprint: tuple[float, float], yaw: float = 0.0) -> LoadedObject:
    return LoadedObject(
        label=label,
        asset_output=label,
        source=label,
        kind="rigid",
        variant_id="0",
        wrapper=None,
        centered_pose=None,
        footprint=np.asarray(footprint, dtype=float),
        height=0.02,
        radius=float(np.linalg.norm(footprint) / 2),
        yaw=yaw,
    )


class SceneSelectionTests(unittest.TestCase):
    def test_sampling_does_not_force_relationship_or_cooccurrence_endpoints(self) -> None:
        weights = {label: 1.0 for label in ("cup", "notebook", "pen", "mouse", "phone", "book")}
        messy = {
            "objects": weights,
            "relationship": [{"subject": "pen", "predicate": "on", "object": "notebook", "frequency": 1.0}],
            "co-occurence": [{"objects": ["cup", "notebook"], "frequency": 1.0}],
        }
        counts = Counter()
        for seed in range(200):
            labels, relations, pairs, occlusion = sample_scene_labels(
                np.random.default_rng(seed), messy, set(weights), 3, 3,
            )
            self.assertEqual(len(set(labels)), 3)
            self.assertEqual((relations, pairs, occlusion), ([], [], None))
            counts.update(labels)
        for label in weights:
            self.assertGreater(counts[label], 0)
            self.assertLess(counts[label], 200)

    def test_sampling_is_reproducible_and_honors_positive_weights(self) -> None:
        weights = {"cup": 20.0, "notebook": 1.0, "mouse": 1.0, "pen": 0.0, "invalid": float("nan")}
        messy = {"objects": weights}
        first = sample_scene_labels(np.random.default_rng(42), messy, set(weights), 2, 2)
        self.assertEqual(first, sample_scene_labels(np.random.default_rng(42), messy, set(weights), 2, 2))
        counts = Counter()
        for seed in range(200):
            labels, *_ = sample_scene_labels(np.random.default_rng(seed), messy, set(weights), 1, 1)
            counts.update(labels)
        self.assertNotIn("pen", counts)
        self.assertNotIn("invalid", counts)
        self.assertGreater(counts["cup"], counts["notebook"] + counts["mouse"])

    def test_sampling_respects_count_and_single_rear_device(self) -> None:
        weights = {f"item_{i}": 1.0 for i in range(12)}
        weights.update({"laptop": 1.0, "monitor": 1.0, "printer": 1.0})
        for seed in range(200):
            labels, *_ = sample_scene_labels(np.random.default_rng(seed), {"objects": weights}, set(weights), 9, 11)
            self.assertTrue(9 <= len(labels) <= 11)
            self.assertEqual(len(set(labels)), len(labels))
            self.assertLessEqual(len(set(labels) & generator.REAR_CENTER_DEVICE_LABELS), 1)
        with self.assertRaisesRegex(ValueError, "positive-weight categories"):
            sample_scene_labels(np.random.default_rng(0), {"objects": {"cup": 1.0}}, {"cup"}, 9, 11)

    def test_generation_without_cup_notebook_or_forced_relations(self) -> None:
        # Exercise real sampling and collision-safe placement, stubbing only
        # simulator/asset/render operations unavailable on CPU test hosts.
        labels = {f"item_{i}": [{}] for i in range(9)}
        args = SimpleNamespace(min_objects=9, max_objects=9, min_placed=9,
                               render_steps=0, width=128, height=128, output_dir=Path("unused"),
                               allow_missing_on=False, allow_missing_occlusion=False)

        def place(obj, xy, z_offset, yaw):
            obj.xy = xy
            obj.z_offset = z_offset
            obj.yaw = yaw

        with ExitStack() as stack:
            for name in ("add_table_scene", "add_camera", "render_camera"):
                stack.enter_context(patch.object(generator, name))
            stack.enter_context(patch.object(generator, "choose_object", side_effect=lambda rng, label, *rest: label))
            stack.enter_context(patch.object(generator, "load_object", side_effect=lambda scene, label: loaded_object(label, (0.02, 0.02))))
            stack.enter_context(patch.object(generator, "place_loaded_object", side_effect=place))
            stack.enter_context(patch.object(generator, "validate_physical_layout", return_value={"valid": True}))
            reapply = stack.enter_context(patch.object(generator, "reapply_on_relations"))
            relation_pose = stack.enter_context(patch.object(generator, "relation_candidate"))
            record = generator._generate_scene_in_scene(
                MagicMock(), 0, np.random.default_rng(42), args,
                {"objects": {label: 1.0 for label in labels}}, labels, {},
            )
        reapply.assert_not_called()
        relation_pose.assert_not_called()
        self.assertEqual(len(record["placed_objects"]), 9)
        self.assertEqual(record["on_relationships"], [])
        self.assertIsNone(record["occlusion"])
        self.assertEqual(record["occlusion_status"], "not_measured")
        self.assertEqual(record["sampling_policy"], generator.SAMPLING_POLICY)
        self.assertTrue(all(obj["relation"] is None for obj in record["placed_objects"]))

    def test_oriented_footprints_detect_intersection_without_circle_false_positive(self) -> None:
        first = loaded_object("first", (0.20, 0.02))
        second = loaded_object("second", (0.20, 0.02))
        self.assertTrue(
            rectangles_overlap(first, np.asarray([0.0, 0.0]), second, np.asarray([0.0, 0.01]))
        )
        self.assertFalse(
            rectangles_overlap(first, np.asarray([0.0, 0.0]), second, np.asarray([0.0, 0.04]))
        )

    def test_xy_overlap_is_allowed_when_objects_are_separated_vertically(self) -> None:
        lower = loaded_object("lower", (0.20, 0.20))
        upper = loaded_object("upper", (0.05, 0.05))
        lower.xy = np.asarray([0.0, 0.0])
        lower.z_offset = 0.0
        placed = {"lower": lower}
        self.assertFalse(collides(upper, np.asarray([0.0, 0.0]), placed, z_offset=0.03))
        self.assertTrue(collides(upper, np.asarray([0.0, 0.0]), placed, z_offset=0.01))

    def test_valid_scene_is_accepted_with_reachable_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "scene_000.png"
            write_test_image(image_path)
            result = evaluate_scene(make_scene(), image_path, SceneFilterConfig())

        self.assertTrue(result["accepted"], result["reasons"])
        self.assertIn("mouse", result["eligible_target_labels"])
        self.assertGreater(result["score"], 80)

    def test_missing_required_relation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "scene_000.png"
            write_test_image(image_path)
            scene = make_scene()
            scene["on_relationships"] = []
            result = evaluate_scene(scene, image_path, SceneFilterConfig(require_on=True))

        self.assertFalse(result["accepted"])
        self.assertIn("missing_on_relationship", result["reasons"])

    def test_natural_scene_is_accepted_without_relation_ranking_bonus(self) -> None:
        scene = make_scene()
        config = SceneFilterConfig(validate_images=False)
        original = evaluate_scene(scene, Path("unused"), config)
        scene.update(on_relationships=[], occlusion=None, occlusion_status="not_measured")
        natural = evaluate_scene(scene, Path("unused"), config)
        self.assertTrue(natural["accepted"], natural["reasons"])
        self.assertEqual(original["score"], natural["score"])
        required = evaluate_scene(scene, Path("unused"), SceneFilterConfig(validate_images=False, require_occlusion=True))
        self.assertFalse(required["accepted"])
        self.assertIn("missing_occlusion", required["reasons"])

    def test_manifest_physical_validation_failure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "scene_000.png"
            write_test_image(image_path)
            scene = make_scene()
            scene["physical_validation"] = {
                "valid": False,
                "violations": ["mouse:intersects:cup"],
            }
            result = evaluate_scene(scene, image_path, SceneFilterConfig())

        self.assertFalse(result["accepted"])
        self.assertIn("invalid_physical_layout", result["reasons"])

    def test_explicitly_occluded_only_target_is_not_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "scene_000.png"
            write_test_image(image_path)
            scene = make_scene()
            scene["occlusion"] = {"occluder": "cup", "target": "mouse"}
            config = SceneFilterConfig(target_labels=("mouse",))
            result = evaluate_scene(scene, image_path, config)

        self.assertFalse(result["accepted"])
        self.assertIn("no_eligible_target_candidate", result["reasons"])
        mouse = result["target_candidates"][0]
        self.assertIn("explicit_occlusion_target", mouse["reasons"])

    def test_diverse_selection_prefers_novel_labels(self) -> None:
        def result(scene_id: str, score: float, labels: list[str]) -> dict:
            return {
                "scene_id": scene_id,
                "score": score,
                "accepted": True,
                "scene": {"placed_objects": [{"label": label} for label in labels]},
            }

        selected = select_diverse(
            [
                result("a", 90.0, ["mouse", "cup"]),
                result("b", 89.0, ["mouse", "cup", "pen"]),
                result("c", 88.0, ["book", "lamp"]),
            ],
            limit=2,
            diversity_bonus=8.0,
        )
        self.assertEqual([item["scene_id"] for item in selected], ["a", "c"])

    def test_manifest_discovery_ignores_pipeline_selection_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / "batches" / "batch_00000"
            selected = root / "selected"
            batch.mkdir(parents=True)
            selected.mkdir()
            payload = {"generator": "script/scence_generator.py", "scenes": []}
            (batch / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            (selected / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

            found = discover_manifests(root)

        self.assertEqual(found, [batch / "manifest.json"])

    def test_seed_derivation_supports_independent_batch_ranges(self) -> None:
        self.assertEqual(derive_seed(20260919, 0, 1_000_003), 20260919)
        self.assertEqual(derive_seed(20260919, 1, 1_000_003), 21260922)
        self.assertNotEqual(
            derive_seed(20260919, 10, 1_000_003),
            derive_seed(20260919, 11, 1_000_003),
        )

    def test_natural_resume_refuses_legacy_staged_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = {"seed": 123, "scenes": [make_scene()], "summary": {"rendered_scenes": 1}}
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(_complete_batch(path, 123, 1))
            with self.assertRaisesRegex(ValueError, "new output directory"):
                _complete_batch(path, 123, 1, require_natural=True)
            manifest["generation_config"] = {"forced_relationships": False, "forced_occlusion": False}
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(_complete_batch(path, 123, 1, require_natural=True))

    def test_filter_only_cli_writes_portable_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input" / "batch_00000"
            output_dir = root / "output"
            input_dir.mkdir(parents=True)
            scene = make_scene()
            scene.update(on_relationships=[], occlusion=None, occlusion_status="not_measured")
            write_test_image(input_dir / scene["image"])
            manifest = {
                "generator": "script/scence_generator.py",
                "seed": 123,
                "scenes": [scene],
                "failures": [],
                "summary": {"rendered_scenes": 1},
            }
            (input_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "batch_generate_scenes.py"),
                    "--filter-only",
                    "--input-dir",
                    str(root / "input"),
                    "--output-dir",
                    str(output_dir),
                    "--no-review-sheets",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            selection = json.loads(
                (output_dir / "selected" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(selection["summary"]["selected_scenes"], 1)
            selected_image = output_dir / selection["scenes"][0]["selected_image"]
            self.assertTrue(selected_image.is_file())


if __name__ == "__main__":
    unittest.main()
