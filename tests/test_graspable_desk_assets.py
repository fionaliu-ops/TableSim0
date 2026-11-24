from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))

import generate_messy_desk_assets as assets
from render_messy_desk_objects import discover_variants, load_specs


def inside_convex(mesh: trimesh.Trimesh, point: tuple[float, float, float]) -> bool:
    offsets = np.asarray(point) - mesh.triangles_center
    return bool(np.all(np.einsum("ij,ij->i", offsets, mesh.face_normals) <= 1e-9))


class GraspableDeskAssetTests(unittest.TestCase):
    def test_folder_is_thick_upright_and_closed(self) -> None:
        visual, collision = assets.graspable_folder()
        center, extents = assets.merge_bounds(collision)
        np.testing.assert_allclose(extents, [0.055, 0.310, 0.260])
        self.assertAlmostEqual(center[1] - extents[1] / 2, 0.0)
        self.assertTrue(all(mesh.is_watertight and mesh.is_convex for mesh in collision))
        self.assertGreater(len(visual), 10)
        design = assets.graspable_folder_metadata()["grasp_design"]
        self.assertLess(design["closing_width_m"], design["candidate_opening_m"])
        self.assertLess(design["candidate_opening_m"], 0.09)

    def test_keyboard_keeps_real_underbody_clearance_and_support_contacts(self) -> None:
        _, collision = assets.supported_keyboard()
        self.assertEqual(len(collision), 4)  # body, key proxy, two rear feet
        self.assertTrue(all(mesh.is_watertight and mesh.is_convex for mesh in collision))
        _, extents = assets.merge_bounds(collision)
        self.assertAlmostEqual(extents[0], 0.300)
        self.assertLess(extents[2], 0.155)
        self.assertLess(extents[1], 0.065)
        body = collision[0]
        self.assertAlmostEqual(body.bounds[0, 1], 0.0)
        touching = body.vertices[np.isclose(body.vertices[:, 1], 0.0)]
        self.assertEqual(len(touching), 2)  # front lower edge, not a floating body
        self.assertTrue(np.all(touching[:, 2] > 0.070))
        for foot in collision[2:]:
            self.assertAlmostEqual(foot.bounds[0, 1], 0.0)
            self.assertLess(foot.bounds[1, 2], -0.050)  # no front feet
            self.assertTrue(any(inside_convex(collision[0], vertex) for vertex in foot.vertices))
        # Actual 3 mm fingertip + 3 mm safety margin, inserted 20 mm from
        # either short side near the raised rear. The front is deliberately
        # NOT a grasp corridor anymore: its lower edge touches the table.
        for side in (-1, 1):
            for x in np.linspace(0.130, 0.150, 5) * side:
                for z in np.linspace(-0.043, -0.027, 5):
                    for y in (0.003, 0.006, 0.012):
                        self.assertFalse(any(inside_convex(mesh, (x, y, z)) for mesh in collision))
        self.assertGreater(assets.keyboard_lower_surface(0), 0.006)
        self.assertGreater(assets.keyboard_lower_surface(-0.035), 0.014)

    def test_keyboard_support_and_grasp_metadata_match_rear_only_design(self) -> None:
        metadata = assets.supported_keyboard_metadata()
        design = metadata["grasp_design"]
        self.assertEqual(design["support_layout"], "rear_only_front_edge_contact")
        self.assertEqual(design["front_underbody_clearance_m"], 0.0)
        self.assertGreater(design["rear_underbody_clearance_m"], 0.020)
        inverse = np.linalg.inv(assets.keyboard_transform())
        for frame in metadata["contact_points_pose"]:
            local = inverse @ np.asarray(frame)
            self.assertAlmostEqual(local[2, 3], assets.KEYBOARD_GRASP_LOCAL_Z)
            self.assertLess(local[2, 3], 0)

    def test_grasp_frames_match_geometry_and_are_marked_unverified(self) -> None:
        for metadata in (assets.graspable_folder_metadata(), assets.supported_keyboard_metadata()):
            self.assertFalse(metadata["grasp_design"]["physics_grasp_verified"])
            for frame in metadata["contact_points_pose"]:
                rotation = np.asarray(frame)[:3, :3]
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

    def test_keyboard_legends_reference_correct_atlas_row(self) -> None:
        visual, _ = assets.supported_keyboard()
        legends = [mesh for mesh in visual if mesh.visual.uv is not None]
        self.assertEqual(len(legends), 75)
        np.testing.assert_allclose(legends[0].visual.uv.mean(axis=0), [1/32, 1-1/12])
        image = np.asarray(legends[0].visual.material.baseColorTexture)
        self.assertLess(image[:64, :64, :3].min(), 100)

    def test_export_preserves_convex_parts_and_matches_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(assets, "OBJECTS_DIR", root):
                for asset in assets.ASSETS:
                    if asset.number not in (136, 137):
                        continue
                    result = assets.build_asset(asset, False)
                    path = root / asset.directory_name
                    scene = trimesh.load_scene(path / "collision" / "base0.glb")
                    self.assertEqual(len(scene.geometry), result["collision_parts"])
                    self.assertTrue(all(mesh.is_watertight for mesh in scene.geometry.values()))
                    metadata = json.loads((path / "model_data0.json").read_text())
                    np.testing.assert_allclose(scene.bounds.mean(axis=0), metadata["center"], atol=1e-7)
                    np.testing.assert_allclose(scene.extents, metadata["extents"], atol=1e-7)
                    visual_scene = trimesh.load_scene(path / "visual" / "base0.glb")
                    np.testing.assert_allclose(visual_scene.bounds, scene.bounds, atol=0.0021)
                    # Rebuilding must not delete unrelated data in an asset.
                    marker = path / "user_notes.txt"
                    marker.write_text("keep", encoding="utf-8")
                    assets.build_asset(asset, True)
                    self.assertEqual(marker.read_text(), "keep")
                    with self.assertRaises(FileExistsError):
                        assets.build_asset(asset, False)

    def test_dataset_mapping_uses_only_new_assets(self) -> None:
        specs = load_specs()
        for label, source in (("folder", "136_file-folder"), ("keyboard", "137_keyboard-supported")):
            entries = [spec for spec in specs if label in spec["objects"]]
            self.assertEqual([spec["source"] for spec in entries], [source])
            # Binary assets are generated locally, not downloaded on test hosts.
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                matching = next(asset for asset in assets.ASSETS if asset.directory_name == source)
                with patch.object(assets, "OBJECTS_DIR", root):
                    assets.build_asset(matching, False)
                with patch("render_messy_desk_objects.ASSETS_ROOT", root):
                    self.assertEqual([variant.display_id for variant in discover_variants(entries[0])], ["0"])


if __name__ == "__main__":
    unittest.main()
