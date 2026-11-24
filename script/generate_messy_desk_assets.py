#!/usr/bin/env python3
"""Generate the first procedural asset batch for the Messy Desk dataset.

The meshes use metres and a Y-up local coordinate system to match the existing
RoboTwin/TableSim GLB assets. Visual meshes contain simple PBR materials;
collision meshes deliberately use fewer, closed primitives.

Run this script inside the TableSim Docker environment:

    python script/generate_messy_desk_assets.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OBJECTS_DIR = ROOT / "assets" / "objects"

WHITE = (245, 243, 235, 255)
PAPER_WHITE = (252, 250, 240, 255)
BLACK = (30, 33, 38, 255)
DARK_GREY = (58, 62, 69, 255)
GREY = (135, 140, 148, 255)
LIGHT_GREY = (205, 209, 214, 255)
METAL = (175, 181, 188, 255)
BLUE = (58, 118, 190, 255)
SCREEN_BLUE = (61, 143, 181, 255)
RED = (196, 55, 52, 255)
YELLOW = (247, 214, 65, 255)
ORANGE = (232, 112, 37, 255)
GREEN = (69, 150, 99, 255)
PURPLE = (112, 76, 156, 255)
BROWN = (142, 88, 45, 255)
WOOD = (218, 164, 84, 255)


@dataclass
class Asset:
    number: int
    name: str
    frequency: float
    build: Callable[[], tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]]
    metadata: Callable[[], dict] | None = None

    @property
    def directory_name(self) -> str:
        return f"{self.number:03d}_{self.name}"


def colour(mesh: trimesh.Trimesh, rgba: tuple[int, int, int, int]) -> trimesh.Trimesh:
    # SAPIEN's GLTF renderer reliably consumes PBR material base colours, while
    # vertex-only colours are ignored by some Vulkan shader configurations.
    material = trimesh.visual.material.PBRMaterial(
        name="rgba_" + "_".join(str(channel) for channel in rgba),
        baseColorFactor=np.asarray(rgba, dtype=np.uint8),
        metallicFactor=0.0,
        roughnessFactor=0.72,
    )
    mesh.visual = trimesh.visual.texture.TextureVisuals(material=material)
    return mesh


def transform_mesh(
    mesh: trimesh.Trimesh,
    center=(0.0, 0.0, 0.0),
    axis=(0.0, 1.0, 0.0),
    angle=0.0,
) -> trimesh.Trimesh:
    if angle:
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(angle, axis, point=(0.0, 0.0, 0.0))
        )
    mesh.apply_translation(np.asarray(center, dtype=float))
    return mesh


def box(
    extents,
    center=(0.0, 0.0, 0.0),
    rgba=GREY,
    axis=(0.0, 1.0, 0.0),
    angle=0.0,
) -> trimesh.Trimesh:
    return colour(
        transform_mesh(
            trimesh.creation.box(extents=np.asarray(extents, dtype=float)),
            center=center,
            axis=axis,
            angle=angle,
        ),
        rgba,
    )


def cylinder_between(
    start,
    end,
    radius,
    rgba=GREY,
    sections=20,
) -> trimesh.Trimesh:
    return colour(
        trimesh.creation.cylinder(
            radius=radius,
            segment=np.asarray([start, end], dtype=float),
            sections=sections,
        ),
        rgba,
    )


def cylinder_y(
    radius,
    height,
    center=(0.0, 0.0, 0.0),
    rgba=GREY,
    sections=32,
) -> trimesh.Trimesh:
    return cylinder_between(
        (center[0], center[1] - height / 2, center[2]),
        (center[0], center[1] + height / 2, center[2]),
        radius,
        rgba,
        sections,
    )


def sphere(radius, center=(0.0, 0.0, 0.0), rgba=GREY) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    mesh.apply_translation(center)
    return colour(mesh, rgba)


def cone_between(start, end, radius, rgba=GREY, sections=32) -> trimesh.Trimesh:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    height = float(np.linalg.norm(direction))
    mesh = trimesh.creation.cone(radius=radius, height=height, sections=sections)
    align = trimesh.geometry.align_vectors((0.0, 0.0, 1.0), direction)
    mesh.apply_transform(align)
    mesh.apply_translation(start)
    return colour(mesh, rgba)


def torus_y(
    major_radius,
    minor_radius,
    center=(0.0, 0.0, 0.0),
    rgba=GREY,
    scale=(1.0, 1.0, 1.0),
    major_sections=40,
    minor_sections=12,
) -> trimesh.Trimesh:
    mesh = trimesh.creation.torus(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_sections=major_sections,
        minor_sections=minor_sections,
    )
    # Trimesh creates a torus in the XY plane. Rotate it into the XZ desk plane.
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(math.pi / 2, (1.0, 0.0, 0.0))
    )
    mesh.apply_scale(scale)
    mesh.apply_translation(center)
    return colour(mesh, rgba)


def polyline_tubes(points, radius, rgba=GREY, closed=False) -> list[trimesh.Trimesh]:
    points = list(points)
    pairs = list(zip(points[:-1], points[1:]))
    if closed:
        pairs.append((points[-1], points[0]))
    return [cylinder_between(a, b, radius, rgba, sections=12) for a, b in pairs]


def paper() -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    visual = [box((0.210, 0.0008, 0.297), (0, 0.0004, 0), PAPER_WHITE)]
    for z in (-0.105, -0.075, -0.045, -0.015, 0.015, 0.045, 0.075):
        visual.append(box((0.145, 0.00025, 0.002), (0.005, 0.00093, z), LIGHT_GREY))
    visual.append(box((0.025, 0.00025, 0.002), (-0.075, 0.00093, 0.115), RED))
    collision = [box((0.210, 0.0015, 0.297), (0, 0.00075, 0))]
    return visual, collision


def folder() -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    visual = [
        box((0.240, 0.004, 0.320), (0, 0.002, 0), BLUE),
        box((0.075, 0.002, 0.035), (-0.078, 0.005, 0.142), BLUE),
        box((0.215, 0.001, 0.285), (0.008, 0.0046, -0.006), PAPER_WHITE),
    ]
    collision = [box((0.240, 0.005, 0.320), (0, 0.0025, 0))]
    return visual, collision


# Grasp-oriented replacements use the same metres/Y-up convention as existing
# assets. Keep legacy folder/keyboard models intact for historical tasks.
BINDER_THICKNESS = 0.055
BINDER_HEIGHT = 0.310
BINDER_DEPTH = 0.260
KEYBOARD_WIDTH = 0.300
KEYBOARD_DEPTH = 0.150
KEYBOARD_BODY_THICKNESS = 0.014
KEYBOARD_KEY_HEIGHT = 0.006
KEYBOARD_FRONT_CLEARANCE = 0.0
KEYBOARD_TILT = math.radians(8.0)
KEYBOARD_CENTER_Y = (
    KEYBOARD_FRONT_CLEARANCE
    + math.sin(KEYBOARD_TILT) * KEYBOARD_DEPTH / 2
    + math.cos(KEYBOARD_TILT) * KEYBOARD_BODY_THICKNESS / 2
)
KEYBOARD_REAR_FOOT_Z = -0.063
KEYBOARD_GRASP_LOCAL_Z = -0.035


def graspable_folder() -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    """Closed, filled A4 hard-cover binder, standing on its lower edge."""
    visual = [
        box((0.0025, BINDER_HEIGHT, BINDER_DEPTH), (-0.02625, 0.155, 0), BLUE),
        box((0.0025, BINDER_HEIGHT, BINDER_DEPTH), (0.02625, 0.155, 0), BLUE),
        box((BINDER_THICKNESS, BINDER_HEIGHT, 0.003), (0, 0.155, 0.1285), BLUE),
        box((0.050, 0.294, 0.243), (0, 0.155, -0.004), PAPER_WHITE),
        box((0.038, 0.170, 0.0008), (0, 0.192, 0.1304), WHITE),
        box((0.031, 0.021, 0.0006), (0, 0.257, 0.1311), SCREEN_BLUE),
    ]
    for y in (0.211, 0.200, 0.189):
        visual.append(box((0.025, 0.001, 0.0006), (0, y, 0.1311), GREY))
    # The spine finger hole is a visual recess, not a grasp handle: the solid
    # filled-binder collision proxy is intentionally not advertised as hollow.
    visual.append(cylinder_between((0, 0.060, 0.1301), (0, 0.060, 0.1310), 0.009, BLACK, 32))
    rim = trimesh.creation.torus(major_radius=0.0095, minor_radius=0.001, major_sections=32, minor_sections=8)
    rim.apply_translation((0, 0.060, 0.1310))
    visual.append(colour(rim, METAL))
    for y in np.linspace(0.020, 0.291, 16):
        visual.append(box((0.050, 0.0004, 0.0004), (0, y, -0.1257), LIGHT_GREY))
    collision = [box((BINDER_THICKNESS, BINDER_HEIGHT, BINDER_DEPTH), (0, 0.155, 0))]
    return visual, collision


def keyboard_transform() -> np.ndarray:
    matrix = trimesh.transformations.rotation_matrix(KEYBOARD_TILT, (1.0, 0.0, 0.0))
    matrix[1, 3] = KEYBOARD_CENTER_Y
    return matrix


def keyboard_lower_surface(z: float) -> float:
    """Height of the pitched body's lower plane at a local depth coordinate."""
    return KEYBOARD_CENTER_Y - math.tan(KEYBOARD_TILT) * z - KEYBOARD_BODY_THICKNESS / (2 * math.cos(KEYBOARD_TILT))


def supported_keyboard() -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    """Rear-supported keyboard: front lower edge rests directly on the table."""
    pitch = keyboard_transform()
    body = box((KEYBOARD_WIDTH, KEYBOARD_BODY_THICKNESS, KEYBOARD_DEPTH), rgba=DARK_GREY)
    body.apply_transform(pitch)
    visual = [body]
    collision = [body.copy()]

    rows = [
        [(name, 1.0) for name in ["Esc"] + [f"F{i}" for i in range(1, 13)] + ["Del"]],
        [(name, 1.0) for name in ["`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "="]] + [("Back", 2.0)],
        [("Tab", 1.5)] + [(name, 1.0) for name in "QWERTYUIOP[]"] + [("\\", 1.5)],
        [("Caps", 1.75)] + [(name, 1.0) for name in "ASDFGHJKL;'"] + [("Enter", 2.25)],
        [("Shift", 2.25)] + [(name, 1.0) for name in "ZXCVBNM,./"] + [("Shift", 2.75)],
        [("Ctrl", 1.25), ("Win", 1.25), ("Alt", 1.25), ("Space", 6.25), ("Alt", 1.25), ("Fn", 1.25), ("Ctrl", 1.25), ("<", 1.25)],
    ]
    # One embedded label atlas; no external image files or fonts are required.
    tile = 64
    atlas = Image.new("RGBA", (16 * tile, 6 * tile), WHITE)
    draw = ImageDraw.Draw(atlas)
    font = ImageFont.load_default()
    label_material = trimesh.visual.material.PBRMaterial(
        name="key_legends", baseColorTexture=atlas, metallicFactor=0.0, roughnessFactor=0.8,
    )
    for row_index, row in enumerate(rows):
        total = sum(units for _, units in row) * 0.019
        x = -total / 2
        z = -0.054 + row_index * 0.0216
        for column, (legend, units) in enumerate(row):
            width = units * 0.019 - 0.002
            center_x = x + units * 0.019 / 2
            x += units * 0.019
            key = box((width, KEYBOARD_KEY_HEIGHT, 0.0185), (center_x, 0.010, z), WHITE)
            key.apply_transform(pitch)
            visual.append(key)
            text_bounds = draw.textbbox((0, 0), legend, font=font)
            text_x = column * tile + (tile - (text_bounds[2] - text_bounds[0])) / 2
            text_y = row_index * tile + (tile - (text_bounds[3] - text_bounds[1])) / 2 - text_bounds[1]
            draw.text((text_x, text_y), legend, font=font, fill=BLACK)
            u0, u1 = column / 16, (column + 1) / 16
            v0, v1 = row_index / 6, (row_index + 1) / 6
            # Order gives an outward +Y normal. Trimesh expects bottom-origin
            # UVs and flips V on glTF export; atlas pixel rows start at the top.
            label = trimesh.Trimesh(
                vertices=[(center_x-width/2, 0.01305, z-0.00925),
                          (center_x-width/2, 0.01305, z+0.00925),
                          (center_x+width/2, 0.01305, z+0.00925),
                          (center_x+width/2, 0.01305, z-0.00925)],
                faces=[(0, 1, 2), (0, 2, 3)], process=False,
            )
            label.visual = trimesh.visual.texture.TextureVisuals(
                uv=np.asarray([(u0, 1-v0), (u0, 1-v1), (u1, 1-v1), (u1, 1-v0)]), material=label_material,
            )
            label.apply_transform(pitch)
            visual.append(label)
    # A shallow cap-level collision proxy matches the key tops but never fills
    # the space below the body. Preserve each primitive as a distinct GLB mesh
    # so add_multiple_convex_collisions_from_file does not build one large hull.
    key_collision = box((0.285, KEYBOARD_KEY_HEIGHT, 0.127), (0, 0.010, 0))
    key_collision.apply_transform(pitch)
    collision.append(key_collision)
    # Only the rear has feet. Front support comes from the body's own lower
    # edge at Y=0; do not float the whole housing or add front feet. The void
    # beneath the inclined body remains triangular in a side view.
    z = KEYBOARD_REAR_FOOT_Z
    for x in (-0.103, 0.103):
        height = keyboard_lower_surface(z - 0.006) + 0.001
        foot = box((0.022, height, 0.012), (x, height / 2, z), BLACK)
        visual.append(foot)
        collision.append(foot.copy())
    return visual, collision


def graspable_folder_metadata() -> dict:
    # Tool X enters downward; tool Y closes across the 55 mm binder spine.
    frame = np.asarray([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    contact = np.eye(4)
    contact[:3, :3] = frame
    contact[:3, 3] = [0, 0.270, 0.065]
    return {
        "contact_points_pose": [contact.tolist()],
        "contact_points_description": ["Candidate top approach; jaws close across binder thickness."],
        "grasp_design": {
            "schema": "graspable_desk_asset_v1", "type": "filled_hard_cover_binder",
            "placement": "upright", "closing_width_m": BINDER_THICKNESS,
            "candidate_opening_m": 0.065, "physics_grasp_verified": False,
        },
    }


def supported_keyboard_metadata() -> dict:
    pitch = keyboard_transform()
    contacts = []
    for sign in (-1., 1.):
        contact = np.eye(4)
        # Enter a short side nearer the raised rear, away from the feet. The
        # front edge intentionally has no room for an underbody finger.
        contact[:3, :3] = np.diag([-sign, 1., -sign])
        contact[:3, 3] = [sign * 0.145, 0.003, KEYBOARD_GRASP_LOCAL_Z]
        contacts.append((pitch @ contact).tolist())
    return {
        "contact_points_pose": contacts,
        "contact_points_description": ["Left raised-side top/bottom pinch candidate.", "Right raised-side top/bottom pinch candidate."],
        "grasp_design": {
            "schema": "graspable_desk_asset_v1", "type": "keyboard_with_rear_feet",
            "support_layout": "rear_only_front_edge_contact",
            "tilt_degrees": 8.0, "front_underbody_clearance_m": KEYBOARD_FRONT_CLEARANCE,
            "rear_underbody_clearance_m": math.sin(KEYBOARD_TILT) * KEYBOARD_DEPTH,
            "side_center_underbody_clearance_m": keyboard_lower_surface(0.0),
            "side_grasp_depth_local_m": KEYBOARD_GRASP_LOCAL_Z,
            "closing_width_m": KEYBOARD_BODY_THICKNESS + KEYBOARD_KEY_HEIGHT,
            "candidate_opening_m": 0.030, "support_attached_to_keyboard": True,
            "physics_grasp_verified": False,
        },
    }


def sticky_note() -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    visual = [
        box((0.076, 0.008, 0.076), (0, 0.004, 0), YELLOW),
        box((0.071, 0.0004, 0.071), (0.004, 0.0083, -0.003), (255, 232, 94, 255)),
    ]
    for offset in (-0.0025, 0.0, 0.0025):
        visual.append(box((0.070, 0.00025, 0.0008), (0, 0.003 + offset, 0.0382), WHITE))
    collision = [box((0.076, 0.008, 0.076), (0, 0.004, 0))]
    return visual, collision


def pencil() -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    visual = [
        cylinder_between((-0.085, 0.004, 0), (0.075, 0.004, 0), 0.004, YELLOW, 6),
        cone_between((0.075, 0.004, 0), (0.095, 0.004, 0), 0.0042, WOOD, 12),
        cone_between((0.091, 0.004, 0), (0.0955, 0.004, 0), 0.0015, BLACK, 12),
        cylinder_between((-0.093, 0.004, 0), (-0.085, 0.004, 0), 0.0042, METAL, 16),
        cylinder_between((-0.103, 0.004, 0), (-0.093, 0.004, 0), 0.0041, (229, 117, 135, 255), 16),
    ]
    collision = [box((0.198, 0.008, 0.008), (-0.004, 0.004, 0))]
    return visual, collision


def scissors() -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    visual = [
        box((0.012, 0.003, 0.145), (-0.014, 0.004, 0.020), METAL, angle=0.18),
        box((0.012, 0.003, 0.145), (0.014, 0.006, 0.020), METAL, angle=-0.18),
        torus_y(0.027, 0.006, (-0.025, 0.008, -0.070), RED, scale=(0.82, 1, 1.15)),
        torus_y(0.027, 0.006, (0.025, 0.008, -0.070), RED, scale=(0.82, 1, 1.15)),
        cylinder_y(0.006, 0.012, (0, 0.007, -0.010), DARK_GREY, 20),
    ]
    collision = [
        box((0.014, 0.006, 0.145), (-0.014, 0.005, 0.020), angle=0.18),
        box((0.014, 0.006, 0.145), (0.014, 0.005, 0.020), angle=-0.18),
        box((0.048, 0.014, 0.060), (-0.025, 0.007, -0.070)),
        box((0.048, 0.014, 0.060), (0.025, 0.007, -0.070)),
    ]
    return visual, collision


def glasses() -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    visual = [
        torus_y(0.027, 0.0024, (-0.034, 0.005, 0), BLACK, scale=(1.25, 1, 0.82)),
        torus_y(0.027, 0.0024, (0.034, 0.005, 0), BLACK, scale=(1.25, 1, 0.82)),
        cylinder_between((-0.008, 0.005, 0), (0.008, 0.005, 0), 0.0022, BLACK, 10),
        cylinder_between((-0.066, 0.005, 0), (-0.073, 0.006, 0.010), 0.0022, BLACK, 10),
        cylinder_between((0.066, 0.005, 0), (0.073, 0.006, 0.010), 0.0022, BLACK, 10),
        cylinder_between((-0.073, 0.006, 0.010), (-0.070, 0.006, 0.115), 0.0020, BLACK, 10),
        cylinder_between((0.073, 0.006, 0.010), (0.070, 0.006, 0.115), 0.0020, BLACK, 10),
    ]
    collision = [
        box((0.142, 0.010, 0.055), (0, 0.005, 0)),
        box((0.145, 0.008, 0.010), (0, 0.006, 0.067)),
    ]
    return visual, collision


def lamp() -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    visual = [
        cylinder_y(0.075, 0.016, (0, 0.008, 0), DARK_GREY),
        cylinder_between((0, 0.016, 0), (-0.035, 0.180, 0), 0.009, METAL),
        cylinder_between((-0.035, 0.180, 0), (0.055, 0.315, 0), 0.008, METAL),
        sphere(0.013, (-0.035, 0.180, 0), DARK_GREY),
        cone_between((0.085, 0.285, 0), (0.055, 0.315, 0), 0.060, BLUE),
        sphere(0.025, (0.075, 0.295, 0), (255, 242, 190, 255)),
    ]
    collision = [
        cylinder_y(0.075, 0.016, (0, 0.008, 0)),
        cylinder_between((0, 0.016, 0), (-0.035, 0.180, 0), 0.011),
        cylinder_between((-0.035, 0.180, 0), (0.055, 0.315, 0), 0.010),
        sphere(0.055, (0.070, 0.300, 0)),
    ]
    return visual, collision


def paper_clip() -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    outer = [
        (-0.025, 0.002, -0.050),
        (-0.035, 0.002, -0.040),
        (-0.035, 0.002, 0.040),
        (-0.025, 0.002, 0.050),
        (0.025, 0.002, 0.050),
        (0.035, 0.002, 0.040),
        (0.035, 0.002, -0.035),
        (0.025, 0.002, -0.045),
    ]
    inner = [
        (0.018, 0.002, -0.025),
        (0.018, 0.002, 0.030),
        (0.010, 0.002, 0.037),
        (-0.015, 0.002, 0.037),
        (-0.020, 0.002, 0.030),
        (-0.020, 0.002, -0.020),
    ]
    visual = polyline_tubes(outer, 0.0021, METAL) + polyline_tubes(inner, 0.0021, METAL)
    collision = [box((0.072, 0.005, 0.102), (0, 0.0025, 0))]
    return visual, collision


def sticker() -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    visual = [
        cylinder_y(0.040, 0.0012, (0, 0.0006, 0), PURPLE, 48),
        cylinder_y(0.026, 0.0003, (0, 0.00135, 0), YELLOW, 40),
        sphere(0.004, (-0.010, 0.004, -0.005), BLACK),
        sphere(0.004, (0.010, 0.004, -0.005), BLACK),
        box((0.020, 0.0004, 0.003), (0, 0.0016, 0.012), RED),
    ]
    collision = [cylinder_y(0.040, 0.002, (0, 0.001, 0), sections=32)]
    return visual, collision


def headphone() -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    arc = []
    for angle in np.linspace(0, math.pi, 13):
        arc.append((0.095 * math.cos(angle), 0.018, 0.105 * math.sin(angle) - 0.035))
    visual = polyline_tubes(arc, 0.006, BLACK)
    visual += [
        box((0.030, 0.024, 0.060), (-0.100, 0.018, -0.040), DARK_GREY),
        box((0.030, 0.024, 0.060), (0.100, 0.018, -0.040), DARK_GREY),
        box((0.020, 0.028, 0.045), (-0.102, 0.018, -0.040), RED),
        box((0.020, 0.028, 0.045), (0.102, 0.018, -0.040), BLUE),
    ]
    collision = [
        box((0.210, 0.024, 0.025), (0, 0.018, 0.050)),
        box((0.035, 0.030, 0.065), (-0.100, 0.018, -0.040)),
        box((0.035, 0.030, 0.065), (0.100, 0.018, -0.040)),
    ]
    return visual, collision


def printer() -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    visual = [
        box((0.380, 0.170, 0.300), (0, 0.085, 0), LIGHT_GREY),
        box((0.320, 0.035, 0.125), (0, 0.183, -0.025), WHITE),
        box((0.235, 0.012, 0.165), (0, 0.202, 0.030), PAPER_WHITE),
        box((0.245, 0.018, 0.035), (0, 0.176, -0.155), DARK_GREY),
        box((0.105, 0.008, 0.045), (0.110, 0.205, -0.090), BLACK),
        cylinder_y(0.008, 0.010, (0.160, 0.210, -0.090), GREEN, 20),
    ]
    collision = [
        box((0.380, 0.170, 0.300), (0, 0.085, 0)),
        box((0.320, 0.045, 0.125), (0, 0.192, -0.025)),
    ]
    return visual, collision


def tablet() -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    visual = [
        box((0.185, 0.008, 0.255), (0, 0.004, 0), DARK_GREY),
        box((0.168, 0.001, 0.225), (0, 0.0085, 0), SCREEN_BLUE),
        cylinder_y(0.003, 0.001, (0, 0.0095, -0.119), BLACK, 16),
        cylinder_y(0.0025, 0.001, (0.075, 0.0095, 0.108), BLACK, 16),
    ]
    collision = [box((0.185, 0.009, 0.255), (0, 0.0045, 0))]
    return visual, collision


def calendar() -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    visual = [
        box((0.190, 0.012, 0.145), (0, 0.006, 0), PAPER_WHITE),
        box((0.190, 0.002, 0.035), (0, 0.013, -0.055), RED),
        box((0.052, 0.001, 0.050), (-0.040, 0.0135, 0.022), BLUE),
        box((0.052, 0.001, 0.050), (0.030, 0.0135, 0.022), YELLOW),
    ]
    for x in np.linspace(-0.075, 0.075, 6):
        visual.append(torus_y(0.008, 0.0015, (x, 0.016, -0.071), METAL, scale=(0.6, 1, 1)))
    collision = [box((0.190, 0.014, 0.145), (0, 0.007, 0))]
    return visual, collision


def rubber_band() -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    visual = [torus_y(0.045, 0.0022, (0, 0.0025, 0), ORANGE, scale=(1.35, 1, 0.72))]
    collision = [box((0.122, 0.005, 0.068), (0, 0.0025, 0))]
    return visual, collision


def magazine() -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    visual = [
        box((0.210, 0.007, 0.297), (0, 0.0035, 0), WHITE),
        box((0.205, 0.001, 0.292), (0, 0.0075, 0), (42, 112, 130, 255)),
        box((0.160, 0.0005, 0.035), (0, 0.0082, -0.115), PAPER_WHITE),
        box((0.135, 0.0005, 0.105), (0, 0.0082, -0.025), ORANGE),
        box((0.160, 0.0005, 0.009), (0, 0.0082, 0.055), YELLOW),
        box((0.120, 0.0005, 0.007), (-0.020, 0.0082, 0.078), PAPER_WHITE),
        box((0.145, 0.0005, 0.007), (-0.008, 0.0082, 0.098), PAPER_WHITE),
    ]
    collision = [box((0.210, 0.008, 0.297), (0, 0.004, 0))]
    return visual, collision


ASSETS = [
    Asset(121, "paper", 0.32, paper),
    Asset(122, "folder", 0.32, folder),
    Asset(123, "sticky-note", 0.26, sticky_note),
    Asset(124, "pencil", 0.26, pencil),
    Asset(125, "scissors", 0.21, scissors),
    Asset(126, "glasses", 0.20, glasses),
    Asset(127, "lamp", 0.11, lamp),
    Asset(128, "paper-clip", 0.08, paper_clip),
    Asset(129, "sticker", 0.07, sticker),
    Asset(130, "headphone", 0.06, headphone),
    Asset(131, "printer", 0.06, printer),
    Asset(132, "tablet", 0.05, tablet),
    Asset(133, "calendar", 0.05, calendar),
    Asset(134, "rubber-band", 0.04, rubber_band),
    Asset(135, "magazine", 0.04, magazine),
    Asset(136, "file-folder", 0.32, graspable_folder, graspable_folder_metadata),
    Asset(137, "keyboard-supported", 0.36, supported_keyboard, supported_keyboard_metadata),
]


def pose_at(position) -> list[list[float]]:
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix.tolist()


def merge_bounds(meshes: list[trimesh.Trimesh]) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray([mesh.bounds for mesh in meshes], dtype=float)
    lower = bounds[:, 0, :].min(axis=0)
    upper = bounds[:, 1, :].max(axis=0)
    return (lower + upper) / 2, upper - lower


def export_glb(meshes: list[trimesh.Trimesh], path: Path, prefix: str) -> None:
    scene = trimesh.Scene()
    for index, mesh in enumerate(meshes):
        scene.add_geometry(mesh, node_name=f"{prefix}_{index:02d}")
    path.write_bytes(scene.export(file_type="glb"))


def preview_asset(asset: Asset, output_dir: Path) -> None:
    """CPU mesh QA only, not a SAPIEN render or proof of a successful grasp."""
    visual, collision = asset.build()
    # Software z-buffer avoids painter-sorting artefacts where a large cover
    # triangle hides labels/keys which are actually in front of it. Orthographic
    # projection keeps both panels at exactly the same scale.
    view = np.array([1.2, -1.8, 1.0])
    view /= np.linalg.norm(view)
    right = np.cross([0., 0., 1.], view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    basis = np.column_stack((right, up, view))
    world_vertices = np.vstack([mesh.vertices[:, [0, 2, 1]] * [1., -1., 1.] for mesh in visual])
    projected = world_vertices @ basis
    lower, upper = projected.min(axis=0), projected.max(axis=0)
    midpoint = (lower + upper) / 2
    size = 720
    scale = (size - 100) / max(upper[0]-lower[0], upper[1]-lower[1])
    panels = []
    light = np.array([0.35, -0.45, 0.82])
    for meshes in (visual, collision):
        pixels = np.full((size, size, 3), 247, dtype=np.uint8)
        depth_buffer = np.full((size, size), -np.inf)
        for mesh in meshes:
            vertices = mesh.vertices[:, [0, 2, 1]] * [1., -1., 1.]
            coordinates = vertices @ basis
            coordinates[:, 0] = size/2 + (coordinates[:, 0]-midpoint[0]) * scale
            coordinates[:, 1] = size/2 - (coordinates[:, 1]-midpoint[1]) * scale
            material = mesh.visual.material
            rgba = material.baseColorFactor
            base = np.asarray(rgba if rgba is not None else WHITE, dtype=float) / 255
            normals = mesh.face_normals[:, [0, 2, 1]] * [1., -1., 1.]
            texture = getattr(material, "baseColorTexture", None)
            texels = np.asarray(texture.convert("RGB")) if texture is not None else None
            uv = mesh.visual.uv
            for face_index, indices in enumerate(mesh.faces):
                if normals[face_index] @ view <= 0:
                    continue
                triangle = coordinates[indices]
                x0, y0 = np.maximum(0, np.floor(triangle[:, :2].min(axis=0))).astype(int)
                x1, y1 = np.minimum(size-1, np.ceil(triangle[:, :2].max(axis=0))).astype(int)
                if x0 > x1 or y0 > y1:
                    continue
                a, b, c = triangle
                denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
                if abs(denominator) < 1e-10:
                    continue
                yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
                xx, yy = xx + 0.5, yy + 0.5
                w0 = ((b[1]-c[1])*(xx-c[0]) + (c[0]-b[0])*(yy-c[1])) / denominator
                w1 = ((c[1]-a[1])*(xx-c[0]) + (a[0]-c[0])*(yy-c[1])) / denominator
                w2 = 1-w0-w1
                depth = w0*a[2] + w1*b[2] + w2*c[2]
                depth_region = depth_buffer[y0:y1+1, x0:x1+1]
                visible = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8) & (depth > depth_region)
                if not visible.any():
                    continue
                shade = 0.5 + 0.5*max(0, normals[face_index] @ light)
                if texels is not None and uv is not None:
                    face_uv = uv[indices]
                    samples = w0[..., None]*face_uv[0] + w1[..., None]*face_uv[1] + w2[..., None]*face_uv[2]
                    tx = np.clip(samples[..., 0]*(texels.shape[1]-1), 0, texels.shape[1]-1).astype(int)
                    ty = np.clip((1-samples[..., 1])*(texels.shape[0]-1), 0, texels.shape[0]-1).astype(int)
                    rgb = texels[ty, tx]*base[:3]*shade
                    pixels[y0:y1+1, x0:x1+1][visible] = np.clip(rgb[visible], 0, 255).astype(np.uint8)
                else:
                    pixels[y0:y1+1, x0:x1+1][visible] = np.clip(base[:3]*shade*255, 0, 255).astype(np.uint8)
                depth_region[visible] = depth[visible]
        panels.append(Image.fromarray(pixels).resize((540, 540), Image.Resampling.LANCZOS))
    image = Image.new("RGB", (1080, 590), (247, 247, 247))
    for index, panel in enumerate(panels):
        image.paste(panel, (index*540, 50))
    draw = ImageDraw.Draw(image)
    draw.text((20, 8), asset.directory_name + " - CPU mesh preview, not SAPIEN", fill="black")
    draw.text((20, 30), "Visual mesh + embedded textures", fill="black")
    draw.text((560, 30), "Collision meshes", fill="black")
    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / f"{asset.directory_name}.png")
    if asset.metadata is not None and asset.metadata().get("grasp_design", {}).get("type") == "keyboard_with_rear_feet":
        # A true orthographic side silhouette makes rear-only support and front
        # edge/table contact visible without perspective hiding either foot.
        profile = Image.new("RGB", (900, 360), (247, 247, 247))
        draw = ImageDraw.Draw(profile)
        draw.text((20, 15), "Keyboard side profile: front edge on table, rear supports only", fill="black")
        scale, floor = 4800, 280
        for index, mesh in enumerate(collision):
            points = np.unique(mesh.vertices[:, [2, 1]], axis=0)
            centered = points - points.mean(axis=0)
            points = points[np.argsort(np.arctan2(centered[:, 1], centered[:, 0]))]
            polygon = [(450-z*scale, floor-y*scale) for z, y in points]
            color = (65, 68, 73) if index == 0 else (220, 219, 211) if index == 1 else (25, 27, 31)
            draw.polygon(polygon, fill=color, outline=(30, 32, 36))
        draw.line((40, floor+1, 860, floor+1), fill=(115, 110, 100), width=3)
        draw.text((70, 310), "FRONT: no foot, no under-edge gap", fill="black")
        draw.text((590, 310), "REAR: ~21 mm underbody lift", fill="black")
        profile.save(output_dir / f"{asset.directory_name}_side.png")


def build_asset(asset: Asset, force: bool) -> dict:
    target = (OBJECTS_DIR / asset.directory_name).resolve()
    if target.parent != OBJECTS_DIR.resolve():
        raise ValueError(f"asset directory must be directly inside {OBJECTS_DIR}: {target}")
    if target.exists():
        if not force:
            raise FileExistsError(f"{target} already exists; use --force to replace it")

    visual, collision = asset.build()
    if not visual or not collision:
        raise ValueError(f"{asset.directory_name} generated an empty mesh list")
    if not all(mesh.is_watertight for mesh in collision):
        raise ValueError(f"{asset.directory_name} has a non-watertight collision primitive")

    # --force only overwrites this generator's known files. Preserve other
    # variants, user annotations and source meshes instead of deleting a tree.
    (target / "visual").mkdir(parents=True, exist_ok=True)
    (target / "collision").mkdir(parents=True, exist_ok=True)
    export_glb(visual, target / "visual" / "base0.glb", "visual")
    export_glb(collision, target / "collision" / "base0.glb", "collision")

    center, extents = merge_bounds(collision)
    top = center.copy()
    top[1] += extents[1] / 2
    model_data = {
        "center": center.tolist(),
        "extents": extents.tolist(),
        "scale": [1.0, 1.0, 1.0],
        "transform_matrix": np.eye(4, dtype=float).tolist(),
        "target_pose": [pose_at(top)],
        "contact_points_pose": [],
        "functional_matrix": [],
        "orientation_point": [],
        "contact_points_group": [],
        "contact_points_mask": [],
        "target_point_description": ["Center point on the top surface."],
        "contact_points_description": [],
        "functional_point_description": [],
        "orientation_point_description": [],
        "stable": True,
    }
    if asset.metadata is not None:
        model_data.update(asset.metadata())
    (target / "model_data0.json").write_text(
        json.dumps(model_data, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    points_info = {
        "contact_points": [
            {"id": index, "description": description}
            for index, description in enumerate(model_data["contact_points_description"])
        ],
        "functional_points": [],
    }
    (target / "points_info.json").write_text(
        json.dumps(points_info, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    return {
        "directory": asset.directory_name,
        "frequency": asset.frequency,
        "extents_m": [round(float(value), 4) for value in extents],
        "visual_parts": len(visual),
        "collision_parts": len(collision),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", action="append", default=[],
        help="Generate only named asset directories; repeat for multiple assets.",
    )
    parser.add_argument(
        "--preview-dir", type=Path, default=None,
        help="Optional CPU visual/collision mesh previews, without SAPIEN.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite generated base0/metadata files, preserving unrelated files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    known = {asset.directory_name for asset in ASSETS}
    unknown = set(args.only) - known
    if unknown:
        raise ValueError(f"unknown --only assets: {sorted(unknown)}")
    selected = [asset for asset in ASSETS if not args.only or asset.directory_name in args.only]
    results = [build_asset(asset, args.force) for asset in selected]
    if args.preview_dir is not None:
        for asset in selected:
            preview_asset(asset, args.preview_dir)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
