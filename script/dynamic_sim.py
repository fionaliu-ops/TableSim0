"""Closed-loop TableSim runner driven by HighLevelPlanner VLA requests.

The simulator owns execution, observations and grounding. The high-level
planner package is intentionally unchanged. A generated graph goal is a
prediction; only a new camera observation can certify task completion here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class DynamicSimError(RuntimeError):
    def __init__(self, message, *, blockers=None, blocker_corridors=None):
        super().__init__(message)
        self.blockers = tuple(dict.fromkeys(blockers or ()))
        self.blocker_corridors = {
            str(name): tuple(tuple(tuple(float(v) for v in point)
                                  for point in segment)
                             for segment in segments)
            for name, segments in (blocker_corridors or {}).items()
        }


def side_grasp_geometry(center, footprint, height, bottom, yaw, *, edge,
                        wrist_pitch_degrees):
    """Build a horizontal edge-pinch frame without simulator dependencies."""
    directions = {"front": (0., -1.), "back": (0., 1.),
                  "left": (-1., 0.), "right": (1., 0.)}
    outward = np.asarray(directions[edge], dtype=float)
    planar = np.array([[math.cos(yaw), -math.sin(yaw)],
                       [math.sin(yaw), math.cos(yaw)]])
    local = outward * (np.asarray(footprint, dtype=float) / 2. - .002)
    xy = np.asarray(center, dtype=float)[:2] + planar @ local
    inward = -(planar @ outward)
    slope = math.tan(math.radians(wrist_pitch_degrees))
    tool_x = np.asarray([inward[0], inward[1], -slope], dtype=float)
    tool_x /= np.linalg.norm(tool_x)
    tool_y = np.array([0., 0., 1.])
    tool_y -= float(tool_y @ tool_x) * tool_x
    tool_y /= np.linalg.norm(tool_y)
    rotation = np.column_stack((tool_x, tool_y, np.cross(tool_x, tool_y)))
    return np.r_[xy, bottom + height / 2.], rotation


def separated_holding_pose_is_safe(held_position, support_position,
                                    required_separation, inter_arm_clearance,
                                    holding_margin):
    """Judge a waiting pose by its safety purpose, not target-point error."""
    return (np.linalg.norm(np.asarray(held_position, dtype=float)[:2]
                           - np.asarray(support_position, dtype=float)[:2])
            >= float(required_separation)
            and float(inter_arm_clearance) >= float(holding_margin))


def request_action(request: Any) -> dict[str, Any]:
    """Read a generated atomic edge; never infer an action from prose."""
    payload = request.to_dict() if hasattr(request, "to_dict") else request
    edge = payload.get("metadata", {}).get("reasoning_edge", {})
    target = payload.get("object_target", {})
    kind = edge.get("action_type")
    source = target.get("source") or edge.get("parameters", {}).get("source_object_id")
    if kind not in {"pick", "place", "move", "push", "stack", "insert", "align", "reorient", "open", "close", "discard", "attach"}:
        raise DynamicSimError(f"Unsupported generated atomic action: {kind!r}")
    if not source:
        raise DynamicSimError("VLARequest has no source object ID")
    return {"kind": kind, "source": str(source), "target": target.get("target"),
            "parameters": dict(edge.get("parameters") or {}), "instruction": payload["instruction"]}


def fit_image_to_world(graph, object_map, task):
    """Fit an empirical affine map from *observed* object centers to sim XY.

    This is scene-specific grounding, not a claim that generated image pixels
    encode metric poses. Reject an ill-conditioned or inconsistent fit.
    """
    image, world = [], []
    for obj in graph.objects:
        name = object_map.get(obj.id)
        if name is None or obj.attributes.xy is None:
            continue
        xy = np.asarray(obj.attributes.xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            continue
        image.append([xy[0], xy[1], 1.0])
        world.append(np.asarray(task.objects[name].get_pose().p[:2], dtype=float))
    matrix = np.asarray(image, dtype=float)
    values = np.asarray(world, dtype=float)
    if len(matrix) < 3 or np.linalg.matrix_rank(matrix) < 3:
        raise DynamicSimError("Need three non-collinear matched objects to ground image coordinates")
    transform, _, _, _ = np.linalg.lstsq(matrix, values, rcond=None)
    residual = np.linalg.norm(matrix @ transform - values, axis=1)
    if float(np.max(residual)) > 0.065:
        raise DynamicSimError(f"Image/world grounding is inconsistent: max error {max(residual):.3f} m")
    return transform


def match_objects(graph, task):
    """Fail closed on ambiguous names; user can provide explicit ID mappings."""
    result, used = {}, set()
    for obj in graph.objects:
        label = obj.name.lower().replace("-", "_").replace(" ", "_")
        candidates = [name for name in task.objects if name == label or name in label.split("_")]
        if len(candidates) == 1 and candidates[0] not in used:
            result[obj.id] = candidates[0]
            used.add(candidates[0])
    return result


def complete_object_mapping(graph, task, mapping=None):
    """Complete ambiguous visual identities by global geometric assignment.

    Repeated detector labels are matched to the remaining simulator inventory
    only after an affine image/world transform is grounded by unambiguous
    objects.  This uses no scene name, object-specific alias, or saved mapping.
    """
    from itertools import permutations

    result = {**match_objects(graph, task), **(mapping or {})}
    unmatched_ids = [obj.id for obj in graph.objects if obj.id not in result
                     and obj.attributes.xy is not None]
    unmatched_names = [name for name in task.objects if name not in result.values()]
    if not unmatched_ids or not unmatched_names:
        return result
    try:
        transform = fit_image_to_world(graph, result, task)
    except DynamicSimError:
        return result
    predicted = {obj.id: np.r_[np.asarray(obj.attributes.xy, dtype=float), 1.] @ transform
                 for obj in graph.objects if obj.id in unmatched_ids}
    count = min(len(unmatched_ids), len(unmatched_names))
    # Scenes contain few tabletop instances. Bound the exact assignment and
    # leave genuinely unobservable objects unmapped instead of guessing.
    if count > 8:
        return result
    best = None
    assignments = ((unmatched_ids, names) for names in permutations(unmatched_names, count)) \
        if len(unmatched_ids) <= len(unmatched_names) else \
        ((ids, unmatched_names) for ids in permutations(unmatched_ids, count))
    for ids, names in assignments:
        cost = sum(float(np.linalg.norm(predicted[object_id]
                   - task.objects[name].get_pose().p[:2]))
                   for object_id, name in zip(ids, names))
        candidate = (cost, ids, names)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is not None:
        for object_id, name in zip(best[1], best[2]):
            if np.linalg.norm(predicted[object_id] - task.objects[name].get_pose().p[:2]) <= .09:
                result[object_id] = name
    return result


class TableSimAdapter:
    """Standard reset/observe/step/evaluate interface for a TableSim scene."""

    motion_error_type = DynamicSimError

    def __init__(self, task, settings: dict[str, Any], output_dir: Path,
                 object_map: dict[str, str] | None = None):
        self.task = task
        self.settings = settings
        self.output_dir = Path(output_dir)
        self.object_map = dict(object_map or {})
        self.transform = None
        self.held: tuple[str, str] | None = None
        self.held_orientation = None
        self.step_index = 0
        self.motion_frame_index = 0
        self.motion_dir = self.output_dir / "motion_frames"
        self.last_placement = None
        self.thin_placement_margin = .04
        self.max_placement_deviation = .08
        self._motion_target = None
        self.last_grasp = None
        self._pinned_poses = {}
        self._pin_sync_installed = False
        self.arm_safety = None
        self.image_goal = None
        self._requested_arm = None
        self.graph_validation_mode = False
        self.hybrid_execution = False
        self.object_motion_validation = False
        self.symbolic_grasp_validation = False
        self.execution_policy = {}
        self.execution_control_dir = None

    def reset(self, seed=0):
        if self.held is not None:
            raise DynamicSimError("Cannot reset while holding an object")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        options = dict(self.settings)
        options.update(seed=seed, now_ep_num=0, need_plan=True, save_data=False,
                       save_freq=None, render_freq=0, eval_mode=False)
        self.task.setup_demo(**options)
        if not hasattr(self.task, "objects") or not hasattr(self.task, "_top_down_pose"):
            raise DynamicSimError("Scene must expose objects and top-down robot motions")
        if hasattr(self.task, "robot"):
            from script.dual_arm_safety import DualArmSafety
            self.arm_safety = DualArmSafety(self.task.robot, self.output_dir, scene=self.task.scene)
            original_dense = self.task.take_dense_action
            def safe_dense(control, save_freq=-1):
                if self.graph_validation_mode and not (self.hybrid_execution or self.symbolic_grasp_validation):
                    return original_dense(control, save_freq)
                self.arm_safety.validate(control)
                # Some assisted-release helpers replace the attachment hook.
                # Install the mandatory monitor at the controller boundary so
                # those replacements cannot disable safety during retreat.
                previous_sync = self.task._sync_attached_actor
                def checked_control_sync():
                    previous_sync()
                    self.arm_safety.check_live()
                self.task._sync_attached_actor = checked_control_sync
                # Not every task implementation invokes the attachment hook
                # from its dense controller. The joint command is common to
                # all of them; checking before the next command observes the
                # state produced by the preceding physics tick.
                original_set_arm_joints = getattr(
                    self.task.robot, 'set_arm_joints', None)
                def checked_set_arm_joints(*args, **kwargs):
                    self.arm_safety.check_live()
                    return original_set_arm_joints(*args, **kwargs)
                if original_set_arm_joints is not None:
                    self.task.robot.set_arm_joints = checked_set_arm_joints
                try:
                    result = original_dense(control, save_freq)
                    if original_set_arm_joints is not None:
                        self.arm_safety.check_live()
                    return result
                finally:
                    if original_set_arm_joints is not None:
                        self.task.robot.set_arm_joints = original_set_arm_joints
                    self.task._sync_attached_actor = previous_sync
            self.task.take_dense_action = safe_dense
            self.arm_safety.check_live()
        self.step_index = 0
        self.motion_frame_index = 0
        self.last_placement = None
        self.motion_dir.mkdir(parents=True, exist_ok=True)
        return self.observe()

    def _capture_motion_frame(self, label: str) -> None:
        """Capture intermediate robot/camera state without advancing the logical step."""
        self.motion_dir.mkdir(parents=True, exist_ok=True)
        frame = self.task.get_obs()
        observations = frame.get("observation", {})
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
        for camera_name, camera in observations.items():
            if not isinstance(camera, dict) or "rgb" not in camera:
                continue
            image = np.asarray(camera["rgb"], dtype=np.uint8)
            Image.fromarray(image).save(
                self.motion_dir / f"motion_{self.motion_frame_index:04d}_{safe}_{camera_name}.png"
            )
        self.motion_frame_index += 1

    def observe(self):
        frame = self.task.get_obs()
        head = frame["observation"]["head_camera"]["rgb"]
        image = self.output_dir / f"observation_{self.step_index:03d}.png"
        revision = 1
        while image.exists():
            image = self.output_dir / f"observation_{self.step_index:03d}_r{revision:02d}.png"
            revision += 1
        Image.fromarray(np.asarray(head, dtype=np.uint8)).save(image)
        self.last_observation_image = image
        return {"image": image, "raw": frame,
                "object_poses": {name: [*map(float, obj.get_pose().p), *map(float, obj.get_pose().q)]
                                 for name, obj in self.task.objects.items()}}

    def bind_graph(self, graph, mapping=None):
        reviewed = graph.metadata.get('object_level_perception', {}).get(
            'visual_label_review', {}).get('status') == 'confirmed'
        if reviewed:
            # Corrected visual semantics drive identity binding. Never repair
            # a reviewed identity by comparing it to simulator world poses.
            completed = {**match_objects(graph, self.task),
                         **self.object_map, **(mapping or {})}
            missing = [o.id for o in graph.objects
                       if o.attributes.movable and o.id not in completed]
            if missing:
                raise DynamicSimError(
                    f'Visual identities have no unambiguous execution binding: {missing}')
        else:
            completed = complete_object_mapping(graph, self.task, {**self.object_map, **(mapping or {})})
        self.object_map.update(completed)
        invalid = {k: v for k, v in self.object_map.items() if v not in self.task.objects}
        if invalid or len(set(self.object_map.values())) != len(self.object_map):
            raise DynamicSimError(f"Invalid or duplicate scene object mapping: {invalid or self.object_map}")
        self.transform = fit_image_to_world(graph, self.object_map, self.task)
        return dict(self.object_map)

    def _name(self, object_id):
        if object_id not in self.object_map:
            raise DynamicSimError(f"No simulator object mapping for {object_id!r}")
        return self.object_map[object_id]

    def _arm(self, name):
        if self._requested_arm and self._requested_arm[0] == name:
            return self._requested_arm[1]
        if not hasattr(self.task, "robot"):
            configured = self.task.OBJECT_SPECS.get(name, {}).get("arm")
            return configured if configured in {"left", "right"} else "left"
        position = np.asarray(self.task.objects[name].get_pose().p, dtype=float)
        footprint = self.task.object_geometry[name].get("footprint")
        # If the source rests on another object, approach from the side with
        # the largest exposed overhang. This generalizes thin-object pickup
        # without naming either the source or its support.
        if footprint is not None:
            source_left = position[0] - footprint[0] / 2
            source_right = position[0] + footprint[0] / 2
            for other, actor in self.task.objects.items():
                if other == name:
                    continue
                other_geometry = self.task.object_geometry[other]
                other_footprint = other_geometry.get("footprint")
                other_position = np.asarray(actor.get_pose().p, dtype=float)
                support_top = other_position[2] + other_geometry["height"]
                if (other_footprint is None or not 0 <= position[2] - support_top <= .03
                        or not self._rectangles_overlap(position[:2], footprint,
                            self._object_yaw(name), other_position[:2], other_footprint,
                            self._object_yaw(other), 0.0)):
                    continue
                left_overhang = max(0.0, other_position[0] - other_footprint[0] / 2 - source_left)
                right_overhang = max(0.0, source_right - (other_position[0] + other_footprint[0] / 2))
                if max(left_overhang, right_overhang) >= .01:
                    return "left" if left_overhang > right_overhang else "right"
        # Workspace ownership follows the live source position. An object
        # placed across the centre can therefore be picked by the other arm on
        # the next graph edge, providing a geometry-derived handoff.
        return "left" if position[0] < 0 else "right"

    def _motion(self, arm, pose, label):
        original_sync = getattr(self.task, "_sync_attached_actor", lambda: None)
        touched = set()
        recovery_allowed = set()
        if hasattr(self.task, "robot"):
            robot_entities = set()
            for articulation in (self.task.robot.left_entity, self.task.robot.right_entity):
                for link in articulation.get_links():
                    entity = getattr(link, "entity", link)
                    robot_entities.add(entity.get_name())
            protected = {actor.get_name(): name for name, actor in self.task.objects.items()
                         if name != self._motion_target
                         and not (self.held and name == self.held[0])}
            if getattr(self.task, "monitor", None) is not None:
                protected[self.task.monitor.get_name()] = "monitor"
            def checked_sync():
                original_sync()
                for contact in self.task.scene.get_contacts():
                    names = {body.entity.name for body in contact.bodies}
                    if names & robot_entities:
                        touched.update(protected[entity] for entity in names
                                       if entity in protected
                                       and protected[entity] not in recovery_allowed)
                active = touched - recovery_allowed
                if active:
                    raise DynamicSimError(
                        f"Robot touched non-target objects during {label}: {sorted(active)}",
                        blockers=sorted(active))
            self.task._sync_attached_actor = checked_sync
        try:
            from script.dual_arm_safety import TrajectoryClearanceError
            try:
                if hasattr(self.task, "robot"):
                    entity = getattr(self.task.robot, arm + '_entity')
                    plan, next_qpos = self._plan_arm_motion(arm, pose, entity.get_qpos().copy())
                    if next_qpos is not None:
                        try:
                            self._execute_planned_motion(arm, plan, label)
                        except DynamicSimError as collision:
                            # Leave the scene in the last safe configuration by
                            # reversing the actually approached prefix. Contact
                            # with the just-hit object is tolerated only while
                            # withdrawing along that same path.
                            message = str(collision).casefold()
                            known = list(self.task.objects)
                            if getattr(self.task, "monitor", None) is not None:
                                known.append("monitor")
                            recovery_allowed.update(name for name in known
                                                    if name.casefold() in message)
                            positions = np.asarray(plan["position"], dtype=float)
                            joint_names = [joint.get_name() for joint in entity.get_active_joints()]
                            indices = [joint_names.index(joint.get_name()) for joint in
                                       getattr(self.task.robot, arm + '_arm_joints')]
                            current = np.asarray(entity.get_qpos(), dtype=float)[indices]
                            nearest = int(np.argmin(np.linalg.norm(positions-current, axis=1)))
                            reverse = np.vstack((current, positions[:nearest+1][::-1]))
                            rollback = {"position": reverse,
                                        "velocity": np.zeros_like(reverse)}
                            self._execute_planned_motion(arm, rollback,
                                                         f"{label}_collision_withdraw")
                            self._capture_motion_frame(f"{label}_collision_withdraw")
                            raise collision
                        ok = True
                    else:
                        ok = False
                else:
                    ok = self.task.move(self.task.move_to_pose(self.task._arm_tag(arm), pose))
                if not ok or not self.task.plan_success:
                    # A failed trajectory optimisation has not executed its
                    # path. Try a fully prevalidated segmented return instead.
                    self.task.plan_success = True
                    ok = self._route_above_other_arm(arm, pose, label)
            except TrajectoryClearanceError:
                # Retry through a higher corridor, but validate every segment
                # before advancing the first one. A runtime violation never retries.
                ok = self._route_above_other_arm(arm, pose, label)
        finally:
            self.task._sync_attached_actor = original_sync
        if not ok or not self.task.plan_success:
            paths = getattr(self.task, f"{arm}_joint_path", [])
            detail = paths[-1].get("failure_status", paths[-1].get("status")) if paths else None
            raise DynamicSimError(f"Robot motion failed at {label}: {detail}")
        self._capture_motion_frame(label)

    def _route_above_other_arm(self, arm, destination, label):
        current = np.asarray(self.task._current_ee_pose(arm), dtype=float)
        destination = np.asarray(destination, dtype=float)
        entity = getattr(self.task.robot, arm + '_entity')
        routes = []
        for height in (1.08, 1.16, 1.24):
            if height < max(current[2], destination[2]) + .03:
                continue
            start_high, end_high = current.copy(), destination.copy()
            start_high[2] = end_high[2] = height
            routes.append((start_high, end_high, destination))
        if self.held is None:
            # Rotate the empty wrist in its own front workspace before reaching
            # laterally. An overhead-only route can require an unreachable IK
            # branch after a previous placement, even with ample arm clearance.
            for height in (1.00, 1.08):
                front = destination.copy()
                front[:3] = [.20 if arm == 'right' else -.20, -.20, height]
                routes.append((front, destination))
        failures = []
        for route in routes:
            qpos = entity.get_qpos().copy()
            plans = []
            for waypoint in route:
                plan, next_qpos = self._plan_arm_motion(arm, waypoint.tolist(), qpos)
                if next_qpos is None:
                    failures.append({'waypoint': waypoint.tolist(),
                                     'reason': str(plan.get('failure_status', plan.get('status')))})
                    break
                plans.append(plan)
                qpos = next_qpos
            if len(plans) == len(route):
                for i, plan in enumerate(plans):
                    self._execute_planned_motion(arm, plan, f'{label}_safe_route_{i}')
                return True
        raise DynamicSimError(f'No separated-arm route found for {arm} {label}: {failures}')

    def _gripper(self, arm, close):
        action = self.task.close_gripper(self.task._arm_tag(arm), pos=0.0) if close else self.task.open_gripper(self.task._arm_tag(arm))
        if not self.task.move(action):
            raise DynamicSimError("Gripper command failed")
        self._capture_motion_frame("gripper_close" if close else "gripper_open")

    def _graph_validation_motion(self, arm, pose, label):
        """Move continuously; never advance a grasp phase after failed motion."""
        self.task.plan_success = True
        ok = self.task.move(self.task.move_to_pose(self.task._arm_tag(arm), pose))
        actual = np.asarray(self.task._current_ee_pose(arm), dtype=float)
        desired = np.asarray(pose, dtype=float)
        distance = float(np.linalg.norm(actual[:3] - desired[:3]))
        self._capture_motion_frame(label)
        if not ok or not self.task.plan_success or distance > .015:
            raise DynamicSimError(f"Gripper motion incomplete at {label}; error={distance:.4f}m")
        return True

    def _graph_validation_place(self, action, name):
        """Grasp, rigidly follow the actual gripper, then release in place."""
        import sapien
        from copy import deepcopy
        if self.held is not None:
            raise DynamicSimError("Graph validation cannot start with a held object")
        actor = self.task.objects[name]
        arm = self._arm(name)
        top = float(actor.get_pose().p[2] + self.task.object_geometry[name]["height"])
        source_xy = np.asarray(actor.get_pose().p[:2], dtype=float)
        approach = self.task._top_down_pose(source_xy, top, arm, clearance=.10)
        grasp = self.task._top_down_pose(source_xy, top, arm, clearance=.04)
        self._gripper(arm, False)
        reached = [self._graph_validation_motion(arm, approach, "grasp_approach")]
        reached.append(self._graph_validation_motion(arm, grasp, "grasp_pose"))
        self._gripper(arm, True)
        # Attachment preserves the observed relative pose; it does not snap
        # the object to the tool or to any requested destination.
        self.task._attach_actor(name, arm)
        self.held = (name, arm)
        observed = deepcopy(actor.get_pose())
        self.held_orientation = np.asarray(observed.q, dtype=float).copy()
        placement = (action.get("parameters", {}).get("pose")
                     or action.get("parameters", {}).get("relative_pose"))
        if action.get("kind") in {"reorient", "align"} and not placement:
            target_xy, bottom = np.asarray(observed.p[:2]), float(observed.p[2])
        else:
            target_xy, bottom = self._destination(action, name)
        orientation = np.asarray(observed.q, dtype=float).copy()
        parameters = action.get("parameters", {})
        target_yaw = (placement or {}).get("yaw_degrees", parameters.get("target_yaw_degrees"))
        if target_yaw is not None:
            matrix = observed.to_transformation_matrix()
            yaw = math.degrees(math.atan2(matrix[1,0], matrix[0,0]))
            orientation = self.task._yaw_quat(float(target_yaw)-yaw, base_quat=orientation)
        target_pose = sapien.Pose([float(target_xy[0]), float(target_xy[1]), float(bottom)], orientation)
        # Solve the desired wrist pose from the fixed grasp transform. The
        # object only moves through _sync_attached_actor during robot ticks.
        relative = np.asarray(self.task._ee_to_actor)
        release_matrix = target_pose.to_transformation_matrix() @ np.linalg.inv(relative)
        release_pose = self.task._matrix_pose(release_matrix)
        release = np.r_[release_pose.p, release_pose.q]
        lift = np.asarray(self.task._current_ee_pose(arm), dtype=float)
        lift[2] += .14
        reached.append(self._graph_validation_motion(arm, lift.tolist(), "grasp_lift"))
        high = release.copy()
        high[2] = max(high[2]+.14, lift[2])
        reached.append(self._graph_validation_motion(arm, high.tolist(), "follow_transfer"))
        reached.append(self._graph_validation_motion(arm, release.tolist(), "follow_lower"))
        actual = deepcopy(actor.get_pose())
        error = float(np.linalg.norm(np.asarray(actual.p)-np.asarray(target_pose.p)))
        if error > .015:
            raise DynamicSimError(f"Attached object did not reach release pose; error={error:.4f}m")
        self._gripper(arm, False)
        # Freeze the actual release pose, never the commanded pose. Opening
        # and retreat cannot introduce a coordinate/rotation correction jump.
        released = deepcopy(actor.get_pose())
        self.task._detach_actor_at(name, released)
        self._clear_attachment_velocity(name)
        self.held = None
        self.held_orientation = None
        retreat = np.asarray(self.task._current_ee_pose(arm), dtype=float)
        retreat[2] += .16
        reached.append(self._graph_validation_motion(arm, retreat.tolist(), "post_release_retreat"))
        actual_xy = np.asarray(actor.get_pose().p[:2], dtype=float)
        self.last_placement = {
            "object": name, "requested_xy": np.asarray(target_xy).tolist(),
            "actual_xy": actual_xy.tolist(),
            "placement_error": float(np.linalg.norm(actual_xy-target_xy)),
            "mode": "continuous_gripper_follow",
            "arm_motion_reached": reached,
        }

    def _side_grasp_obstacle_clearance(self, name, point, rotation):
        """Measure live clearance along the finger insertion corridor.

        This deliberately uses only the current actor geometry.  It prevents a
        side pinch from being selected merely because the wrist IK succeeds
        while a finger or the object edge would sweep through a neighbour.
        """
        point = np.asarray(point, dtype=float)
        inward = np.asarray(rotation, dtype=float)[:, 0]
        # Cover the open fingers from the approach pose through the contact
        # edge.  The wrist itself is checked by the regular sphere preflight.
        start = point - inward * .212
        samples = np.linspace(start[:2], point[:2], 25)
        tool_bottom, tool_top = point[2] - .025, point[2] + .025
        clearances = []
        obstacles = [(other, actor, self.task.object_geometry[other])
                     for other, actor in self.task.objects.items() if other != name]
        monitor = getattr(self.task, "monitor", None)
        monitor_spec = getattr(self.task, "MONITOR_SPEC", None)
        if monitor is not None and monitor_spec is not None:
            geometry = getattr(self.task, "object_geometry_by_spec", {}).get(id(monitor_spec))
            if geometry is not None:
                obstacles.append(("monitor", monitor, geometry))
        for other, actor, geometry in obstacles:
            footprint = geometry.get("footprint")
            if footprint is None:
                continue
            center = np.asarray(actor.get_pose().p, dtype=float)
            bottom = float(center[2])
            top = bottom + float(geometry["height"])
            if top < tool_bottom or bottom > tool_top:
                continue
            yaw = self._object_yaw(other)
            c, s = math.cos(yaw), math.sin(yaw)
            delta = samples - center[:2]
            local = np.column_stack((c * delta[:, 0] + s * delta[:, 1],
                                     -s * delta[:, 0] + c * delta[:, 1]))
            outside = np.maximum(np.abs(local) - np.asarray(footprint) / 2, 0.)
            clearances.append((float(np.min(np.linalg.norm(outside, axis=1))), other))
        return min(clearances, default=(float("inf"), None))

    def _select_side_grasp(self, name, arm, height, yaw, *, cup_frame=False):
        """Choose an edge from live geometry and continuous IK feasibility."""
        import transforms3d as t3d

        actor = self.task.objects[name]
        center = actor.get_pose().p
        footprint = self.task.object_geometry[name]["footprint"]
        candidates = []
        # Wrist pitch is an IK degree of freedom, not a scene constant.  A
        # buffer move can leave the same object at a different part of the
        # workspace where the formerly reachable pitch belongs to another IK
        # branch.  Search a small morphology-derived family for every edge.
        # These angles are independent of object positions and scene ids.
        pitches = (30.,) if cup_frame else (12., 18., 24.)
        for edge in ("front", "right", "back", "left"):
            for pitch in pitches:
                point, rotation = side_grasp_geometry(
                    center, footprint, height, float(center[2]),
                    0. if cup_frame else yaw, edge=edge,
                    wrist_pitch_degrees=pitch)
                if cup_frame:
                    jaw = np.cross([0., 0., 1.], rotation[:, 0])
                    jaw /= np.linalg.norm(jaw)
                    rotation = np.column_stack((rotation[:, 0], jaw,
                                                np.cross(rotation[:, 0], jaw)))
                clearance, blocker = self._side_grasp_obstacle_clearance(
                    name, point, rotation)
                corridor_start = point - rotation[:, 0] * .212
                quat = t3d.quaternions.mat2quat(rotation).tolist()
                grasp = np.r_[point - rotation[:, 0] * .147, quat]
                approach = np.r_[point - rotation[:, 0] * .212, quat]
                high = approach.copy()
                high[2] += .07
                candidates.append({"edge": edge, "pitch": pitch,
                                   "clearance": clearance, "blocker": blocker,
                                   "corridor": [corridor_start[:2].tolist(),
                                                point[:2].tolist()],
                                   "grasp": grasp, "approach": approach,
                                   "high": high})

        # Prefer the largest live insertion clearance.  For equal clearances,
        # this stable order avoids introducing scene-specific coordinates.
        candidates.sort(key=lambda item: item["clearance"], reverse=True)
        entity = getattr(self.task.robot, arm + "_entity")
        audit = []
        selected = None
        collision_blockers = []
        # Reserve most of the same clearance envelope used for dual-arm
        # planning.  The centerline distance alone omits the open finger and
        # wrist width, which is why an IK-valid edge can still touch a nearby
        # object during insertion.
        configured_clearance = (self.arm_safety.minimum_clearance
                                if self.arm_safety is not None else .05)
        minimum_insertion_clearance = .8 * configured_clearance
        for candidate in candidates:
            record = {key: candidate[key]
                      for key in ("edge", "pitch", "clearance", "blocker")}
            if candidate["clearance"] < minimum_insertion_clearance:
                record["result"] = "insertion_corridor_blocked"
                audit.append(record)
                continue
            qpos = entity.get_qpos().copy()
            failures = []
            for label in ("high", "approach", "grasp"):
                self._side_grasp_preflight = True
                try:
                    plan, next_qpos = self._plan_arm_motion(
                        arm, candidate[label].tolist(), qpos)
                finally:
                    self._side_grasp_preflight = False
                if next_qpos is None:
                    reason = str(plan.get("failure_status", plan.get("status")))
                    failures.append({"segment": label,
                                     "reason": reason})
                    prefix = "trajectory intersects "
                    if reason.startswith(prefix):
                        collision_blockers.append(reason[len(prefix):].strip())
                    break
                qpos = next_qpos
            if failures:
                record["result"] = "ik_or_clearance_rejected"
                record["failures"] = failures
            else:
                record["result"] = "selected"
                selected = candidate
            audit.append(record)
            if selected is not None:
                break
        with (self.output_dir / "side_grasp_candidates.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": self.step_index, "object": name,
                                     "arm": arm, "candidates": audit}) + "\n")
        if selected is None:
            known = set(self.task.objects)
            blockers = list(dict.fromkeys(
                item for item in collision_blockers if item in known))
            # A nearby object is causal only when the measured insertion
            # corridor is too narrow.  If at least one corridor is clear and
            # MotionGen reports pure IK/trajectory-optimisation failure,
            # moving the nearest object cannot change that IK branch.  Return
            # no blocker so the online graph switches arm/grasp family rather
            # than repeatedly relocating innocent neighbours.
            geometrically_open = any(
                record.get("result") == "ik_or_clearance_rejected"
                for record in audit)
            if not blockers and not geometrically_open:
                ranked = list(dict.fromkeys(
                    record.get("blocker") for record in audit
                    if record.get("result") == "insertion_corridor_blocked"
                    and record.get("blocker") in known))
                movable = []
                for blocker in ranked:
                    actor = getattr(self.task.objects[blocker], "actor",
                                    self.task.objects[blocker])
                    static = any("RigidStatic" in type(component).__name__
                                 for component in getattr(actor, "get_components", lambda: [])())
                    if not static:
                        movable.append(blocker)
                blockers = (movable or ranked)[:1]
            blocker_corridors = {
                blocker: [candidate["corridor"] for candidate in candidates
                          if candidate.get("blocker") == blocker
                          and candidate["clearance"] < minimum_insertion_clearance]
                for blocker in blockers
            }
            raise DynamicSimError(
                f"No live-geometry side grasp for {name}: {audit}",
                blockers=blockers, blocker_corridors=blocker_corridors)
        return selected

    def _select_top_grasp(self, name, arm, xy, top):
        """Select a live IK-feasible wrist attitude for a top pinch."""
        from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
        import transforms3d as t3d

        labels = ([arm + "_arm_perf", "top_down_little_right",
                   "top_down_little_left", "top_down"] if arm == "left" else
                  [arm + "_arm_perf", "top_down_little_left",
                   "top_down_little_right", "top_down"])
        entity = getattr(self.task.robot, arm + "_entity")
        audit = []
        collision_blockers = []
        for orientation in labels:
            quat = np.asarray(GRASP_DIRECTION_DIC[orientation], dtype=float)
            rotation = t3d.quaternions.quat2mat(quat)
            contact = np.asarray([xy[0], xy[1], top], dtype=float)
            grasp = np.r_[contact + rotation @ np.asarray([-.125, 0., 0.]), quat]
            approach = grasp.copy()
            approach[:3] += rotation @ np.asarray([-.075, 0., 0.])
            qpos = entity.get_qpos().copy()
            failures = []
            for segment, target in (("approach", approach), ("grasp", grasp)):
                plan, next_qpos = self._plan_arm_motion(arm, target.tolist(), qpos)
                if next_qpos is None:
                    reason = str(plan.get("failure_status", plan.get("status")))
                    failures.append({"segment": segment, "reason": reason})
                    prefix = "trajectory intersects "
                    if reason.startswith(prefix):
                        collision_blockers.append(reason[len(prefix):].strip())
                    break
                qpos = next_qpos
            audit.append({"orientation": orientation,
                          "result": "selected" if not failures else "rejected",
                          "failures": failures})
            if not failures:
                with (self.output_dir / "top_grasp_candidates.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": self.step_index, "object": name,
                        "arm": arm, "candidates": audit}) + "\n")
                return approach, grasp
        with (self.output_dir / "top_grasp_candidates.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": self.step_index, "object": name,
                "arm": arm, "candidates": audit}) + "\n")
        known = set(self.task.objects)
        raise DynamicSimError(f"No live-IK top grasp for {name}: {audit}",
                              blockers=[item for item in collision_blockers if item in known])

    def _pick(self, name):
        if self.held is not None:
            raise DynamicSimError("Cannot pick another object while one is held")
        self._pinned_poses.pop(name, None)
        arm = self._arm(name)
        if self.arm_safety is not None:
            self._motion_target = None
            self._park_arm("right" if arm == "left" else "left")
        actor = self.task.objects[name]
        self._motion_target = name
        grasp_orientation = np.asarray(actor.get_pose().q, dtype=float).copy()
        xy = actor.get_pose().p[:2]
        height = self.task.object_geometry[name]["height"]
        top = float(actor.get_pose().p[2] + height)
        if name in {"book", "keyboard"} or (name == "cup" and arm == "right"):
            if name in {"book", "keyboard"}:
                # Side-pinches are sensitive to the preceding wrist posture.
                # Retract along the known home path before planning the edge
                # approach so the same grasp does not depend on the last item.
                if not self.task.move(self.task.back_to_origin(
                        arm_tag=self.task._arm_tag(arm))):
                    raise DynamicSimError(f"Failed to retract before {name} side grasp")
                self._capture_motion_frame(f"{name}_pregrasp_arm_home")
            matrix = actor.get_pose().to_transformation_matrix()
            yaw = math.atan2(matrix[1, 0], matrix[0, 0])
            candidate = self._select_side_grasp(
                name, arm, height, yaw, cup_frame=name == "cup")
            edge = candidate["edge"]
            grasp = candidate["grasp"].tolist()
            approach = candidate["approach"].tolist()
            high = candidate["high"]
            self._gripper(arm, False)
            self._motion(arm, high.tolist(), "side_high")
            self._motion(arm, approach, "side_approach")
            self._motion(arm, grasp, "side_grasp")
            grasp_error = float("inf")
            for correction in range(3):
                grasp_error = float(np.linalg.norm(
                    np.asarray(self.task._current_ee_pose(arm), dtype=float)[:3]
                    - np.asarray(grasp, dtype=float)[:3]))
                if grasp_error <= .08:
                    break
                self._motion(arm, grasp, f"side_grasp_correction_{correction + 1}")
            grasp_error = float(np.linalg.norm(
                np.asarray(self.task._current_ee_pose(arm), dtype=float)[:3]
                - np.asarray(grasp, dtype=float)[:3]))
            if grasp_error > .08:
                raise DynamicSimError(
                    f"Gripper did not reach the {name} side grasp: {grasp_error:.3f} m")
            self._gripper(arm, True)
            self.last_grasp = {"object": name, "arm": arm,
                               "mode": "horizontal_side_pinch",
                               "edge": edge, "position_error": grasp_error,
                               "follow_started_after_gripper_close": True}
            if name == 'book':
                # An assisted rigid grasp must not fight the free-body
                # solver between pose updates, especially during rotation.
                for component in actor.actor.get_components():
                    if hasattr(component, 'set_kinematic'):
                        component.set_kinematic(True)
            self.task._attach_actor(name, arm)
            self.held = (name, arm)
            self.held_orientation = grasp_orientation
            lifted = np.asarray(self.task._current_ee_pose(arm), dtype=float)
            lifted[2] += .07
            self._motion(arm, lifted.tolist(), "lift")
            # The joint controller can lag a single Cartesian command.  A
            # second command to the same pose lets the held footprint clear
            # nearby objects before any horizontal sweep begins.
            self._motion(arm, lifted.tolist(), "lift_settle")
            self._motion_target = None
            return
        self._gripper(arm, False)
        if hasattr(self.task, "robot") and self.arm_safety is not None:
            approach, grasp = self._select_top_grasp(name, arm, xy, top)
        else:
            pose_arm = "right" if name == "stapler" else arm
            approach = self.task._top_down_pose(xy, top, pose_arm, clearance=.08)
            grasp = self.task._top_down_pose(xy, top, pose_arm, clearance=.005)
        self._motion(arm, np.asarray(approach).tolist(), "approach")
        self._motion(arm, np.asarray(grasp).tolist(), "grasp")
        self._gripper(arm, True)
        self.task._attach_actor(name, arm)
        self.held = (name, arm)
        self.held_orientation = grasp_orientation
        lifted = np.asarray(grasp, dtype=float).copy()
        lifted[2] += .065
        self._motion(arm, lifted.tolist(), "lift")
        if height < .025:
            self._motion(arm, lifted.tolist(), "lift_settle")
        self._motion_target = None

    def _destination(self, action, name):
        params = action["parameters"]
        placement = params.get("pose") or params.get("relative_pose") or {}
        if not placement and isinstance(params.get("placement_target"), dict):
            placement = params["placement_target"].get("parameters", {}).get("pose", {})
        target = action["target"]
        if target and target not in {"table", "held above tabletop"}:
            support_name = self._name(target)
            support = self.task.objects[support_name].get_pose().p
            support_top = float(support[2] + self.task.object_geometry[support_name]["height"])
            # Relative image coordinates are not world displacements. Centering
            # over the named support is the conservative default.
            gap = .002 if action["kind"] == "stack" else .006
            return np.asarray(support[:2], float), support_top + gap
        if placement.get("frame") != "image_normalized" or not isinstance(placement.get("xy"), (list, tuple)):
            raise DynamicSimError("Table placement requires an image_normalized xy pose")
        xy = np.asarray(placement["xy"], dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or np.any((xy < 0) | (xy > 1)):
            raise DynamicSimError("Invalid normalized placement")
        if self.transform is None:
            raise DynamicSimError("Call bind_graph before executing image placements")
        world_xy = np.r_[xy, 1] @ self.transform
        return world_xy, self.task._table_z() + .006

    def _monitor_clear(self, name, xy, bottom, top, margin):
        monitor = getattr(self.task, 'monitor', None)
        if monitor is None:
            return True
        from script.collision_geometry import height_band_bounds
        if not hasattr(self, '_monitor_meshes'):
            meshes = []
            for component in monitor.actor.get_components():
                for shape in getattr(component, 'get_collision_shapes', lambda: [])():
                    if not hasattr(shape, 'vertices'):
                        raise DynamicSimError('Unsupported monitor collision shape')
                    transform = (monitor.get_pose() * shape.local_pose).to_transformation_matrix()
                    vertices = np.asarray(shape.vertices) * np.asarray(shape.scale)
                    meshes.append(vertices @ transform[:3, :3].T + transform[:3, 3])
            if not meshes:
                raise DynamicSimError('Monitor collision geometry is unavailable')
            self._monitor_meshes = meshes
        band_key = (round(float(bottom - margin), 4), round(float(top + margin), 4))
        cache = getattr(self, "_monitor_band_cache", None)
        if cache is None:
            cache = self._monitor_band_cache = {}
        if band_key not in cache:
            cache[band_key] = [height_band_bounds(vertices, *band_key)
                               for vertices in self._monitor_meshes]
        for bounds in cache[band_key]:
            if bounds is not None and self._rectangles_overlap(
                    xy, self.task.object_geometry[name]['footprint'], self._object_yaw(name),
                    bounds[0], bounds[1], 0., margin):
                return False
        return True

    def _clear_table_placement(self, name, xy, support=None, margin=None):
        """Conservative footprint check for a non-stacking release."""
        own = self.task.object_geometry[name].get("footprint")
        if own is None:
            return True
        if margin is None and self.image_goal is not None and name in self.image_goal.targets:
            if np.linalg.norm(np.asarray(xy) - self.image_goal.targets[name]) <= .025:
                # Scale final-layout spacing by the smaller object dimension;
                # arm clearance remains a separate constraint.
                footprint = self.task.object_geometry[name].get("footprint")
                margin = (.005 if footprint is None else
                          float(np.clip(.12 * min(footprint), .004, .015)))
        if margin is None:
            if self.task.object_geometry[name]["height"] < .025:
                margin = max(self.thin_placement_margin, .025)
            else:
                margin = .01
        own_yaw = self._object_yaw(name)
        own_extent_x = (abs(math.cos(own_yaw)) * own[0] + abs(math.sin(own_yaw)) * own[1]) / 2
        if abs(float(xy[0])) + own_extent_x + margin > .35:
            return False
        obstacles = [(other, actor, self.task.object_geometry[other])
                     for other, actor in self.task.objects.items()
                     if other != name and other != support]
        bottom = self.task._table_z() + .006
        if not self._monitor_clear(name, xy, bottom,
                                   bottom + self.task.object_geometry[name]['height'], margin):
            return False
        for other, actor, geometry in obstacles:
            footprint = geometry.get("footprint")
            if footprint is None:
                continue
            if self._rectangles_overlap(xy, own, own_yaw, actor.get_pose().p[:2],
                                        footprint, self._object_yaw(other), margin):
                return False
        return True

    def _placement_blockers(self, name, xy):
        """Name actors that overlap a requested non-stacking footprint."""
        own = self.task.object_geometry[name].get("footprint")
        if own is None:
            return []
        obstacles = [(other, actor, self.task.object_geometry[other])
                     for other, actor in self.task.objects.items() if other != name]
        monitor = getattr(self.task, "monitor", None)
        monitor_spec = getattr(self.task, "MONITOR_SPEC", None)
        if monitor is not None and monitor_spec is not None:
            geometry = getattr(self.task, "object_geometry_by_spec", {}).get(id(monitor_spec))
            if geometry is not None:
                obstacles.append(("monitor", monitor, geometry))
        return [other for other, actor, geometry in obstacles
                if geometry.get("footprint") is not None
                and self._rectangles_overlap(xy, own, self._object_yaw(name),
                                             actor.get_pose().p[:2], geometry["footprint"],
                                             self._object_yaw(other), self.thin_placement_margin
                                             if self.task.object_geometry[name]["height"] < .025 else .01)]

    def _object_yaw(self, name):
        actor = getattr(self.task, 'monitor', None) if name == 'monitor' else self.task.objects.get(name)
        if actor is not None and hasattr(actor.get_pose(), 'to_transformation_matrix'):
            rotation = actor.get_pose().to_transformation_matrix()[:3, :3]
            return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        spec = (self.task.MONITOR_SPEC if name == "monitor" else
                self.task.OBJECT_SPECS.get(name, {}))
        return math.radians(float(spec.get("initial_yaw", 0)))

    def _rotate_held_to_yaw(self, name, arm, target_degrees, allowed_contacts=()):
        """Rotate the wrist around the carried actor, preserving grasp transform."""
        import transforms3d as t3d
        target = math.radians(float(target_degrees))
        if not math.isfinite(target):
            raise DynamicSimError('Non-finite target yaw')
        # Recompute the object-centred tool pose from the measured state on
        # each segment. Never rotate the released object independently.
        for index in range(16):
            current = self._object_yaw(name)
            error = math.atan2(math.sin(target-current), math.cos(target-current))
            obj = self.task.objects[name].get_pose().to_transformation_matrix()
            tilt = math.acos(float(np.clip(obj[2, 1], -1., 1.)))
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / 'orientation_motion.jsonl').open('a') as audit:
                audit.write(json.dumps({'object':name, 'arm':arm, 'segment':index,
                    'target_degrees':target_degrees, 'observed_degrees':math.degrees(current),
                    'error_degrees':math.degrees(error), 'tilt_degrees':math.degrees(tilt),
                    'object_position':self.task.objects[name].get_pose().p.tolist()}) + '\n')
            if abs(error) <= math.radians(.5) and tilt <= math.radians(3.):
                self.held_orientation = self.task.objects[name].get_pose().q.copy()
                return
            delta = float(np.clip(error, -math.radians(15), math.radians(15)))
            desired = obj.copy()
            desired[:3, :3] = t3d.quaternions.quat2mat(self.task._yaw_quat(
                math.degrees(current + delta), self.task.RIGID_Y_UP_TO_Z_UP))
            ee = desired @ np.linalg.inv(self.task._ee_to_actor)
            pose = np.r_[ee[:3, 3], t3d.quaternions.mat2quat(ee[:3, :3])]
            entity = self.task.robot.left_entity if arm == 'left' else self.task.robot.right_entity
            plan, _ = self._plan_arm_motion(arm, pose.tolist(), entity.get_qpos().copy())
            if plan.get('status') != 'Success':
                raise DynamicSimError(f'Wrist yaw IK failed for {name}: {plan.get("failure_status", plan.get("status"))}')
            self._execute_planned_motion(arm, plan, 'transfer_correction', allowed_contacts)
            endpoint = np.asarray(plan['position'])[-1]
            self._execute_planned_motion(arm, {'position':np.repeat(endpoint[None,:],100,axis=0),
                'velocity':np.zeros((100,len(endpoint)))}, 'transfer_hold', allowed_contacts)
        raise DynamicSimError(f'{name} wrist rotation did not converge to {target_degrees} degrees')

    def _align_empty_folder(self):
        """Left fingers press/turn the exposed folder while right holds book.

        This uses the user's assisted-follow mode for a thin support: the
        folder follows the contacting wrist and remains fixed only after release.
        """
        folder_target = (self.image_goal.targets.get('folder')
                         if self.image_goal is not None else None)
        if (abs(self._object_yaw('folder')) <= math.radians(3) and
                (folder_target is None or np.linalg.norm(self.task.objects['folder'].get_pose().p[:2] - folder_target) <= .015)):
            return
        if self.held != ('book', 'right'):
            raise DynamicSimError('Folder alignment requires the book held clear by the right arm')
        task = self.task
        # Reserve the left workspace only after the book and right wrist have
        # exited it. Sample the holding point from the live table transform,
        # object footprints and configured arm clearance; no scene coordinate
        # or recorded successful pose is used.
        book_position = np.asarray(self.task.objects['book'].get_pose().p, dtype=float)
        folder_position = np.asarray(self.task.objects['folder'].get_pose().p, dtype=float)
        book_size = np.asarray(self.task.object_geometry['book']['footprint'], dtype=float)
        folder_size = np.asarray(self.task.object_geometry['folder']['footprint'], dtype=float)
        clearance = float(self.arm_safety.minimum_clearance)
        required_separation = .5 * (np.linalg.norm(book_size) + np.linalg.norm(folder_size)) + clearance
        # Carrying a wide payload amplifies tracking error during the long
        # cross-workspace move.  Reserve one live collision-sphere radius on
        # top of the ordinary path margin.  This derives the envelope from the
        # current robot model rather than a scene coordinate or replayed pose.
        model_radius = max(
            (float(np.max(radii)) for geometry in self.arm_safety.geometry.values()
             for _link, _centres, radii in geometry), default=0.)
        holding_margin = max(
            float(self.arm_safety.planning_margin) + model_radius,
            clearance, .25 * float(np.min(book_size)))
        root_x = float(self.task.robot.right_entity.get_root_pose().p[0])
        height = max(book_position[2], self.task._table_z() + 5. * clearance)
        entity = task.robot.right_entity
        indices = self.arm_safety.indices['right']
        candidates = []
        rejected = []
        current_ee = np.asarray(task._current_ee_pose('right'), dtype=float)

        def consider(waiting, source):
            target = book_position + np.asarray(waiting[:3]) - current_ee[:3]
            if np.linalg.norm(target[:2] - folder_position[:2]) < required_separation:
                rejected.append({'source': source, 'reason': 'support_separation'})
                return False
            normal_margin = self.arm_safety.planning_margin
            self.arm_safety.planning_margin = holding_margin
            try:
                # Prevalidate a stop-and-settle route. A single long joint
                # trajectory can cut through the shared workspace even when
                # its endpoint is far away. The transit height scales from
                # live robot geometry and the configured clearance.
                transit_height = (max(float(current_ee[2]), float(waiting[2]))
                                  + 2. * clearance + model_radius)
                start_high = current_ee.copy(); start_high[2] = transit_height
                end_high = np.asarray(waiting, dtype=float).copy()
                end_high[2] = transit_height
                route = (start_high, end_high, np.asarray(waiting, dtype=float))
                plans = []
                qpos = entity.get_qpos().copy()
                failure = None
                for waypoint in route:
                    plan, next_qpos = self._plan_arm_motion(
                        'right', waypoint.tolist(), qpos)
                    if next_qpos is None:
                        failure = plan
                        break
                    plans.append(plan)
                    qpos = next_qpos
            finally:
                self.arm_safety.planning_margin = normal_margin
            if failure is not None:
                rejected.append({'source': source, 'reason': str(
                    failure.get('failure_status', failure.get('status')))})
                return False
            candidates.append((float(np.linalg.norm(target-book_position)), target,
                               np.asarray(waiting, dtype=float), plans))
            return True

        home = np.asarray(self.task.robot.right_original_pose, dtype=float)
        for level in range(4):
            waiting = home.copy()
            waiting[2] += level * clearance
            if consider(waiting, f'robot_home_plus_{level}_clearances'):
                break
        lattice = []
        if not candidates:
            for v in np.linspace(.08, .92, 8):
                for u in np.linspace(.08, .92, 8):
                    xy = np.r_[np.array([u, v]), 1.] @ self.transform
                    if np.sign(xy[0]) != np.sign(root_x):
                        continue
                    if np.linalg.norm(xy - folder_position[:2]) < required_separation:
                        continue
                    lattice.append((float(np.linalg.norm(xy-book_position[:2])), u, v, xy))
            for _, u, v, xy in sorted(lattice):
                target = np.r_[xy, height]
                waiting = current_ee.copy()
                waiting[:3] += target - book_position
                if consider(waiting, f'table_lattice_{u:.3f}_{v:.3f}'):
                    break
        if not candidates:
            with (self.output_dir / 'separated_holding_candidates.jsonl').open('a') as stream:
                stream.write(json.dumps({'step': self.step_index, 'rejected': rejected}) + '\n')
            raise DynamicSimError(
                'No live-geometry separated holding pose for the supported object')
        _, waiting_target, waiting_pose, waiting_plans = min(
            candidates, key=lambda item: item[0])
        with (self.output_dir / 'separated_holding_candidates.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': self.step_index, 'selected': waiting_target.tolist(),
                                     'rejected': rejected}) + '\n')
        for route_index, waiting_plan in enumerate(waiting_plans):
            self._execute_planned_motion(
                'right', waiting_plan,
                f'held_object_wait_outside_other_workspace_route_{route_index}')
        from script.dual_arm_safety import sphere_clearance
        for attempt in range(6):
            book_position = np.asarray(task.objects['book'].get_pose().p)
            live_clearance = sphere_clearance(
                self.arm_safety.spheres('left'),
                self.arm_safety.spheres('right'))
            # The waiting target is only a means to clear the support and the
            # other arm. Controller tracking can leave the carried actor offset
            # from that arbitrary point even though both safety conditions are
            # already satisfied. Chasing the point then drives the wrist back
            # through the shared workspace. Re-evaluate the live purpose after
            # every settled segment and stop as soon as it is achieved.
            if separated_holding_pose_is_safe(
                    book_position, folder_position, required_separation,
                    live_clearance, holding_margin):
                break
            waiting = np.asarray(task._current_ee_pose('right'), dtype=float)
            waiting[:3] += waiting_target - book_position
            normal_margin = self.arm_safety.planning_margin
            self.arm_safety.planning_margin = holding_margin
            try:
                current_waiting = np.asarray(
                    task._current_ee_pose('right'), dtype=float)
                correction_distance = float(np.linalg.norm(
                    waiting[:3] - current_waiting[:3]))
                correction_steps = max(1, int(math.ceil(
                    correction_distance / max(.5 * clearance, 1e-9))))
                corrections = []
                corrected_qpos = entity.get_qpos().copy()
                failure = None
                for fraction in np.linspace(
                        1. / correction_steps, 1., correction_steps):
                    waypoint = current_waiting + fraction * (
                        waiting - current_waiting)
                    correction, next_qpos = self._plan_arm_motion(
                        'right', waypoint.tolist(), corrected_qpos)
                    if next_qpos is None:
                        failure = correction
                        break
                    corrections.append(correction)
                    corrected_qpos = next_qpos
            finally:
                self.arm_safety.planning_margin = normal_margin
            if failure is not None:
                reason = failure.get('failure_status', failure.get('status'))
                raise DynamicSimError(
                    f'No strictly separated correction for held book: {reason}')
            for correction_index, correction in enumerate(corrections):
                self._execute_planned_motion(
                    'right', correction,
                    f'held_object_wait_correction_route_{correction_index}')
            parked = entity.get_qpos()[indices].copy()
            self._execute_planned_motion('right', {
                'position': np.repeat(parked[None,:], 150, axis=0),
                'velocity': np.zeros((150,len(parked)))}, 'book_wait_settle')
        book_position = np.asarray(task.objects['book'].get_pose().p)
        live_clearance = sphere_clearance(
            self.arm_safety.spheres('left'), self.arm_safety.spheres('right'))
        if not separated_holding_pose_is_safe(
                book_position, folder_position, required_separation,
                live_clearance, holding_margin):
            raise DynamicSimError(
                'Held book did not reach a live-geometry separated waiting pose')
        self.arm_safety.check_live()
        saved = (self.held, self.held_orientation, task._attached_actor,
                 task._attached_arm, task._ee_to_actor)
        position = task.objects['folder'].get_pose().p.copy()
        left_root = np.asarray(task.robot.left_entity.get_root_pose().p[:2], dtype=float)
        toward_arm = left_root - position[:2]
        toward_arm /= max(float(np.linalg.norm(toward_arm)), 1e-9)
        contact = position[:2] + toward_arm * (.35 * float(min(folder_size)))
        top = position[2] + task.object_geometry['folder']['height']
        self._motion_target = 'folder'
        if not task.is_left_gripper_open():
            self._gripper('left', False)
        self._motion('left', task._top_down_pose(contact, top, 'left', clearance=.07), 'folder_press_approach')
        self._motion('left', task._top_down_pose(contact, top, 'left', clearance=.003), 'folder_press_contact')
        # Preserve the held book while the other wrist turns the support.
        book_pose = task.objects['book'].get_pose()
        self._pin_released_pose('book', book_pose)
        try:
            task._attach_actor('folder', 'left')
            self.held = ('folder', 'left')
            self._rotate_held_to_yaw('folder', 'left', 0., allowed_contacts={'book'})
            if folder_target is not None:
                slide = np.asarray(task._current_ee_pose('left'), dtype=float)
                slide[:2] += folder_target - task.objects['folder'].get_pose().p[:2]
                self._motion('left', slide.tolist(), 'folder_slide_to_image_goal')
                self._motion('left', slide.tolist(), 'folder_slide_settle')
            folder_pose = task.objects['folder'].get_pose()
            import sapien
            # Preserve the table-supported height; only the wrist's planar
            # displacement/yaw are used for this assisted press-turn.
            p = folder_pose.p.copy(); p[2] = position[2]
            folder_pose = sapien.Pose(p, task._yaw_quat(
                math.degrees(self._object_yaw('folder')), task.RIGID_Y_UP_TO_Z_UP))
            task.objects['folder'].actor.set_pose(folder_pose)
        finally:
            self.held, self.held_orientation, task._attached_actor, task._attached_arm, task._ee_to_actor = saved
            self._pinned_poses.pop('book', None)
        self._motion('left', task._top_down_pose(contact, top, 'left', clearance=.07), 'folder_press_retreat')
        self._gripper('left', False)
        self._motion_target = None
        self._park_arm('left')

    def orientation_status(self, targets, tolerance_degrees=3.0):
        result = {}
        for name, target in targets.items():
            actual = math.degrees(self._object_yaw(name))
            error = abs((actual - target + 180.0) % 360.0 - 180.0)
            result[name] = {'target_yaw_degrees': target, 'actual_yaw_degrees': actual,
                            'error_degrees': error, 'passed': error <= tolerance_degrees}
        return result

    @staticmethod
    def _rectangles_overlap(center_a, size_a, yaw_a, center_b, size_b, yaw_b, margin):
        def axes(yaw):
            c, s = math.cos(yaw), math.sin(yaw)
            return (np.array([c, s]), np.array([-s, c]))
        a_axes, b_axes = axes(yaw_a), axes(yaw_b)
        delta = np.asarray(center_b, dtype=float) - np.asarray(center_a, dtype=float)
        for axis in (*a_axes, *b_axes):
            ra = sum(abs(float(axis @ basis)) * float(width) / 2
                     for basis, width in zip(a_axes, size_a))
            rb = sum(abs(float(axis @ basis)) * float(width) / 2
                     for basis, width in zip(b_axes, size_b))
            if abs(float(delta @ axis)) > ra + rb + margin:
                return False
        return True

    def _clear_transfer_corridor(self, name, start_xy, end_xy, carried_bottom, support=None, carried_top=None):
        """Reject a low carried-object sweep across another tabletop object."""
        own = self.task.object_geometry[name].get("footprint")
        if own is None:
            return True
        own_yaw = self._object_yaw(name)
        start = np.asarray(start_xy, dtype=float)
        segment = np.asarray(end_xy, dtype=float) - start
        obstacles = [(other, actor, self.task.object_geometry[other])
                     for other, actor in self.task.objects.items()
                     if other != name and other != support]
        monitor_overlaps = [not self._monitor_clear(
            name, start + fraction * segment, carried_bottom,
            carried_top if carried_top is not None else
            carried_bottom + self.task.object_geometry[name]['height'], .01)
            for fraction in np.linspace(0, 1, 17)]
        if monitor_overlaps[0]:
            first_clear = next((i for i, value in enumerate(monitor_overlaps)
                                if not value), len(monitor_overlaps))
            if any(monitor_overlaps[first_clear:]):
                return False
        elif any(monitor_overlaps):
            return False
        for other, actor, geometry in obstacles:
            footprint = geometry.get("footprint")
            if footprint is None:
                continue
            center = np.asarray(actor.get_pose().p[:2], dtype=float)
            obstacle_top = float(actor.get_pose().p[2] + geometry["height"])
            if carried_bottom < obstacle_top + .01:
                overlaps = [self._rectangles_overlap(
                    start + fraction * segment, own, own_yaw, center, footprint,
                    self._object_yaw(other), .01) for fraction in np.linspace(0, 1, 17)]
                # A picked object may start above its support while their XY
                # footprints overlap.  Permit a single contiguous exit from
                # that initial overlap, but reject any later entry/re-entry.
                if overlaps[0]:
                    first_clear = next((i for i, value in enumerate(overlaps) if not value), len(overlaps))
                    if any(overlaps[first_clear:]):
                        return False
                elif any(overlaps):
                    return False
        return True

    def _transfer_route(self, name, start_xy, end_xy, carried_bottom, support=None):
        """Find a live, footprint-aware polyline around tabletop obstacles."""
        import heapq
        start = np.asarray(start_xy, dtype=float)
        goal = np.asarray(end_xy, dtype=float)
        if self._clear_transfer_corridor(name, start, goal, carried_bottom, support):
            return [goal]
        # The lattice is camera/table normalized and regenerated for every
        # action. It contains no saved scene coordinate or object ordering.
        values = np.linspace(.10, .90, 9)
        nodes = [np.r_[np.array([u, v]), 1.] @ self.transform
                 for v in values for u in values]
        nodes.extend([start, goal])
        start_i, goal_i = len(nodes) - 2, len(nodes) - 1
        neighbors = {i: [] for i in range(len(nodes))}
        step = max(np.linalg.norm(self.transform[0]), np.linalg.norm(self.transform[1])) / 8.
        for i, first in enumerate(nodes):
            distances = sorted((float(np.linalg.norm(first-second)), j)
                               for j, second in enumerate(nodes) if i != j)
            for distance, j in distances[:12]:
                if distance > 1.8 * step and i not in {start_i, goal_i}:
                    continue
                if self._clear_transfer_corridor(
                        name, first, nodes[j], carried_bottom, support):
                    neighbors[i].append((distance, j))
        queue = [(0., start_i)]
        cost, previous = {start_i: 0.}, {}
        while queue:
            value, node = heapq.heappop(queue)
            if value != cost.get(node):
                continue
            if node == goal_i:
                break
            for distance, nxt in neighbors[node]:
                proposed = value + distance
                if proposed < cost.get(nxt, float('inf')):
                    cost[nxt] = proposed
                    previous[nxt] = node
                    heapq.heappush(queue, (proposed, nxt))
        if goal_i not in cost:
            return None
        path, node = [], goal_i
        while node != start_i:
            path.append(np.asarray(nodes[node], dtype=float))
            node = previous[node]
        path.reverse()
        return path

    def _plan_transfer_waypoints(self, name, arm, actor_pos, ee_pose, candidate,
                                 bottom, transfer_clearance, qpos, support=None):
        """Try geometry-short routes and remove each edge that fails live IK."""
        import heapq
        carried_bottom = min(actor_pos[2], bottom + transfer_clearance)
        start = np.asarray(actor_pos[:2], dtype=float)
        goal = np.asarray(candidate, dtype=float)
        footprint = self.task.object_geometry[name].get("footprint")
        table_span = max(np.linalg.norm(self.transform[0]), np.linalg.norm(self.transform[1]))
        object_span = max(footprint) if footprint is not None else table_span / 8.
        divisions = int(np.clip(np.ceil(table_span / max(object_span, table_span / 8.)), 5, 7))
        values = np.linspace(.08, .92, divisions + 1)
        nodes = [np.r_[np.array([u, v]), 1.] @ self.transform for v in values for u in values]
        grid_count = len(nodes); nodes.extend([start, goal])
        start_i, goal_i, width = grid_count, grid_count + 1, len(values)
        candidates = {i: set() for i in range(len(nodes))}
        for row in range(width):
            for column in range(width):
                node = row * width + column
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = row + dr, column + dc
                        if (dr or dc) and 0 <= rr < width and 0 <= cc < width:
                            candidates[node].add(rr * width + cc)
        for special in (start_i, goal_i):
            closest = sorted(range(grid_count), key=lambda i: np.linalg.norm(nodes[i]-nodes[special]))
            for node in closest[:max(6, width)]:
                candidates[special].add(node); candidates[node].add(special)
        candidates[start_i].add(goal_i)
        clear_cache, blocked, last_failure = {}, set(), None

        def clear_edge(first, second):
            key = (min(first, second), max(first, second))
            if key not in clear_cache:
                clear_cache[key] = self._clear_transfer_corridor(
                    name, nodes[first], nodes[second], carried_bottom, support)
            return clear_cache[key]

        def shortest():
            queue = [(0., start_i)]; cost = {start_i: 0.}; previous = {}
            while queue:
                value, node = heapq.heappop(queue)
                if value != cost.get(node): continue
                if node == goal_i: break
                for nxt in candidates[node]:
                    if (node, nxt) in blocked or not clear_edge(node, nxt): continue
                    proposed = value + float(np.linalg.norm(nodes[nxt]-nodes[node]))
                    if proposed < cost.get(nxt, float('inf')):
                        cost[nxt] = proposed; previous[nxt] = node
                        heapq.heappush(queue, (proposed, nxt))
            if goal_i not in cost: return None
            path = []; node = goal_i
            while node != start_i:
                path.append((previous[node], node)); node = previous[node]
            return list(reversed(path))

        for _ in range(3):
            route = shortest()
            if route is None: break
            plans, current_qpos = [], np.asarray(qpos).copy()
            for first, nxt in route:
                high = ee_pose.copy(); high[:2] += nodes[nxt] - actor_pos[:2]
                high[2] += bottom - actor_pos[2] + transfer_clearance
                plan, next_qpos = self._plan_arm_motion(arm, high.tolist(), current_qpos)
                if next_qpos is None:
                    last_failure = plan; blocked.add((first, nxt)); break
                plans.append(plan); current_qpos = next_qpos
            else:
                return plans, current_qpos
        return None, last_failure

    @staticmethod
    def _nearby_targets(xy, *, stacked=False):
        yield np.asarray(xy, dtype=float)
        if stacked:
            return
        # Search nearby table positions, biased toward the front of the desk.
        for radius in (.01, .02, .024, .04, .08, .12, .16, .20):
            for angle in (math.pi / 2, 3 * math.pi / 4, math.pi / 4,
                          math.pi, 0, 5 * math.pi / 4, 7 * math.pi / 4, 3 * math.pi / 2):
                yield np.asarray(xy, dtype=float) + radius * np.array([math.cos(angle), math.sin(angle)])

    def _plan_arm_motion(self, arm, pose, qpos):
        robot = self.task.robot
        planner = robot.left_plan_path if arm == "left" else robot.right_plan_path
        result = planner(pose, last_qpos=qpos)
        if result.get("status") != "Success":
            return result, None
        if self.arm_safety is not None:
            from script.dual_arm_safety import ArmClearanceError
            try:
                self.arm_safety.validate({f"{arm}_arm": result}, start_qpos={arm: qpos})
            except ArmClearanceError as exc:
                return {"status": "InterArmClearanceRejected", "failure_status": str(exc)}, None
            collision = self._trajectory_obstacle_collision(arm, qpos, result)
            if collision is not None:
                return {"status": "SceneObstacleRejected",
                        "failure_status": f"trajectory intersects {collision}"}, None
        entity = robot.left_entity if arm == "left" else robot.right_entity
        arm_joints = robot.left_arm_joints if arm == "left" else robot.right_arm_joints
        names = [joint.get_name() for joint in entity.get_active_joints()]
        indices = [names.index(joint.get_name()) for joint in arm_joints]
        next_qpos = np.asarray(qpos).copy()
        next_qpos[indices] = np.asarray(result["position"])[-1]
        return result, next_qpos

    def _plan_arm_motions(self, arm, poses, qpos):
        """Plan independent targets from one joint state in one GPU batch."""
        # Test doubles and non-Curobo backends retain the scalar contract.
        robot = getattr(self.task, "robot", None)
        batch_planner = getattr(robot, arm + "_plan_multi_path", None) if robot is not None else None
        if batch_planner is None or "_plan_arm_motion" in self.__dict__:
            return [self._plan_arm_motion(arm, pose, qpos) for pose in poses]
        if not poses:
            return []
        # Curobo's captured batch graph is warmed at ROTATE_NUM. Pad without
        # changing the returned logical candidates so the graph shape stays
        # fixed across A* expansions.
        import envs._GLOBAL_CONFIGS as configs
        batch_size = int(configs.ROTATE_NUM)
        padded = list(poses)
        if len(padded) < batch_size:
            padded.extend([padded[-1]] * (batch_size - len(padded)))
        batch = batch_planner(padded, last_qpos=qpos)
        statuses = list(batch.get("status", []))[:len(poses)]
        positions = batch.get("position")
        velocities = batch.get("velocity")
        results = []
        for index, status in enumerate(statuses):
            if status != "Success" or positions is None:
                results.append(({"status": str(status)}, None))
                continue
            plan = {"status": "Success", "position": np.asarray(positions[index]),
                    "velocity": np.asarray(velocities[index])}
            entity = robot.left_entity if arm == "left" else robot.right_entity
            arm_joints = robot.left_arm_joints if arm == "left" else robot.right_arm_joints
            names = [joint.get_name() for joint in entity.get_active_joints()]
            indices = [names.index(joint.get_name()) for joint in arm_joints]
            next_qpos = np.asarray(qpos).copy()
            next_qpos[indices] = np.asarray(plan["position"])[-1]
            results.append((plan, next_qpos))
        while len(results) < len(poses):
            results.append(({"status": "Failure"}, None))
        return results

    def _trajectory_obstacle_collision(self, arm, start_qpos, plan):
        """Preflight arm spheres against live oriented object volumes."""
        if self.arm_safety is None:
            return None
        moving_name = (self.held[0] if self.held else self._motion_target)
        side_preflight = bool(getattr(self, "_side_grasp_preflight", False)
                              and self.held is None and moving_name in self.task.objects)
        moving_actor = self.task.objects.get(moving_name) if side_preflight else None
        moving_geometry = (self.task.object_geometry.get(moving_name)
                           if side_preflight else None)
        obstacle_actors = []
        for name, actor in self.task.objects.items():
            if name == self._motion_target or (self.held and name == self.held[0]):
                continue
            geometry = self.task.object_geometry[name]
            if (side_preflight and moving_geometry is not None
                    and moving_geometry.get("footprint") is not None
                    and geometry.get("footprint") is not None):
                moving_position = np.asarray(moving_actor.get_pose().p, dtype=float)
                moving_bottom = float(moving_position[2]
                                      + moving_geometry.get("bottom_offset", 0.0))
                support_bottom = float(actor.get_pose().p[2]
                                       + geometry.get("bottom_offset", 0.0))
                support_top = support_bottom + float(geometry["height"])
                # Contact settling can put the measured lower surface a few
                # millimetres into the support collision skin.
                if (-.015 <= moving_bottom - support_top <= .03
                        and support_bottom < moving_bottom
                        and self._rectangles_overlap(
                            moving_position[:2], moving_geometry["footprint"],
                            self._object_yaw(moving_name), actor.get_pose().p[:2],
                            geometry["footprint"], self._object_yaw(name), 0.0)):
                    # Entering the immediate neighbourhood of the object being
                    # grasped necessarily overlaps its direct support in a
                    # coarse sphere model. Runtime contacts still protect all
                    # non-target actors during the actual motion.
                    continue
            if geometry.get("footprint") is not None:
                obstacle_actors.append((name, actor, geometry))
        monitor = getattr(self.task, "monitor", None)
        monitor_spec = getattr(self.task, "MONITOR_SPEC", None)
        held_geometry = (self.task.object_geometry[moving_name]
                         if moving_name in self.task.object_geometry else None)
        flat_carried = bool(held_geometry and held_geometry.get("footprint") is not None
                            and held_geometry["height"] < min(held_geometry["footprint"]))
        if (monitor is not None and monitor_spec is not None
                and self._motion_target != "monitor" and not flat_carried):
            geometry = getattr(self.task, "object_geometry_by_spec", {}).get(id(monitor_spec))
            if geometry is not None and geometry.get("footprint") is not None:
                obstacle_actors.append(("monitor", monitor, geometry))
        if not obstacle_actors:
            return None
        obstacles = []
        for name, actor, geometry in obstacle_actors:
            center = np.asarray(actor.get_pose().p, dtype=float)
            size = np.asarray(geometry["footprint"], dtype=float)
            yaw = self._object_yaw(name)
            c, s = math.cos(yaw), math.sin(yaw)
            obstacles.append((name, center, size, c, s, center[2],
                              center[2] + float(geometry["height"])))
        entity = getattr(self.task.robot, arm + '_entity')
        names = [joint.get_name() for joint in entity.get_active_joints()]
        indices = [names.index(joint.get_name()) for joint in getattr(self.task.robot, arm + '_arm_joints')]
        qpos = np.asarray(start_qpos, dtype=float).copy()
        waypoints = np.asarray(plan["position"], dtype=float)
        if len(waypoints) > 40:
            waypoints = waypoints[np.linspace(0, len(waypoints)-1, 40, dtype=int)]
        initial_contacts = None
        cleared_initial = set()
        # A grasp necessarily enters the target's local neighbourhood (for
        # example a cup resting on a book). Runtime contact monitoring remains
        # active there; the larger tracking allowance applies after pickup.
        tall_carried = bool(held_geometry and held_geometry.get("footprint") is not None
                            and held_geometry["height"] >= min(held_geometry["footprint"]))
        obstacle_margin = (max(.004, .6 * self.arm_safety.minimum_clearance)
                           if tall_carried or getattr(self, "_side_grasp_preflight", False)
                           else .004)
        for waypoint in waypoints:
            qpos[indices] = waypoint
            contacts = set()
            for sx, sy, sz, radius in self.arm_safety.spheres(arm, qpos):
                for name, center, size, c, s, bottom, top in obstacles:
                    delta = np.array([sx, sy]) - center[:2]
                    local = np.array([c * delta[0] + s * delta[1],
                                      -s * delta[0] + c * delta[1]])
                    horizontal = np.maximum(np.abs(local) - size / 2, 0.)
                    vertical = max(bottom - sz, 0., sz - top)
                    if math.sqrt(float(horizontal @ horizontal) + vertical * vertical) <= radius + obstacle_margin:
                        contacts.add(name)
            if initial_contacts is None:
                initial_contacts = contacts
            else:
                cleared_initial.update(initial_contacts - contacts)
                new_or_reentered = contacts - (initial_contacts - cleared_initial)
                if new_or_reentered:
                    return sorted(new_or_reentered)[0]
        return None

    def _execute_planned_motion(self, arm, result, label, allowed_contacts=()):
        control = {"left_arm": None, "left_gripper": None,
                   "right_arm": None, "right_gripper": None}
        control[f"{arm}_arm"] = result
        contact_with = set()
        original_sync = self.task._sync_attached_actor
        held_name = self.held[0] if self.held else None
        executed_result = result
        if (held_name is not None and "hold" not in label
                and np.asarray(result.get("position", [])).ndim == 2):
            # A carried payload increases controller lag. Hold each validated
            # joint target for a scale-derived number of physics ticks so the
            # measured path follows the collision-checked path instead of
            # cutting its corners between sparse drive targets.
            minimum = float(self.arm_safety.minimum_clearance) if self.arm_safety else .05
            margin = float(self.arm_safety.planning_margin) if self.arm_safety else 0.
            slowdown = max(1, int(math.ceil(1. + margin / max(minimum, 1e-9))))
            if slowdown > 1:
                # Repeating sparse waypoints still leaves a discontinuity at
                # every boundary. Interpolate from the measured joint state at
                # half the swept-collision sampling step, then command zero
                # velocity. This makes the physical controller follow the same
                # joint-space segments that the safety validator approved.
                validation_step = float(self.arm_safety.stats.get(
                    'trajectory_joint_step_rad', .005))
                control_step = validation_step / 50.
                entity = getattr(self.task.robot, arm + '_entity')
                indices = self.arm_safety.indices[arm]
                previous = np.asarray(entity.get_qpos(), dtype=float)[indices]
                samples = []
                for target in np.asarray(result['position'], dtype=float):
                    count = max(1, int(math.ceil(
                        np.max(np.abs(target-previous)) / control_step)))
                    samples.extend(previous + alpha * (target-previous)
                                   for alpha in np.linspace(1. / count, 1., count))
                    previous = target
                positions = np.asarray(samples, dtype=float)
                settle_ticks = max(150, 50 * slowdown)
                positions = np.vstack((positions, np.repeat(
                    positions[-1][None, :], settle_ticks, axis=0)))
                velocities = np.zeros_like(positions)
                executed_result = dict(result, position=positions, velocity=velocities)
        if held_name and (label in {"transfer", "transfer_hold", "transfer_correction",
                                   "lower", "release_correction"}
                          or label.startswith("transfer_route_")):
            held_entity = self.task.objects[held_name].get_name()
            allowed_contacts = set(allowed_contacts)
            other_entities = {actor.get_name(): name for name, actor in self.task.objects.items()
                              if name != held_name and name not in allowed_contacts}
            if getattr(self.task, "monitor", None) is not None:
                other_entities[self.task.monitor.get_name()] = "monitor"
            robot_entities = set()
            for articulation in (self.task.robot.left_entity, self.task.robot.right_entity):
                for link in articulation.get_links():
                    entity = getattr(link, "entity", link)
                    robot_entities.add(entity.get_name())
            allowed_robot_contacts = ({"book"} if label in {"lower", "release_correction"}
                                      else set())
            def detected_contacts():
                carried_contacts, robot_contacts = set(), set()
                for contact in self.task.scene.get_contacts():
                    try:
                        names = {body.entity.name for body in contact.bodies}
                    except (AttributeError, TypeError):
                        continue
                    if held_entity in names:
                        carried_contacts.update(other_entities[entity] for entity in names
                                                if entity in other_entities)
                    if names & robot_entities:
                        robot_contacts.update(other_entities[entity] for entity in names
                                              if entity in other_entities)
                return carried_contacts | (robot_contacts - allowed_robot_contacts)
            initial_contacts = detected_contacts()
            cleared_initial = set()
            def checked_sync():
                original_sync()
                current_contacts = detected_contacts()
                cleared_initial.update(initial_contacts - current_contacts)
                new_or_reentered = current_contacts - (initial_contacts - cleared_initial)
                if new_or_reentered:
                    contact_with.update(new_or_reentered)
                    raise DynamicSimError(
                        f"Robot or carried {held_name} touched other objects during "
                        f"{label}: {sorted(contact_with)}",
                        blockers=sorted(contact_with))
            self.task._sync_attached_actor = checked_sync
        try:
            control[f"{arm}_arm"] = executed_result
            if self.arm_safety is not None:
                self.arm_safety.active_context = {
                    'label': label, 'moving_arm': arm,
                    'held_object': held_name,
                    'planned_endpoint': np.asarray(
                        executed_result['position'][-1], dtype=float).tolist(),
                }
            self.task.take_dense_action(control, save_freq=None)
        finally:
            self.task._sync_attached_actor = original_sync
            if self.arm_safety is not None:
                self.arm_safety.active_context = None
        if self.arm_safety is not None:
            entity = getattr(self.task.robot, arm + '_entity')
            indices = self.arm_safety.indices[arm]
            endpoint = np.asarray(executed_result['position'][-1], dtype=float)
            actual = np.asarray(entity.get_qpos(), dtype=float)[indices]
            velocity = np.asarray(entity.get_qvel(), dtype=float)[indices]
            from script.dual_arm_safety import sphere_clearance
            sample = {
                'step': self.step_index, 'label': label, 'arm': arm,
                'held_object': held_name,
                'joint_endpoint_error_rad': float(np.max(np.abs(actual-endpoint))),
                'joint_speed_rad_s': float(np.max(np.abs(velocity))),
                'inter_arm_clearance_m': sphere_clearance(
                    self.arm_safety.spheres('left'), self.arm_safety.spheres('right')),
            }
            with (self.output_dir / 'motion_tracking.jsonl').open('a') as stream:
                stream.write(json.dumps(sample) + '\n')
        self._capture_motion_frame(label)
        if contact_with:
            raise DynamicSimError(
                f"Robot or carried {held_name} touched other objects during "
                f"{label}: {sorted(contact_with)}",
                blockers=sorted(contact_with))

    def _snapshot_robot_state(self):
        """Capture a reversible controller state for one high-level action."""
        robot = getattr(self.task, "robot", None)
        if robot is None:
            return None
        state = {}
        for arm in ("left", "right"):
            entity = getattr(robot, arm + "_entity", None)
            if entity is None or not hasattr(entity, "get_qpos"):
                continue
            state[arm] = {
                "qpos": np.asarray(entity.get_qpos(), dtype=float).copy(),
                "qvel": (np.asarray(entity.get_qvel(), dtype=float).copy()
                         if hasattr(entity, "get_qvel") else None),
            }
        return state

    def _restore_robot_state(self, state):
        """Finish a failed-action rollback from its last known safe state.

        Runtime contact can leave a controller in self-collision, where neither
        Cartesian IK nor a joint trajectory has a valid starting state.  Object
        poses are already transactional; restoring the two empty-arm
        articulations makes the same transaction complete.  The snapshot comes
        from the immediately preceding live state, never from a recorded run.
        """
        if not state:
            return
        robot = self.task.robot
        for arm, values in state.items():
            entity = getattr(robot, arm + "_entity")
            qpos = values["qpos"]
            entity.set_qpos(qpos)
            if values["qvel"] is not None and hasattr(entity, "set_qvel"):
                entity.set_qvel(np.zeros_like(values["qvel"]))
            names = [joint.get_name() for joint in entity.get_active_joints()]
            for joint in entity.get_active_joints():
                index = names.index(joint.get_name())
                target = np.asarray(qpos[index]).reshape(-1)
                joint.set_drive_target(float(target[0]))
                if hasattr(joint, "set_drive_velocity_target"):
                    joint.set_drive_velocity_target(0.0)
        self.task.scene.step()
        self._capture_motion_frame("failed_action_robot_rollback")

    def _clear_attachment_velocity(self, name):
        """A kinematically followed actor should be at rest before release."""
        actor = self.task.objects[name]
        entity = getattr(actor, "actor", actor)
        for component in getattr(entity, "get_components", lambda: [])():
            if hasattr(component, "set_linear_velocity") and hasattr(component, "set_angular_velocity"):
                component.set_linear_velocity(np.zeros(3))
                component.set_angular_velocity(np.zeros(3))

    def _pin_released_pose(self, name, pose):
        """Keep a completed stack stable after a physically staged release."""
        from copy import deepcopy
        actor = getattr(self.task.objects[name], "actor", self.task.objects[name])
        self._pinned_poses[name] = (actor, deepcopy(pose))
        if self._pin_sync_installed:
            return
        base_sync = self.task._sync_attached_actor
        def sync_with_stable_stacks():
            base_sync()
            for pinned_actor, pinned_pose in self._pinned_poses.values():
                pinned_actor.set_pose(pinned_pose)
        self.task._sync_attached_actor = sync_with_stable_stacks
        self._pin_sync_installed = True

    def _place_cup_below_screen(self, xy):
        """Find and execute a live collision-free low insertion for the cup."""
        name, arm = self.held
        task = self.task
        entity = getattr(task.robot, arm + '_entity')
        bottom = task._table_z() + .009
        height = task.object_geometry[name]['height']
        footprint = task.object_geometry[name]['footprint']
        insertion_step = max(footprint)
        transfer_clearance = max(height, insertion_step) * .75
        selected = None
        for candidate in self._nearby_targets(xy):
            if np.linalg.norm(candidate - xy) > .02001:
                continue
            if not self._clear_table_placement(name, candidate, margin=.005):
                continue
            actor = np.asarray(task.objects[name].get_pose().p, dtype=float)
            current = np.asarray(task._current_ee_pose(arm), dtype=float)
            initial_qpos = entity.get_qpos().copy()
            # Sample approach directions and distances from the live object
            # scale. This replaces the scene-specific fixed front coordinate.
            angles = (-math.pi / 2, -3 * math.pi / 4, -math.pi / 4,
                      math.pi, 0., math.pi / 2, 3 * math.pi / 4, math.pi / 4)
            for distance_scale in (1.25, 1.75, 2.25):
                for angle in angles:
                    direction = np.array([math.cos(angle), math.sin(angle)])
                    approach = candidate + distance_scale * insertion_step * direction
                    if not self._clear_transfer_corridor(name, approach, candidate, bottom):
                        continue
                    route_plans, route_qpos = self._plan_transfer_waypoints(
                        name, arm, actor, current, approach, bottom,
                        transfer_clearance, initial_qpos)
                    if route_plans is None:
                        continue
                    tail_plans = []
                    qpos = route_qpos
                    locations = (np.r_[approach, bottom], np.r_[candidate, bottom],
                                 np.r_[approach, bottom],
                                 np.r_[approach, bottom + transfer_clearance])
                    for location in locations:
                        pose = current.copy()
                        pose[:3] += location - actor
                        result, qpos = self._plan_arm_motion(arm, pose.tolist(), qpos)
                        if qpos is None:
                            break
                        tail_plans.append(result)
                    if len(tail_plans) == len(locations):
                        selected = candidate, route_plans, tail_plans
                        break
                if selected is not None:
                    break
            if selected is not None:
                break
        if selected is None:
            raise DynamicSimError('No safe low insertion path for cup at the image goal')
        candidate, route_plans, tail_plans = selected
        for route_index, plan in enumerate(route_plans):
            self._execute_planned_motion(arm, plan, f'transfer_route_{route_index}')
        self._execute_planned_motion(arm, tail_plans[0], 'lower')
        self._execute_planned_motion(arm, tail_plans[1], 'transfer')
        if np.linalg.norm(candidate - task.objects[name].get_pose().p[:2]) > .025:
            raise DynamicSimError('Cup failed to track the planned low insertion')
        self._gripper(arm, False)
        self._clear_attachment_velocity(name)
        task._attached_actor = task._attached_arm = task._ee_to_actor = None
        self.held = self.held_orientation = None
        self._motion_target = name
        self._execute_planned_motion(arm, tail_plans[2], 'cup_retreat_from_goal')
        self._execute_planned_motion(arm, tail_plans[3], 'cup_retreat_lift')
        self._motion_target = None
        task.delay(1, save_freq=None)
        actual = np.asarray(task.objects[name].get_pose().p[:2])
        error = float(np.linalg.norm(actual - xy))
        self.last_placement = {'object': name, 'requested_xy': np.asarray(xy).tolist(),
                               'selected_xy': candidate.tolist(), 'actual_xy': actual.tolist(),
                               'placement_error': error, 'mode': 'kinematic_follow_low_insertion'}
        if error > .025:
            raise DynamicSimError(f'Cup moved away from image goal after release: {error:.3f} m')

    def _place(self, action, name):
        if name == "book" and self.held is None:
            components = getattr(getattr(self.task.objects[name], "actor", None), "get_components", lambda: [])()
            if any("RigidStatic" in type(component).__name__ for component in components):
                raise DynamicSimError("Book is a static actor; physical book manipulation requires a dynamic book")
        picked_here = self.held is None
        source_pose = None
        if picked_here:
            from copy import deepcopy
            source_pose = deepcopy(self.task.objects[name].get_pose())
            self._pick(name)
        held_name, arm = self.held
        if held_name != name:
            raise DynamicSimError(f"Held {held_name}, cannot place {name}")
        if name == 'cup' and arm == 'right' and self.image_goal is not None:
            xy, _ = self._destination(action, name)
            if np.linalg.norm(xy - self.image_goal.targets[name]) <= .025:
                return self._place_cup_below_screen(xy)
        target_yaw = (action['parameters'].get('pose') or {}).get('yaw_degrees')
        if target_yaw is not None:
            if name == 'book' and action['kind'] == 'stack':
                # Rotating a wide book near the support sweeps its corner
                # into the folder. Raise the wrist before any yaw change.
                raised = np.asarray(self.task._current_ee_pose(arm), dtype=float)
                raised[2] += .12
                self._motion(arm, raised.tolist(), 'book_rotation_clearance')
                self._motion(arm, raised.tolist(), 'book_rotation_clearance_hold')
                self._align_empty_folder()
            self._rotate_held_to_yaw(name, arm, target_yaw)
        xy, bottom = self._destination(action, name)
        self.last_placement = None
        support = self._name(action["target"]) if action["target"] not in {None, "table", "held above tabletop"} else None
        stacked = action["kind"] == "stack" and support is not None
        if hasattr(self.task, "robot") and hasattr(self.task, "_current_ee_pose"):
            low_clearance = (.002 if stacked else
                             .025 if self.task.object_geometry[name]["height"] < .025 else .007)
            actor_pos = np.asarray(self.task.objects[name].get_pose().p, dtype=float)
            ee_pose = np.asarray(self.task._current_ee_pose(arm), dtype=float)
            entity = self.task.robot.left_entity if arm == "left" else self.task.robot.right_entity
            qpos = entity.get_qpos().copy()
            selection = None
            failures = []
            rejection_counts = {key: 0 for key in
                                ("too_far", "occupied", "corridor", "transfer_ik", "lower_ik", "retreat_ik")}
            for candidate in self._nearby_targets(xy, stacked=stacked):
                if not stacked and np.linalg.norm(candidate - xy) > self.max_placement_deviation + 1e-9:
                    rejection_counts["too_far"] += 1
                    continue
                if not stacked and not self._clear_table_placement(name, candidate):
                    rejection_counts["occupied"] += 1
                    continue
                if name == "cup" and arm == "left":
                    clearances = (.065, .075, .095)
                elif self.task.object_geometry[name]["height"] < .025:
                    clearances = (.095, .075, .065, .045, .035)
                else:
                    clearances = (.095, .075, .065)
                for transfer_clearance in clearances:
                    # Keep the observed grasp transform: shifting the gripper
                    # by the object's desired displacement shifts it there.
                    high = ee_pose.copy()
                    high[:2] += candidate - actor_pos[:2]
                    high[2] += bottom - actor_pos[2] + transfer_clearance
                    low = high.copy()
                    low[2] -= transfer_clearance - low_clearance
                    high_plans, high_result = self._plan_transfer_waypoints(
                        name, arm, actor_pos, ee_pose, candidate, bottom,
                        transfer_clearance, qpos, support)
                    if high_plans is None:
                        if high_result is None:
                            rejection_counts["corridor"] += 1
                        else:
                            rejection_counts["transfer_ik"] += 1
                            failures.append(high_result.get("failure_status", high_result.get("status")))
                        continue
                    high_qpos = high_result
                    if high_qpos is None:
                        rejection_counts["transfer_ik"] += 1
                        continue
                    low_plan, low_qpos = self._plan_arm_motion(arm, low.tolist(), high_qpos)
                    if low_qpos is None:
                        rejection_counts["lower_ik"] += 1
                        failures.append(low_plan.get("failure_status", low_plan.get("status")))
                        continue
                    retreat_plan, retreat_qpos = self._plan_arm_motion(arm, high.tolist(), low_qpos)
                    if retreat_qpos is None:
                        rejection_counts["retreat_ik"] += 1
                        failures.append(retreat_plan.get("failure_status", retreat_plan.get("status")))
                        continue
                    selection = (candidate, high_plans, low_plan, retreat_plan, transfer_clearance)
                    break
                if selection is not None:
                    break
            if selection is None:
                blockers = [] if stacked else self._placement_blockers(name, xy)
                if picked_here:
                    self._restore_failed_pick(name, arm, source_pose)
                raise DynamicSimError(
                    f"No collision-free IK placement within {self.max_placement_deviation:.2f} m "
                    f"of requested target for {name}; blockers at target: {blockers}; "
                    f"rejections: {rejection_counts}; planner failures: {failures[-3:]}")
            candidate, high_plans, low_plan, retreat_plan, transfer_clearance = selection
            release_xy = (np.asarray(xy, dtype=float) if not stacked
                          and np.linalg.norm(candidate - xy) <= .08
                          and self._clear_table_placement(name, xy) else candidate)
            for route_index, high_plan in enumerate(high_plans):
                self._execute_planned_motion(arm, high_plan, f"transfer_route_{route_index}")
            endpoint = np.asarray(high_plans[-1]["position"])[-1]
            hold = {"position": np.repeat(endpoint[None, :], 100, axis=0),
                    "velocity": np.zeros((100, len(endpoint)))}
            self._execute_planned_motion(arm, hold, "transfer_hold")
            # Controllers may lag the planned end state. Re-solve from the
            # observed pose instead of executing the predicted next segment.
            correction_limit = (0 if self.task.object_geometry[name]["height"] < .025
                                else 5 if name == "cup" else 2)
            for _ in range(correction_limit):
                actor_pos = np.asarray(self.task.objects[name].get_pose().p, dtype=float)
                if np.linalg.norm(candidate - actor_pos[:2]) <= .02:
                    break
                ee_pose = np.asarray(self.task._current_ee_pose(arm), dtype=float)
                correction = ee_pose.copy()
                correction[:2] += candidate - actor_pos[:2]
                correction[2] += bottom + transfer_clearance - actor_pos[2]
                correction_plan, _ = self._plan_arm_motion(arm, correction.tolist(), entity.get_qpos().copy())
                if correction_plan.get("status") != "Success":
                    break
                self._execute_planned_motion(arm, correction_plan, "transfer_correction")
            actor_pos = np.asarray(self.task.objects[name].get_pose().p, dtype=float)
            ee_pose = np.asarray(self.task._current_ee_pose(arm), dtype=float)
            live_low = ee_pose.copy()
            live_low[:2] += candidate - actor_pos[:2]
            live_low[2] += bottom + low_clearance - actor_pos[2]
            live_low_plan, _ = self._plan_arm_motion(arm, live_low.tolist(), entity.get_qpos().copy())
            if live_low_plan.get("status") != "Success":
                # The validated segment may still be usable when feedback
                # correction moves the observed state outside the IK basin.
                live_low_plan = low_plan
            approximate_release = False
            try:
                self._execute_planned_motion(
                    arm, live_low_plan, "lower",
                    allowed_contacts=({support} if stacked and support else set()))
            except DynamicSimError as exc:
                if self.image_goal is not None:
                    # A contact stop is a failed action, even if an assisted
                    # release could make its final coordinates look correct.
                    raise
                # Thin, awkward objects are kinematically followed once the
                # gripper has carried them over the correct neighborhood.  A
                # link brushing a nearby item during the final descent should
                # stop that descent, then release the followed object at the
                # already validated target instead of leaving it held in a
                # half-completed action.
                if name not in {"keyboard", "phone"} or "during lower" not in str(exc):
                    raise
                approximate_release = True
            if not approximate_release:
                endpoint = np.asarray(live_low_plan["position"])[-1]
                hold_ticks = 120 if stacked else 100
                hold = {"position": np.repeat(endpoint[None, :], hold_ticks, axis=0),
                        "velocity": np.zeros((hold_ticks, len(endpoint)))}
                self._execute_planned_motion(
                    arm, hold, "lower_hold",
                    allowed_contacts=({support} if stacked and support else set()))
                actor_pos = np.asarray(self.task.objects[name].get_pose().p, dtype=float)
                if actor_pos[2] < self.task._table_z() + .005:
                    if name == "book" and stacked:
                        approximate_release = True
                    else:
                        raise DynamicSimError(f"Object {name} did not reach a safe release pose")
                stack_release_tolerance = .13 if name == "book" else .07
                if stacked:
                    stack_error = float(np.linalg.norm(candidate - actor_pos[:2]))
                    if stack_error > stack_release_tolerance:
                        raise DynamicSimError(f"Stacked {name} did not reach its support center")
                    if name == "book" and stack_error > .07:
                        approximate_release = True
                if stacked and abs(float(actor_pos[2] - (bottom + low_clearance))) > .015:
                    raise DynamicSimError(f"Stacked {name} did not descend onto its support")
        else:
            # Lightweight task doubles keep the legacy contract path.
            top = bottom + self.task.object_geometry[name]["height"]
            self._motion(arm, self.task._top_down_pose(xy, top, arm, clearance=.065), "transfer")
            self._motion(arm, self.task._top_down_pose(xy, top, arm, clearance=.007), "lower")
        self._gripper(arm, False)
        if hasattr(self.task, "robot") and hasattr(self.task, "_current_ee_pose"):
            # The user permits object following once the gripper is near the
            # destination.  Do not chase the exact object centre after the
            # fingers open: that extra arm sweep can hit already placed items.
            self._clear_attachment_velocity(name)
            ee_pose = np.asarray(self.task._current_ee_pose(arm), dtype=float)
            retreat_pose = ee_pose.copy()
            retreat_pose[2] += transfer_clearance - low_clearance
            live_retreat, _ = self._plan_arm_motion(arm, retreat_pose.tolist(), entity.get_qpos().copy())
            if live_retreat.get("status") != "Success":
                live_retreat = retreat_plan
            import sapien
            observed_pose = self.task.objects[name].get_pose()
            frozen_position = np.asarray(observed_pose.p, dtype=float).copy()
            frozen_position[:2] = release_xy
            if stacked:
                frozen_position[2] = bottom + low_clearance
            frozen_orientation = (self.held_orientation.copy()
                                  if self.held_orientation is not None
                                  else np.asarray(observed_pose.q, dtype=float).copy())
            frozen_pose = sapien.Pose(frozen_position, frozen_orientation)
            original_sync = self.task._sync_attached_actor
            actor_setter = getattr(self.task.objects[name], "actor", self.task.objects[name])
            attached_before_retreat = self.task._attached_actor
            self.task._attached_actor = None
            def frozen_release_sync():
                actor_setter.set_pose(frozen_pose)
                original_sync()  # Keep video, stable supports, and other hooks alive.
            self.task._sync_attached_actor = frozen_release_sync
            try:
                self._execute_planned_motion(arm, live_retreat, "retreat")
            finally:
                self.task._sync_attached_actor = original_sync
                self.task._attached_actor = attached_before_retreat
        self.task._attached_actor = None
        self.task._attached_arm = None
        self.task._ee_to_actor = None
        self.held = None
        self.held_orientation = None
        if hasattr(self.task, "robot"):
            # Discard contact impulses accumulated while the followed object
            # was frozen between the fingers.  The arm is already clear.
            actor_setter.set_pose(frozen_pose)
            self._clear_attachment_velocity(name)
            if stacked and name == "book":
                self._pin_released_pose(name, frozen_pose)
            self.task.delay(1, save_freq=None)
            if stacked and name == "book":
                # ``delay`` advances raw physics without invoking the task's
                # per-control synchronization hook. Reassert the supported
                # pose once after settling; later robot controls are covered
                # by the persistent stable-stack hook above.
                actor_setter.set_pose(frozen_pose)
                self._clear_attachment_velocity(name)
        if hasattr(self.task, "robot") and hasattr(self.task, "_current_ee_pose"):
            actual = np.asarray(self.task.objects[name].get_pose().p[:2], dtype=float)
            self.last_placement = {"object": name, "requested_xy": np.asarray(xy).tolist(),
                                   "selected_xy": candidate.tolist(), "actual_xy": actual.tolist(),
                                   "selected_distance": float(np.linalg.norm(candidate - xy)),
                                   "placement_error": float(np.linalg.norm(actual - release_xy)),
                                   "mode": ("kinematic_follow_approximate_release" if approximate_release
                                            else "kinematic_follow"),
                                   "candidate_rejections": rejection_counts,
                                   "stacked": stacked}
            if target_yaw is not None:
                self.last_placement['orientation'] = self.orientation_status({name:float(target_yaw)})[name]
                if not self.last_placement['orientation']['passed']:
                    raise DynamicSimError(f'Released {name} outside the target yaw tolerance')
            # The user-level target is an approximate region.  Controller
            # tracking may shift a released object a few centimetres; accept
            # that only when the measured pose is still collision-free.
            if np.linalg.norm(actual - release_xy) > .08:
                raise DynamicSimError(f"Released {name} outside the selected placement tolerance")
            if not stacked and not self._clear_table_placement(name, actual):
                raise DynamicSimError(f"Released {name} too close to another object")
        else:
            self.task._attached_actor = None
            self.task._attached_arm = None
            self.task._ee_to_actor = None
            self.held = None
            self.held_orientation = None
            self._motion(arm, self.task._top_down_pose(xy, top, arm, clearance=.065), "retreat")

    def _restore_failed_pick(self, name, arm, source_pose):
        """Undo a pick when no complete placement trajectory was found."""
        actor = self.task.objects[name]
        current = np.asarray(actor.get_pose().p, dtype=float)
        ee_pose = np.asarray(self.task._current_ee_pose(arm), dtype=float)
        lower = ee_pose.copy()
        lower[:3] += np.asarray(source_pose.p, dtype=float) - current
        self._motion_target = name
        try:
            self._motion(arm, lower.tolist(), "failed_pick_return")
            self._gripper(arm, False)
            self.task._attached_actor = None
            self.task._attached_arm = None
            self.task._ee_to_actor = None
            self.held = None
            self.held_orientation = None
            setter = getattr(actor, "actor", actor)
            setter.set_pose(source_pose)
            self._clear_attachment_velocity(name)
            retreat = np.asarray(self.task._current_ee_pose(arm), dtype=float)
            retreat[2] += max(.04, 2. * self.task.object_geometry[name]["height"])
            try:
                self._motion(arm, retreat.tolist(), "failed_pick_retreat")
            except DynamicSimError:
                # The transaction is already safe: the object is restored and
                # both grippers are open. A later action may choose the other
                # arm or a different retreat route from this observed state.
                pass
        finally:
            self._motion_target = None

    def _push_keyboard(self, action):
        """Move the keyboard by measured physical contact, without attachment."""
        if self.held is not None:
            raise DynamicSimError("Cannot push while an object is held")
        name, arm = "keyboard", "right"
        target, _ = self._destination(action, name)
        actor = self.task.objects[name]
        start_pose = np.asarray(actor.get_pose().p, dtype=float)
        if abs(float(target[0] - start_pose[0])) > .035:
            raise DynamicSimError("Keyboard contact push only supports a near-straight rearward path")
        geometry = self.task.object_geometry[name]
        half_depth = float(geometry["footprint"][1]) / 2
        # ``_top_down_push_pose`` expects the contact point at the finger-side
        # surface, whose centre is about 31 mm above the tool reference used
        # by the scene.  Using table height directly puts the fingers below
        # the thin keyboard and produces no physical contact.
        contact_z = (self.task._table_z()
                     + min(float(geometry["height"]) * .5, .012) + .031)
        start_xy = np.array([start_pose[0], start_pose[1] - half_depth - .004])
        end_xy = np.array([target[0], target[1] - half_depth - .004])
        self._gripper(arm, False)
        self._motion(arm, self.task._top_down_push_pose(start_xy, contact_z, clearance=.070),
                     "push_approach")
        self._gripper(arm, True)
        contact_pose = self.task._top_down_push_pose(start_xy, contact_z)
        self._motion(arm, contact_pose, "push_contact")
        self._motion(arm, contact_pose, "push_contact_settle")
        previous = start_pose.copy()
        for index, waypoint in enumerate(np.linspace(start_xy, end_xy, 10)[1:], start=1):
            self.task._slow_contact = True
            try:
                waypoint_pose = self.task._top_down_push_pose(waypoint, contact_z)
                self._motion(arm, waypoint_pose, f"push_{index}")
                self._motion(arm, waypoint_pose, f"push_{index}_settle")
            finally:
                self.task._slow_contact = False
            current = np.asarray(actor.get_pose().p, dtype=float)
            if index >= 3 and np.linalg.norm(current[:2] - previous[:2]) < .0002:
                raise DynamicSimError("Keyboard did not slide under physical contact")
            previous = current
        self.task.delay(.5, save_freq=None)
        actual = np.asarray(actor.get_pose().p[:2], dtype=float)
        retreat = np.asarray(self.task._current_ee_pose(arm), dtype=float)
        retreat[1] -= .04
        retreat[2] += .07
        self._motion(arm, retreat.tolist(), "push_retreat")
        self._gripper(arm, False)
        self.last_placement = {"object": name, "requested_xy": target.tolist(),
                               "actual_xy": actual.tolist(), "mode": "physical_push",
                               "placement_error": float(np.linalg.norm(actual - target))}
        if self.last_placement["placement_error"] > .05:
            raise DynamicSimError("Physical keyboard push did not reach the requested region")

    def step(self, request):
        action = request_action(request)
        self.last_placement = None
        name = self._name(action["source"])
        kind = action["kind"]
        requested_arm = (action.get("parameters", {}).get("pose") or {}).get("arm")
        held_before = self.held
        source_pose = None
        scene_poses = None
        robot_state = None
        if held_before is None and kind in {"move", "place", "stack", "insert", "align", "reorient", "attach"}:
            from copy import deepcopy
            source_pose = deepcopy(self.task.objects[name].get_pose())
            # Failed physical execution is a transaction: recovery must not
            # leave unrelated objects displaced by the collision or retreat.
            # These poses are used only for rollback, never for planning.
            scene_poses = {object_name: deepcopy(actor.get_pose())
                           for object_name, actor in self.task.objects.items()}
            robot_state = self._snapshot_robot_state()
        self._requested_arm = ((name, requested_arm)
                               if requested_arm in {"left", "right"} else None)
        try:
            if (self.graph_validation_mode
                    and kind in {"move", "place", "stack", "insert", "align",
                                 "reorient", "attach"}):
                if self.symbolic_grasp_validation:
                    self.symbolic_grasp_controller.place(action, name)
                elif self.object_motion_validation:
                    self.object_motion_controller.place(action, name)
                elif self.hybrid_execution:
                    self.hybrid_controller.place(action, name)
                else:
                    self._graph_validation_place(action, name)
            elif kind == "pick":
                self._pick(name)
            elif kind == "push":
                if name != "keyboard":
                    raise DynamicSimError("Physical push is currently validated only for the keyboard")
                self._push_keyboard(action)
            elif kind in {"move", "place", "stack", "insert", "align", "reorient", "attach"}:
                self._place(action, name)
            elif kind in {"open", "close"}:
                obj = self.task.objects[name]
                joint = getattr(obj, "actor", obj)
                if not hasattr(joint, "set_qpos") or not hasattr(joint, "get_qlimits"):
                    raise DynamicSimError(f"{name} has no actuated articulation for {kind}")
                limits = np.asarray(joint.get_qlimits())
                qpos = np.asarray(joint.get_qpos()).copy()
                qpos[0] = limits[0, 1 if kind == "open" else 0]
                joint.set_qpos(qpos)
                self.task.scene.step()
            else:
                raise DynamicSimError(f"{kind} needs an explicit scene-specific executor")
        except Exception:
            if self.graph_validation_mode:
                # Preserve the measured state, including an active grasp.
                # A failed trajectory must never rewind actors or joints.
                raise
            if (held_before is None and source_pose is not None and self.held is not None
                    and self.held[0] == name):
                try:
                    self._restore_failed_pick(name, self.held[1], source_pose)
                except Exception:
                    # Mechanical withdrawal can itself be unreachable after a
                    # runtime contact. Complete the failed-action transaction
                    # in simulator state so the next observed action starts
                    # with no stale attachment or closed-gripper deadlock.
                    failed_arm = self.held[1]
                    try:
                        self.task.move(self.task.open_gripper(
                            self.task._arm_tag(failed_arm)))
                    except Exception:
                        pass
                    self.task._attached_actor = None
                    self.task._attached_arm = None
                    self.task._ee_to_actor = None
                    self.held = None
                    self.held_orientation = None
                    setter = getattr(self.task.objects[name], "actor", self.task.objects[name])
                    setter.set_pose(source_pose)
                    self._clear_attachment_velocity(name)
            if scene_poses is not None:
                for object_name, pose in scene_poses.items():
                    actor = self.task.objects[object_name]
                    setter = getattr(actor, "actor", actor)
                    setter.set_pose(pose)
                    self._clear_attachment_velocity(object_name)
            # Mechanical recovery above is useful when it succeeds because it
            # records a continuous withdrawal.  Always finish at the exact
            # pre-action controller state so a failed contact cannot poison
            # the IK start state for every later graph edge.
            self._restore_robot_state(robot_state)
            raise
        finally:
            self._requested_arm = None
        if self.arm_safety is not None and self.held is None and not self.graph_validation_mode:
            # A supported handoff completes with the releasing arm outside
            # the shared workspace before the receiving arm approaches.
            self.park_arms()
        self.step_index += 1
        observation = self.observe()
        observation["action"] = action
        if self.last_placement is not None:
            observation["placement"] = self.last_placement
        return observation

    def evaluate(self):
        return bool(self.task.check_success())

    def _park_arm(self, arm):
        if self.held is not None and self.held[1] == arm:
            raise DynamicSimError(f"Cannot park {arm} while holding an object")
        pose = np.asarray(getattr(self.task.robot, f"{arm}_original_pose"), dtype=float).copy()
        if self.image_goal is not None:
            # The factory home leaves the fingers facing the shared centre.
            # Derive an outward parking displacement from the live collision
            # model, so larger grippers/arms automatically reserve more room.
            clearance = float(self.arm_safety.minimum_clearance if self.arm_safety else .05)
            spheres = self.arm_safety.spheres(arm) if self.arm_safety else None
            link_radius = (float(np.max(spheres[:, 3])) if spheres is not None and len(spheres)
                           else .5 * clearance)
            root_x = float(getattr(self.task.robot, arm + '_entity').get_root_pose().p[0])
            direction = np.sign(root_x)
            if direction == 0:
                direction = -1. if arm == 'left' else 1.
            pose[0] += direction * (clearance + 2. * link_radius)
            pose[2] += clearance
        current = np.asarray(self.task._current_ee_pose(arm), dtype=float)
        same_position = np.linalg.norm(current[:3] - pose[:3]) <= .008
        same_orientation = (len(current) < 7 or len(pose) < 7 or
                            abs(float(np.dot(current[3:7], pose[3:7]))) >= .999)
        if not (same_position and same_orientation):
            try:
                self._motion(arm, pose.tolist(), f"{arm}_arm_home")
            except DynamicSimError:
                # After a failed partial transfer, Cartesian IK can remain on
                # an isolated branch. The robot's joint-space origin command
                # is a scene-independent empty-arm recovery state.
                if not self.task.move(self.task.back_to_origin(
                        arm_tag=self.task._arm_tag(arm))):
                    raise
                self._capture_motion_frame(f"{arm}_joint_origin_recovery")
        if self.arm_safety is not None:
            # Position convergence alone does not make an arm stationary. A
            # previous controller command can pass through the parking pose at
            # significant speed; planning the other arm against that snapshot
            # then underestimates their relative motion. Always settle, even
            # when the Cartesian early-exit above applies.
            entity = getattr(self.task.robot, arm + '_entity')
            indices = self.arm_safety.indices[arm]
            for _ in range(3):
                if np.max(np.abs(entity.get_qvel()[indices])) <= .03:
                    break
                parked = entity.get_qpos()[indices].copy()
                self._execute_planned_motion(arm, {
                    'position': np.repeat(parked[None, :], 100, axis=0),
                    'velocity': np.zeros((100, len(parked)))}, f'{arm}_park_settle')
            if np.max(np.abs(entity.get_qvel()[indices])) > .03:
                raise DynamicSimError(f'{arm} arm did not settle before workspace handoff')

    def park_arms(self):
        """Return both arms through checked trajectories after releasing objects."""
        if self.held is not None:
            raise DynamicSimError("Cannot park arms while holding an object")
        for arm in ("left", "right"):
            self._park_arm(arm)
        if self.arm_safety is not None:
            self.arm_safety.save()

    def inject_event(self, event: dict[str, Any]):
        """Move one actor between high-level steps to test replanning."""
        if event.get("type") != "move_object":
            raise DynamicSimError(f"Unsupported dynamic event: {event.get('type')!r}")
        name = self._name(str(event["object_id"]))
        if self.held is not None and self.held[0] == name:
            raise DynamicSimError("Cannot perturb an object while it is held")
        delta = np.asarray(event.get("delta_xy"), dtype=float)
        if delta.shape != (2,) or not np.isfinite(delta).all():
            raise DynamicSimError("move_object requires finite delta_xy with two values")
        actor = self.task.objects[name]
        pose = actor.get_pose()
        # SAPIEN's wrapper stores the underlying actor in .actor; test doubles
        # may expose set_pose directly.
        from copy import deepcopy
        moved = deepcopy(pose)
        moved.p[:2] += delta
        setter = getattr(actor, "actor", actor)
        setter.set_pose(moved)
        self.task.scene.step()
        return {"type": "move_object", "object_id": event["object_id"],
                "delta_xy": delta.tolist(), "new_pose": moved.p.tolist()}

    def close(self):
        if self.arm_safety is not None:
            self.arm_safety.save()
        self.task.close_env(clear_cache=True)


@dataclass
class LoopResult:
    status: str
    steps: int
    observed_goal: bool
    reasoning_goal: bool
    trace_path: Path


from script.episode_evaluation import evaluated_closed_loop


@evaluated_closed_loop
def run_closed_loop(env: TableSimAdapter, planner, instruction: str, *, max_steps=20,
                    goal_generator=None, mapping=None, initial_graph=None,
                    target_graph=None, position_tolerance=.02, events=None,
                    cached_reasoning_graph=None, online_rearrangement=False,
                    graph_validation=False):
    """Evidence-conditioned execution with an optional live geometry overlay.

    The overlay derives actions from the current scene and retains the remote
    evidence ledger and observation checkpoints used by graph-driven runs.
    """
    from high_level_planner.reasoning_graph.target_conditioned import (
        TargetGraphEvaluator,
        generate_between_scenes as generate_target_graph,
    )
    from high_level_planner.scene_graph.graph import SceneGraph

    metrics = getattr(env, "_episode_metrics", None)
    def generate_between_scenes(*args, **kwargs):
        if metrics is None:
            return generate_target_graph(*args, **kwargs)
        with metrics.phase("planning"):
            return generate_target_graph(*args, **kwargs)

    if initial_graph is not None or cached_reasoning_graph is not None:
        raise DynamicSimError("Live camera runs cannot use cached scene or reasoning graphs")
    if target_graph is not None and not (online_rearrangement or graph_validation):
        raise DynamicSimError(
            "A supplied target layout graph requires online rearrangement or graph validation")
    if graph_validation and online_rearrangement:
        raise DynamicSimError("Graph validation executes external Reasoning Graph edges without an online overlay")
    first = env.observe()
    provenance = {"source": "simulator.head_camera.rgb",
                  "image": str(first["image"].resolve()),
                  "sha256": hashlib.sha256(first["image"].read_bytes()).hexdigest(),
                  "cached_graphs_used": False,
                  "pipeline": ["head_camera", "SAM3_initial_graph", "Kimi_goal_image",
                               "layout_target_graph", "reasoning_graph", "execution"]}
    (env.output_dir / "input_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if target_graph is None:
        if online_rearrangement:
            from high_level_planner.goal_generator import GoalGenerationError
            from script.online_rearrangement import refine_visual_rotations
            initial = planner.scene_constructor.construct(first["image"], instruction=instruction)
            prediction = goal_generator.generate(first["image"], initial_scene_graph=initial,
                                                  instruction=instruction)
            if not prediction.ready_for_planning:
                raise GoalGenerationError(f"Goal identity needs review: {prediction.bundle_path}")
            target = prediction.scene_graph
            # A target-image detector may miss an otherwise unchanged item.
            # Preserve that initial visual node instead of interpreting the
            # miss as permission for the item to disappear. This is based on
            # the two camera graphs only and never reads simulator poses.
            from copy import deepcopy
            target_ids = {obj.id for obj in target.objects}
            # ``mapping`` is optional in normal live runs.  Recover the same
            # semantic identity relation from detector labels and the known
            # movable inventory; unlike complete_object_mapping this does not
            # inspect simulator poses or fit against world truth.
            visual_mapping = match_objects(initial, env.task)
            visual_mapping.update(mapping or {})
            retained = []
            for obj in initial.objects:
                if obj.id in target_ids or obj.id not in visual_mapping:
                    continue
                copy_obj = deepcopy(obj)
                copy_obj.metadata = dict(copy_obj.metadata,
                    target_inference="retained_from_initial_after_target_visual_miss")
                target.add_object(copy_obj)
                retained.append(obj.id)
            if retained:
                target.metadata["retained_after_target_visual_miss"] = retained
            refine_visual_rotations(initial, target, first["image"], prediction.image.image_path,
                                    planner.scene_constructor.perception_backend,
                                    audit_path=env.output_dir / "visual_rotation_audit.json")
            graph = generate_between_scenes(initial, target, instruction,
                                            generator=planner.reasoning_generator,
                                            position_tolerance=position_tolerance)
            graph.metadata["goal_bundle_path"] = str(prediction.bundle_path)
            search = planner.graph_planner.plan(graph)
            prediction.record_plan(goal_reached=search.goal_state is not None)
            planner.memory.update(initial, graph, preserve_execution=False)
        else:
            output = planner.plan_initial_image(first["image"], instruction,
                                                goal_generator=goal_generator,
                                                graph_validation=graph_validation)
            initial = output.scene_graph
            target = SceneGraph.from_dict(output.reasoning_graph.metadata["target_terminal_scene_graph"])
            graph, search = output.reasoning_graph, output.search_result
    else:
        initial = planner.scene_constructor.construct(first["image"], instruction=instruction)
        if graph_validation:
            initial.metadata["graph_validation_ignore_physical_feasibility"] = True
        target = target_graph.copy()
        graph = generate_between_scenes(initial, target, instruction,
                                        generator=planner.reasoning_generator,
                                        position_tolerance=position_tolerance)
        search = planner.graph_planner.plan(graph)
        planner.memory.update(initial, graph, preserve_execution=False)
        planner.evidence_store.update_from_scene(
            initial,
            source="initial_camera_observation",
            coverage={"kind": "camera_frame", "image": str(first["image"])},
        )
        planner.evidence_store.attach_execution_context(initial)
        provenance["pipeline"] = ["head_camera", "SAM3_initial_graph",
                                  "validated_target_layout_graph", "reasoning_graph", "execution"]
        provenance["validated_target_layout_graph"] = target.metadata
        (env.output_dir / "input_provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    env.bind_graph(initial, mapping)
    metrics = getattr(env, "_episode_metrics", None)
    if metrics is not None:
        metrics.configure(initial, target, position_tolerance)
        metrics.observe(initial, payload_released=env.held is None)
        metrics.phase_seconds["initialization"] = metrics.clock()-metrics.started
    if online_rearrangement:
        from script.online_rearrangement import OnlineRearrangementPlanner
        online_planner = OnlineRearrangementPlanner(
            env, target, position_tolerance=position_tolerance)
        env.image_goal = online_planner.goal
    else:
        online_planner = None
    evaluator = TargetGraphEvaluator(initial, target, position_tolerance)
    current = initial
    previous_observation_image = first["image"]
    records = []
    trace_path = env.output_dir / "dynamic_trace.json"
    scheduled = {}
    for event in events or []:
        after_step = event.get("after_step")
        if not isinstance(after_step, int) or isinstance(after_step, bool) or after_step < 1:
            raise ValueError("Every dynamic event needs after_step >= 1")
        scheduled.setdefault(after_step, []).append(event)
    initial.to_json(env.output_dir / "scene_graph_000.json")
    target.to_json(env.output_dir / "target_scene_graph.json")
    graph.to_json(env.output_dir / "reasoning_graph_000.json")
    if getattr(env, 'symbolic_grasp_validation', False):
        from script.symbolic_feedback import run_feedback_loop
        revision = None
        if getattr(env, 'visual_only_execution', False) and goal_generator is not None:
            bundle_path = graph.metadata.get('goal_bundle_path')
            if bundle_path:
                from script.adaptive_goal import AdaptiveGoalRevision
                bundle = json.loads(Path(bundle_path).read_text())
                revision = AdaptiveGoalRevision(goal_generator, initial, target,
                    bundle['image_path'], env.output_dir)
        return run_feedback_loop(env, planner, initial, target, graph, search, instruction, max_steps,
                                 goal_revision=revision)
    if getattr(env, 'object_motion_validation', False):
        from script.object_motion_validation import run_video_path
        return run_video_path(env, planner, initial, target, graph, search, instruction)
    execution_graph = {
        "schema": "tablesim.observed_execution_reasoning_graph.v1",
        "source_reasoning_graph": "reasoning_graph_000.json",
        "source_model": graph.metadata.get("model_execution", {}).get("model"),
        "source_model_api_called": bool(graph.metadata.get("model_execution", {}).get("api_called")),
        "overlay": "online_geometry_rearrangement" if online_rearrangement else None,
        "nodes": [{"id": "O0", "image": str(first["image"]), "goal": False,
                   "object_poses": first.get("object_poses", {})}],
        "edges": [], "goal_ids": [],
    }
    (env.output_dir / "execution_reasoning_graph.json").write_text(
        json.dumps(execution_graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def checkpoint(status, observed_goal, reasoning_goal):
        if metrics is not None and metrics.last_observation_valid:
            metrics.observe(current, payload_released=env.held is None)
        image_goal = env.image_goal.evaluate(env) if env.image_goal is not None else None
        evidence_path = env.output_dir / "evidence_store.json"
        evidence_path.write_text(
            json.dumps(planner.evidence_store.to_dict(), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        trace_path.write_text(json.dumps({"status": status, "instruction": instruction,
            "input_provenance": provenance,
            "evidence_store": str(evidence_path),
            "repair_history": list(getattr(planner, "_closed_loop_decisions", [])),
            "fixed_scene_task_success": bool(env.evaluate()),
            "execution_model": ("high_level_graph_validation_follow"
                                if graph_validation else
                                "kinematic_follow_not_contact_verified"),
            "observed_goal": observed_goal, "reasoning_goal": reasoning_goal,
            "online_rearrangement": bool(online_rearrangement),
            "graph_validation": bool(graph_validation),
            "image_goal": image_goal,
            "steps": records, "target_graph": target.to_dict()},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (env.output_dir / "execution_reasoning_graph.json").write_text(
            json.dumps(execution_graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return LoopResult(status, len(records), observed_goal, reasoning_goal, trace_path)

    def image_terminal():
        if env.image_goal is None or env.held is not None:
            return False
        if not (env.task.is_left_gripper_open() and env.task.is_right_gripper_open()):
            return False
        return bool(env.image_goal.evaluate(env)["passed"])

    def validation_next_edge(node_id):
        """Follow the deepest generated branch when the graph has no goal."""
        visited_execution = set(planner.memory.traversed_node_ids) | {node_id}

        def depth(current_id, path):
            if current_id in path:
                return -1
            if graph.is_goal(current_id):
                return 10_000
            next_path = path | {current_id}
            values = [
                depth(edge.to_state, next_path)
                for edge in graph.neighbors(current_id)
                if edge.to_state not in next_path
            ]
            return 0 if not values else 1 + max(values)

        outgoing = [
            edge for edge in graph.neighbors(node_id)
            if edge.to_state not in visited_execution
        ]
        if not outgoing:
            return None
        return min(outgoing, key=lambda edge: (
            -depth(edge.to_state, {node_id}),
            str(edge.metadata.get("edge_id", "")),
            edge.to_state))

    # The external graph remains an independent model proposal. Online
    # execution derives additional edges from live dependency and occupancy
    # geometry; it never reads a recorded trace or a scene-specific sequence.
    overlay_active = bool(online_rearrangement)

    for index in range(max_steps + 1):
        report = evaluator.evaluate(current, instruction)
        physical_goal = bool(env.evaluate())
        # Compare the observed graph with the generated target. The fixed
        # scene task predicate is a separate diagnostic, not this goal.
        observed_goal = image_terminal() if online_rearrangement else bool(report.terminal)
        reasoning_goal = observed_goal if overlay_active else graph.is_goal(
            planner.memory.current_node_id or graph.start_id)
        if observed_goal and reasoning_goal:
            if overlay_active:
                execution_graph["goal_ids"] = [execution_graph["nodes"][-1]["id"]]
            return checkpoint("certified_image_terminal" if online_rearrangement
                              else "certified_terminal", True, True)
        if index == max_steps:
            return checkpoint("step_limit", observed_goal, reasoning_goal)
        edge = (None if overlay_active else
                validation_next_edge(planner.memory.current_node_id or graph.start_id)
                if graph_validation and search.next_subtask is None else
                search.next_subtask)
        if edge is None and graph_validation:
            return checkpoint("graph_validation_leaf_without_terminal"
                              if not reasoning_goal else "certified_terminal",
                              observed_goal, reasoning_goal)
        if edge is None and not online_rearrangement:
            return checkpoint("no_executable_path", observed_goal, reasoning_goal)
        if edge is None:
            overlay_active = True
            from script.compare_scene003_modalities import request_from_action
            action, decision = online_planner.next_action()
            request = request_from_action(action)
            from_id, to_id = f"O{len(records)}", f"O{len(records)+1}"
            record = {"index": index, "reasoning_from": from_id,
                      "reasoning_to": to_id, "request": request,
                      "predicted_path_exists": False,
                      "reasoning_edge_source": "online_geometry_rearrangement",
                      "executed_action": action, "online_decision": decision,
                      "observed_goal_before": observed_goal}
        else:
            request = planner.vla_interface.format_subtask(edge)
            record = {"index": index, "reasoning_from": edge.from_state,
                      "reasoning_to": edge.to_state, "request": request.to_dict(),
                      "predicted_path_exists": search.goal_state is not None,
                      "reasoning_edge_source": "external_model_reasoning_graph",
                      "observed_goal_before": observed_goal}
        execution_event = None
        try:
            if getattr(env, 'execution_control_dir', None) is not None:
                from script.hybrid_execution import supervise_action
                env.execution_policy = supervise_action(
                    env.execution_control_dir, index, record['request'],
                    getattr(env, 'last_observation_image', env.output_dir / 'observation_000.png'))
                record['local_execution_decision'] = dict(env.execution_policy)
            observation = env.step(request)
            if edge is None:
                online_planner.record_success(record["executed_action"])
            if edge is not None:
                execution_event = planner.record_execution_result(
                    edge,
                    success=True,
                    details={
                        "observation": str(observation["image"]),
                        "evidence_source": "camera_rgb_and_execution_status",
                    },
                )
                if not graph_validation:
                    planner.mark_subtask_completed(edge)
            else:
                from high_level_planner.reasoning_graph import Subtask

                action = record["executed_action"]
                overlay_edge = Subtask(
                    from_state=record["reasoning_from"],
                    to_state=record["reasoning_to"],
                    action=action.get("reason") or "Execute audited Scene 003 action",
                    action_type=action["action_type"],
                    object_target={
                        "source": action.get("source_object_id"),
                        "target": action.get("target"),
                    },
                    parameters={
                        "source_object_id": action.get("source_object_id"),
                        "pose": action.get("pose", {}),
                    },
                    metadata={
                        "edge_id": f"scene003:{record['reasoning_from']}->{record['reasoning_to']}",
                        "source": record["reasoning_edge_source"],
                    },
                )
                execution_event = planner.evidence_store.record_execution(
                    overlay_edge,
                    success=True,
                    details={
                        "observation": str(observation["image"]),
                        "evidence_source": "camera_rgb_and_execution_status",
                    },
                )
            record["events"] = [env.inject_event(event) for event in scheduled.get(index + 1, [])]
            if record["events"]:
                observation = env.observe()
            if graph_validation:
                current = planner.scene_updater.update_scene_graph(
                    current, observation["image"], instruction=instruction)
                from high_level_planner.reasoning_graph.node_localization import (
                    NodeLocalization, locate_reasoning_node,
                    locate_transition_from_image_change,
                )
                localization = locate_reasoning_node(
                    current, graph,
                    candidate_ids={edge.from_state, edge.to_state})
                expected_ids = {o.id for o in graph.get_node(edge.to_state).scene_graph.objects}
                observed_ids = {o.id for o in current.objects}
                identity_consistent = expected_ids == observed_ids
                if not localization.conclusive and identity_consistent:
                    changed_ids = locate_transition_from_image_change(
                        previous_observation_image, observation["image"], graph,
                        edge.from_state, edge.to_state)
                    # Pixel changes alone cannot certify object identity or
                    # placement. Keep them as diagnostics until detection
                    # supplies the corresponding identities in this frame.
                    fresh_ids = {o.id for o in current.objects
                                 if o.metadata.get('observed_in_current_frame', True)}
                    if changed_ids and set(changed_ids) <= fresh_ids and localization.margin > .01:
                        localization = NodeLocalization(
                            edge.to_state, 0.0, float("inf"),
                            localization.ranking, True, changed_ids,
                            localization.stale_object_ids,
                            "temporal_camera_change")
                previous_node = planner.memory.current_node_id or graph.start_id
                if previous_node not in planner.memory.traversed_node_ids:
                    planner.memory.traversed_node_ids.append(previous_node)
                planner.memory.current_node_id = localization.node_id
                planner.memory.scene_graph = current
                search = planner.graph_planner.plan(
                    graph, start_id=localization.node_id)
                record["node_localization"] = {
                    "selected_node_id": localization.node_id,
                    "expected_node_id": edge.to_state,
                    "matches_expected": localization.node_id == edge.to_state,
                    "conclusive": localization.conclusive,
                    "evidence_object_ids": list(localization.evidence_object_ids),
                    "stale_object_ids": list(localization.stale_object_ids),
                    "evidence_kind": localization.evidence_kind,
                    "score": localization.score,
                    "margin": localization.margin,
                    "ranking": [[node_id, score]
                                for node_id, score in localization.ranking],
                    "source": "simulator.head_camera.rgb",
                    "image": str(observation["image"]),
                }
                with (env.output_dir / "node_localization.jsonl").open(
                        "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record["node_localization"],
                                            ensure_ascii=False) + "\n")
                repair_record = {
                    "mode": "visual_reasoning_node_localization",
                    "current_node_id": localization.node_id,
                    "expected_node_id": edge.to_state,
                }
            elif overlay_active:
                current = planner.scene_updater.update_scene_graph(
                    current, observation["image"], instruction=instruction
                )
                evidence_update = planner.evidence_store.update_from_scene(
                    current,
                    source="simulator.head_camera.rgb",
                    coverage={"kind": "camera_frame", "image": str(observation["image"])},
                )
                planner.evidence_store.attach_execution_context(current)
                repair_record = {
                    "mode": "online_geometry_rearrangement",
                    "evidence_update": evidence_update.to_dict(),
                    "evidence_version": planner.evidence_store.version,
                }
            else:
                updated = planner.update_target_conditioned_image(
                    observation["image"],
                    instruction,
                    observation_source="simulator.head_camera.rgb",
                )
                current, graph, search = (
                    updated.scene_graph,
                    updated.reasoning_graph,
                    updated.search_result,
                )
                repair_record = graph.metadata.get("closed_loop", {}).get("latest")
            current.to_json(env.output_dir / f"scene_graph_{index+1:03d}.json")
            graph.to_json(env.output_dir / f"reasoning_graph_{index+1:03d}.json")
            record["observation"] = str(observation["image"])
            record["execution_evidence"] = execution_event
            record["repair"] = repair_record
            record["object_poses"] = observation.get("object_poses", {})
            record["physical_task_success"] = env.evaluate()
            record["goal_checks_after"] = evaluator.evaluate(current, instruction).goal_checks
            record["status"] = "executed"
            records.append(record)
            previous_observation_image = observation["image"]
            if overlay_active or graph_validation:
                execution_graph["edges"].append({
                    "from_state": record["reasoning_from"], "to_state": record["reasoning_to"],
                    "action": (record.get("executed_action")
                               if overlay_active else record["request"]),
                    "source": record["reasoning_edge_source"],
                    "online_decision": record.get("online_decision"),
                    "node_localization": record.get("node_localization"),
                })
                execution_graph["nodes"].append({
                    "id": record["reasoning_to"], "image": str(observation["image"]),
                    "goal": (image_terminal() if overlay_active else
                             graph.is_goal(record["reasoning_to"])),
                    "object_poses": observation.get("object_poses", {}),
                })
            checkpoint("running", bool(evaluator.evaluate(current, instruction).terminal),
                       image_terminal() if overlay_active else
                       graph.is_goal(planner.memory.current_node_id or graph.start_id))
            if graph_validation and not localization.conclusive:
                return checkpoint("graph_validation_visual_evidence_missing",
                                  False, False)
            if graph_validation and localization.node_id != edge.to_state:
                return checkpoint("graph_validation_transition_not_observed",
                                  False, False)
        except Exception as exc:
            from script.dual_arm_safety import ArmClearanceError

            record["error"] = f"{type(exc).__name__}: {exc}"
            fatal = isinstance(exc, ArmClearanceError)
            try:
                if execution_event is None:
                    if edge is not None:
                        execution_event = planner.record_execution_result(
                            edge,
                            success=False,
                            details={"error_type": type(exc).__name__},
                            fatal=fatal,
                        )
                    else:
                        from high_level_planner.reasoning_graph import Subtask

                        action = record["executed_action"]
                        overlay_edge = Subtask(
                            from_state=record["reasoning_from"],
                            to_state=record["reasoning_to"],
                            action=action.get("reason") or "Execute audited Scene 003 action",
                            action_type=action["action_type"],
                            object_target={
                                "source": action.get("source_object_id"),
                                "target": action.get("target"),
                            },
                            parameters={
                                "source_object_id": action.get("source_object_id"),
                                "pose": action.get("pose", {}),
                            },
                            metadata={
                                "edge_id": f"scene003:{record['reasoning_from']}->{record['reasoning_to']}",
                                "source": record["reasoning_edge_source"],
                            },
                        )
                        execution_event = planner.evidence_store.record_execution(
                            overlay_edge,
                            success=False,
                            details={"error_type": type(exc).__name__},
                            fatal=fatal,
                        )
                    record["execution_evidence"] = execution_event
            except Exception as evidence_exc:
                record["evidence_recording_error"] = (
                    f"{type(evidence_exc).__name__}: {evidence_exc}"
                )
            record["status"] = "failed"
            records.append(record)
            if graph_validation:
                record["held_object"] = env.held[0] if env.held else None
                try:
                    observation = env.observe()
                    record["failure_observation"] = str(observation["image"])
                except Exception as observe_exc:
                    record["failure_observation_error"] = str(observe_exc)
                return checkpoint("graph_validation_execution_failed", False, False)
            if overlay_active:
                online_planner.record_failure(record.get("executed_action") or {}, exc)
                record["online_rearrangement_after_failure"] = online_planner.diagnostics()
                failure_image = None
                failure_poses = {}
                try:
                    observation = env.observe()
                    failure_image = observation["image"]
                    failure_poses = observation.get("object_poses", {})
                    record["failure_observation"] = str(observation["image"])
                    record["failure_object_poses"] = failure_poses
                    record["held_object"] = env.held[0] if env.held else None
                    current = planner.scene_updater.update_scene_graph(
                        current, observation["image"], instruction=instruction)
                    evidence_update = planner.evidence_store.update_from_scene(
                        current,
                        source="simulator.head_camera.rgb.after_failure",
                        coverage={"kind": "camera_frame", "image": str(observation["image"])},
                    )
                    planner.evidence_store.attach_execution_context(current)
                    record["repair"] = {
                        "mode": "online_geometry_rearrangement_retry",
                        "evidence_update": evidence_update.to_dict(),
                        "evidence_version": planner.evidence_store.version,
                    }
                    current.to_json(env.output_dir / f"scene_graph_{index+1:03d}.json")
                except Exception as observe_exc:
                    record["failure_observation_error"] = f"{type(observe_exc).__name__}: {observe_exc}"
                if env.task.plan_success is False:
                    env.task.plan_success = True
                    record["motion_failure_latch_reset_for_retry"] = True
                failed_to = record["reasoning_to"]
                execution_graph["edges"].append({
                    "from_state": record["reasoning_from"], "to_state": failed_to,
                    "action": record.get("executed_action"), "source": record["reasoning_edge_source"],
                    "status": "failed", "error": record["error"],
                })
                execution_graph["nodes"].append({
                    "id": failed_to, "image": str(failure_image) if failure_image else None,
                    "goal": False,
                    "partial_transition": True, "held_object": record.get("held_object"),
                    "object_poses": failure_poses,
                })
                checkpoint("recovering_after_overlay_failure", False, False)
                if fatal:
                    return checkpoint("safety_stop", False, False)
                if index + 1 < max_steps:
                    continue
                return checkpoint("step_limit", False, False)
            if fatal:
                return checkpoint("safety_stop", False, False)
            try:
                recovery_observation = env.observe()
                recovered = planner.update_target_conditioned_image(
                    recovery_observation["image"],
                    instruction,
                    observation_source="simulator.head_camera.rgb.after_failure",
                )
                current, graph, search = (
                    recovered.scene_graph,
                    recovered.reasoning_graph,
                    recovered.search_result,
                )
                record["failure_observation"] = str(recovery_observation["image"])
                record["repair"] = graph.metadata.get("closed_loop", {}).get("latest")
                if search.next_subtask is None:
                    return checkpoint("no_executable_path", False, False)
                continue
            except Exception as recovery_exc:
                record["recovery_error"] = f"{type(recovery_exc).__name__}: {recovery_exc}"
                return checkpoint("execution_or_replan_error", False, False)
    raise AssertionError("unreachable")


def build_environment(scene: str, task_config: str, output_dir: Path):
    """Instantiate an existing task without invoking its scripted play_once."""
    import yaml
    from script import collect_data

    if scene == "organize_table_scene003":
        raise DynamicSimError("organize_table_scene003.setup_demo loads a fixed action chain; use scene_003")
    cls = type(collect_data.class_decorator(scene))
    if not hasattr(cls, "_attach_actor") or not hasattr(cls, "_top_down_pose"):
        raise DynamicSimError(f"Scene {scene!r} has no kinematic-follow task interface")
    config = ROOT / "task_config" / f"{task_config}.yml"
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    settings["task_name"] = scene
    embodiment = settings["embodiment"]
    if len(embodiment) != 1:
        raise DynamicSimError("Dynamic runner currently supports one dual-arm embodiment")
    robot_dir = ROOT / "assets" / "embodiments" / embodiment[0]
    robot_config = yaml.safe_load((robot_dir / "config.yml").read_text(encoding="utf-8"))
    settings.update(left_robot_file=str(robot_dir), right_robot_file=str(robot_dir),
                    left_embodiment_config=robot_config, right_embodiment_config=robot_config,
                    dual_arm_embodied=True, embodiment_name=embodiment[0], task_config=task_config,
                    save_path=str(output_dir))
    return TableSimAdapter(cls(), settings, output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="scene_003")
    parser.add_argument("--task-config", default="scene_003")
    parser.add_argument("--instruction")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--position-tolerance", type=float, default=.02)
    parser.add_argument("--sam3-endpoint", default="http://127.0.0.1:8003")
    parser.add_argument("--api-config", type=Path,
                        help="API settings file; defaults to config/goal_generator.local.json in the repository")
    parser.add_argument("--goal-image", type=Path,
                        help="Use an assistant-generated terminal image; keep layout extraction and reasoning unchanged")
    parser.add_argument("--initial-scene-graph", type=Path)
    parser.add_argument("--target-scene-graph", type=Path)
    parser.add_argument("--cached-reasoning-graph", type=Path,
                        help="Reuse an authenticated graph path and skip repeated model replanning")
    parser.add_argument("--object-map", type=Path, help="JSON map of planner IDs to scene object names")
    parser.add_argument("--events", type=Path, help="JSON list of move_object events keyed by after_step")
    parser.add_argument("--validated-target-layout-graph", type=Path,
                        help="Reuse a previously audited target layout graph while rebuilding the live initial graph and reasoning graph")
    parser.add_argument("--online-rearrangement", action="store_true",
                        help="Generate fresh target/model graphs, then add live geometry-derived rearrangement edges")
    parser.add_argument("--graph-validation", action="store_true",
                        help="Execute external Reasoning Graph edges with assisted grasp/follow/release and visually localize every node")
    parser.add_argument('--hybrid-execution', action='store_true',
                        help='Try contact grasping when feasible, then use continuous assisted follow on failure')
    parser.add_argument('--object-motion-validation', action='store_true',
                        help='Execute the entire fresh graph path by continuous object motion, bypassing robot IK; audit visual verification separately')
    parser.add_argument('--symbolic-grasp-validation', action='store_true',
                        help='Execute every graph action with real robot joint drives and symbolic closed-gripper attachment')
    parser.add_argument('--visual-only-execution', action='store_true',
                        help='Ground actions only in camera RGB-D; no simulator object state')
    parser.add_argument('--execution-control-dir', type=Path,
                        help='Wait for local supervisor decisions before executing each generated action')
    args = parser.parse_args(argv)
    if args.visual_only_execution and (not args.symbolic_grasp_validation or args.object_map or args.events):
        parser.error('--visual-only-execution requires --symbolic-grasp-validation and forbids aliases/events')
    if args.symbolic_grasp_validation and (not args.graph_validation or args.hybrid_execution or args.object_motion_validation):
        parser.error('--symbolic-grasp-validation requires --graph-validation and excludes other assisted modes')
    if args.object_motion_validation and (not args.graph_validation or args.hybrid_execution):
        parser.error('--object-motion-validation requires --graph-validation and excludes --hybrid-execution')
    if args.hybrid_execution and not args.graph_validation:
        parser.error('--hybrid-execution requires --graph-validation')
    if args.goal_image and not args.goal_image.is_file():
        parser.error("--goal-image must point to an existing image")
    if args.initial_scene_graph or args.target_scene_graph or args.cached_reasoning_graph:
        parser.error("Dynamic runs require live head_camera input; cached graphs are not permitted")
    if args.validated_target_layout_graph and not args.validated_target_layout_graph.is_file():
        parser.error("--validated-target-layout-graph must point to an existing Scene Graph JSON")
    if args.online_rearrangement and args.validated_target_layout_graph:
        parser.error("Online rearrangement must derive a fresh target graph; prior target layouts are not accepted")
    if args.online_rearrangement and args.graph_validation:
        parser.error("--graph-validation cannot be combined with --online-rearrangement")
    if args.max_steps < 0 or not np.isfinite(args.position_tolerance) or args.position_tolerance < 0:
        parser.error("max-steps and position-tolerance must be nonnegative")
    if bool(args.initial_scene_graph) != bool(args.target_scene_graph):
        parser.error("Cached initial and target Scene Graphs must be supplied together")
    if args.cached_reasoning_graph and not (args.initial_scene_graph and args.target_scene_graph):
        parser.error("Cached reasoning graph requires --initial-scene-graph and --target-scene-graph")
    from high_level_planner.goal_generator import GoalGenerator
    from high_level_planner.planner.planner import HighLevelPlanner
    from high_level_planner.reasoning_graph.generator import ReasoningGraphGenerator
    from high_level_planner.reasoning_graph.language_models.openai_compatible import OpenAICompatibleLanguageModel
    from high_level_planner.scene_graph.constructor import SceneGraphConstructor
    from high_level_planner.perception.sam3_client import Sam3PerceptionClient
    from high_level_planner.scene_image_generator import load_goal_image_config
    from high_level_planner.scene_graph.graph import SceneGraph
    from script.goal_config import configured_tokens, load_api_settings, dynamic_reasoning_config

    settings = load_api_settings(args.api_config)
    overrides = {field: settings["image"][key]
                 for key, field in (("url", "base_url"), ("model", "model"),
                                    ("max_retries", "max_retries"))
                 if key in settings.get("image", {})}
    image_config = load_goal_image_config(**overrides)
    model_config = dynamic_reasoning_config(settings, args.output_dir / "model_audit")
    env = build_environment(args.scene, args.task_config, args.output_dir)
    if args.visual_only_execution:
        from script.rgbd_execution import RGBDAdapter
        env = RGBDAdapter(env.task, env.settings, env.output_dir)
    if args.online_rearrangement or args.graph_validation:
        # Online graph runs are long physical executions whose intermediate
        # states cannot be reconstructed from logical observations alone.
        # Record every simulated control interval at the configured video FPS.
        from script.compare_scene003_modalities import install_recording
        env = install_recording(env)
    env.graph_validation_mode = bool(args.graph_validation)
    env.hybrid_execution = bool(args.hybrid_execution)
    env.object_motion_validation = bool(args.object_motion_validation)
    env.symbolic_grasp_validation = bool(args.symbolic_grasp_validation)
    env.execution_control_dir = args.execution_control_dir
    env.evaluation_context = {"scene": args.scene, "task_config": args.task_config,
                              "seed": args.seed, "visual_only_execution": args.visual_only_execution,
                              "symbolic_grasp_validation": args.symbolic_grasp_validation}
    class InteriorTabletopPerception(Sam3PerceptionClient):
        """Keep low-confidence small objects while rejecting border robot fragments."""
        def detect(self, image, *, instruction=None):
            from copy import deepcopy
            key = str(Path(image).resolve()) if isinstance(image, (str, Path)) else None
            cache = getattr(self, '_visual_cache', {})
            if key in cache:
                return deepcopy(cache[key])
            result = super().detect(image, instruction=instruction)
            width, height = result.image_size
            border = max(2.0, .005 * min(width, height))
            result.detections = [d for d in result.detections
                if min(d.bbox[0], d.bbox[1], width - d.bbox[2], height - d.bbox[3]) > border]
            result.metadata["interior_tabletop_filter_px"] = border
            geometry = getattr(env, 'sensor_packets', {}).get(key)
            if geometry is not None:
                result = geometry.filter(result)
            if key:
                cache[key] = deepcopy(result)
                self._visual_cache = cache
            return result

    perception = InteriorTabletopPerception(
        endpoint=args.sam3_endpoint, confidence_threshold=.12)
    env.perception = perception
    from dataclasses import replace
    from high_level_planner.scene_graph.label_review import VisualLabelReviewer
    label_model = OpenAICompatibleLanguageModel(replace(
        model_config, audit_dir=str(args.output_dir / 'label_model_audit')))
    constructor_type = SceneGraphConstructor
    constructor_options = {}
    if args.visual_only_execution:
        from script.rgbd_execution import RGBDSceneGraphConstructor
        constructor_type = RGBDSceneGraphConstructor
        constructor_options['sensor_packets'] = lambda: getattr(env, 'sensor_packets', {})
    constructor = constructor_type(
        **constructor_options, perception_backend=perception,
        label_reviewer=VisualLabelReviewer(
            label_model, audit_dir=args.output_dir / 'label_review',
            preserve_uncertain=args.visual_only_execution))
    if args.graph_validation:
        # Validation must not let one rejected model proposal empty the whole
        # frontier. Feed local validation failures back through ImproveActions
        # and leave enough depth for temporary unblocking plus final placement.
        graph_options = {"max_depth": 16, "beam_width": 1, "num_branches": 3,
                         "minimum_beam_width": 1,
                         "augment_model_proposals": True,
                         "max_states": 24, "enable_improve": True}
        if args.visual_only_execution:
            # Use a short observable execution horizon: distant predictions are
            # replaced by fresh feedback before their actions can execute.
            graph_options.update(max_depth=3, max_states=6, max_repair_expansions=3)
    elif args.online_rearrangement:
        graph_options = {"max_depth": 12, "beam_width": 1, "num_branches": 3,
                         "max_states": 24, "enable_improve": False}
    else:
        graph_options = {"enable_improve": True}
    planner = HighLevelPlanner(scene_constructor=constructor,
            reasoning_generator=ReasoningGraphGenerator(
                model=OpenAICompatibleLanguageModel(model_config),
                # Feed rejected candidates and their local feasibility errors
                # back to the external model.  Without this graph operation an
                # all-invalid frontier terminates before UCS can find a path.
                **graph_options,
            ))
    goal_generator = GoalGenerator(image_config=image_config, output_dir=args.output_dir / "goals",
                                   perception_backend=perception,
                                   sam3_endpoint=args.sam3_endpoint)
    if args.goal_image:
        # Reuse the existing image-first entry point. Only image creation is
        # replaced; SAM3, layout extraction, identity review and planning retain
        # their existing implementations.
        from functools import partial
        goal_generator.generate = partial(goal_generator.generate,
                                          layout_image=args.goal_image.resolve())
    if args.visual_only_execution:
        original_goal_generate = goal_generator.generate
        def measured_goal(initial_image, **kwargs):
            prediction = original_goal_generate(initial_image, **kwargs)
            from script.online_rearrangement import refine_visual_rotations
            refine_visual_rotations(kwargs['initial_scene_graph'], prediction.scene_graph,
                initial_image, prediction.image.image_path, perception,
                audit_path=env.output_dir/'target_rotation_audit.json')
            prediction.scene_graph.to_json(prediction.output_dir/'sensor_refined_target.json')
            return prediction
        goal_generator.generate = measured_goal
    instruction = args.instruction or (
        'Match the supplied terminal image for every visible object: placement, support relations '
        'and visually observable orientation. Preserve all objects, clear occluders before '
        'moving their supports, and revise the reasoning graph using execution feedback. '
        'The terminal image is the initial desired layout: when execution reveals blocked or '
        'unreachable destinations, regenerate a minimally adjusted terminal image from the '
        'current observed layout, preserving object identities and the organizing intent.'
        if args.visual_only_execution and args.goal_image else env.task.get_instruction())
    mapping = json.loads(args.object_map.read_text(encoding="utf-8")) if args.object_map else None
    events = json.loads(args.events.read_text(encoding="utf-8")) if args.events else None
    initial = SceneGraph.from_json(args.initial_scene_graph) if args.initial_scene_graph else None
    target = (SceneGraph.from_json(args.validated_target_layout_graph)
              if args.validated_target_layout_graph else None)
    if target is not None and args.goal_image:
        expected = target.metadata.get("derived_from_image_sha256")
        actual = hashlib.sha256(args.goal_image.read_bytes()).hexdigest()
        if expected != actual:
            parser.error("Validated target layout graph does not match --goal-image SHA-256")
    cached_graph = args.cached_reasoning_graph
    with configured_tokens(settings, image_config):
        try:
            env.reset(args.seed)
            if args.symbolic_grasp_validation:
                from script.symbolic_grasp_execution import SymbolicGraspExecutor
                if args.visual_only_execution:
                    from script.rgbd_execution import RGBDGraspExecutor
                    env.symbolic_grasp_controller = RGBDGraspExecutor(env)
                else:
                    env.symbolic_grasp_controller = SymbolicGraspExecutor(env)
                env.symbolic_grasp_controller.install()
                if args.visual_only_execution:
                    env.symbolic_grasp_controller.clear_camera_view()
            if args.object_motion_validation:
                from script.object_motion_validation import ObjectMotionExecutor
                env.object_motion_controller = ObjectMotionExecutor(env)
            if args.hybrid_execution:
                from script.hybrid_execution import HybridExecutor
                env.hybrid_controller = HybridExecutor(env)
                env.hybrid_controller.install()
            if args.graph_validation and not args.visual_only_execution:
                # High-level validation animates only the selected object. Keep
                # every other prop kinematic so unchecked arm contacts cannot
                # perturb visual evidence for unrelated graph nodes.
                for scene_object in env.task.objects.values():
                    entity = getattr(scene_object, "actor", scene_object)
                    for component in getattr(entity, "get_components", lambda: [])():
                        if hasattr(component, "set_kinematic"):
                            component.set_kinematic(True)
            if args.online_rearrangement:
                if not args.goal_image:
                    parser.error("--online-rearrangement requires --goal-image")
                env.max_placement_deviation = .024
                env.thin_placement_margin = .01
                # Clear self-occlusion before the first visual graph. The offset is
                # derived from the configured inter-arm clearance.
                env.image_goal = object()
                env.park_arms()
                env.image_goal = None
            result = run_closed_loop(env, planner, instruction, max_steps=args.max_steps,
                goal_generator=goal_generator, mapping=mapping, initial_graph=initial,
                target_graph=target, position_tolerance=args.position_tolerance, events=events,
                cached_reasoning_graph=cached_graph,
                online_rearrangement=args.online_rearrangement,
                graph_validation=args.graph_validation)
        finally:
            env.close()
    print(json.dumps({"status": result.status, "steps": result.steps,
                      "trace": str(result.trace_path)}, ensure_ascii=False))
    return 0 if result.status in {"certified_terminal", "certified_image_terminal", "symbolic_grasp_visually_verified"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
