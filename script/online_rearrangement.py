"""Online target-layout rearrangement without recorded paths or fixed buffers.

The planner derives every edge from the live simulator state and the current
target Scene Graph.  Temporary placements are sampled anew from the observed
image workspace and filtered against current occupancy and reserved goal
footprints.  Object labels never determine ordering or buffer coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np

from high_level_planner.reasoning_graph.geometry import pose, rectangle, intersects
from high_level_planner.perception.mask_codec import decode_binary_mask


def _mask_axis(detection, image_size):
    """Return an undirected visual principal-axis angle and anisotropy."""
    if detection.metadata.get("rgbd_yaw_axis_observable") is False:
        return None
    mask = decode_binary_mask(detection.mask, expected_image_size=image_size)
    if mask is None:
        return None
    yy, xx = np.nonzero(mask)
    if len(xx) < 16:
        return None
    try:
        import cv2
        _, (width, height), angle = cv2.minAreaRect(
            np.column_stack((xx, yy)).astype(np.float32))
        if min(width, height) <= 0:
            return None
        # OpenCV's angle follows the rectangle width axis. Select its longer
        # undirected axis, which remains usable for an object partly covered by
        # another target object.
        if height > width:
            angle += 90.0
        return angle % 180.0, float(max(width, height) / min(width, height))
    except ImportError:
        points = np.column_stack((xx - xx.mean(), yy - yy.mean()))
        values, vectors = np.linalg.eigh(points.T @ points / len(points))
        if values[-1] <= 0:
            return None
        vector = vectors[:, -1]
        angle = math.degrees(math.atan2(vector[1], vector[0])) % 180.0
        return angle, math.sqrt(float(values[-1] / max(values[0], 1e-9)))


def _assign_detections(objects, detections, image_size, *, target=False,
                       reference_areas=None):
    """Associate reviewed identities by visual geometry, despite raw SAM labels."""
    from scipy.optimize import linear_sum_assignment
    if not objects or not detections:
        return {}
    size = np.asarray(image_size, float)
    costs = np.full((len(objects), len(detections)), 100., dtype=float)
    for i, obj in enumerate(objects):
        p = obj.metadata.get('planning_pose', {}) if target else {}
        xy = p.get('xy', obj.attributes.xy)
        if xy is None:
            continue
        box = obj.attributes.bbox
        for j, d in enumerate(detections):
            b = np.asarray(d.bbox, float)
            center = (b[:2]+b[2:])/2/size
            cost = float(np.linalg.norm(center-xy))
            measured_box = obj.metadata.get('target_detection_bbox') if target else None
            if measured_box is not None:
                a = np.asarray(measured_box, float)
                overlap = np.prod(np.maximum(0., np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
                union = np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-overlap
                cost += 2.*(1.-overlap/max(1.,union))
            if box is not None and not target:
                a = np.asarray(box, float)
                overlap = np.prod(np.maximum(0., np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
                union = np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-overlap
                cost += .5*(1.-overlap/max(1.,union))
            # Labels are weak hints; reviewed folder may still be SAM "book".
            cost += .03*(obj.name.casefold()!=d.label.casefold())
            area = float(np.prod(b[2:]-b[:2])/np.prod(size))
            reference = (reference_areas or {}).get(obj.id)
            if target and reference and area:
                cost += .02*abs(math.log(area/reference))
            costs[i,j] = cost
    rows, columns = linear_sum_assignment(costs)
    return {objects[i].id:detections[j] for i,j in zip(rows,columns) if costs[i,j]<.5}


def refine_visual_rotations(initial_graph, target_graph, initial_image, target_image,
                            perception_backend, *, audit_path=None):
    """Measure relative rotations from fresh SAM masks in both supplied images."""
    initial_result = perception_backend.detect(initial_image, instruction="detect every tabletop object")
    target_result = perception_backend.detect(target_image, instruction="detect every tabletop object")
    initial_matches = _assign_detections(initial_graph.objects, initial_result.detections,
                                         initial_result.image_size)
    initial_areas = {}
    for obj in initial_graph.objects:
        box = obj.attributes.bbox
        initial_areas[obj.id] = (0.0 if box is None else
            abs(float(box[2]-box[0]) * float(box[3]-box[1])) / np.prod(initial_result.image_size))
    target_matches = _assign_detections(target_graph.objects, target_result.detections,
        target_result.image_size, target=True, reference_areas=initial_areas)
    records = []
    for obj in target_graph.objects:
        first, last = initial_matches.get(obj.id), target_matches.get(obj.id)
        before = _mask_axis(first, initial_result.image_size) if first else None
        after = _mask_axis(last, target_result.image_size) if last else None
        record = {"object_id": obj.id, "initial_axis": before, "target_axis": after,
                  "applied": False}
        if first and first.metadata.get('rgbd_yaw_axis_observable') is False:
            # A rotationally ambiguous camera observation cannot justify a
            # separate absolute yaw goal supplied by the image model.
            obj.attributes.yaw_degrees = None
            obj.metadata.get('planning_pose', {}).pop('yaw_degrees', None)
            obj.metadata['visual_yaw_observable'] = False
            record['reason'] = 'rgbd_shape_has_no_observable_planar_yaw'
        elif before and after and before[1] >= 1.15:
            # A nearly square *projection* does not make an originally
            # elongated object invariant under quarter turns. Keep the same
            # undirected-axis convention across both views; otherwise a book
            # drawn upright can silently acquire a horizontal goal rotation.
            period = 180.0
            delta = (after[0] - before[0] + period/2) % period - period/2
            if after[1] <= 1.02 and obj.attributes.yaw_degrees is not None:
                # At almost equal side lengths, subpixel contour changes can
                # swap minAreaRect's major/minor axes. Resolve that ambiguity
                # with the image model's or previous frame's visual estimate;
                # do not declare the object quarter-turn symmetric.
                prior = float(obj.attributes.yaw_degrees)
                alternative = (delta+90.+90.) % 180.-90.
                delta = min((delta,alternative), key=lambda angle:abs((angle-prior+90.)%180.-90.))
                record['axis_branch_visual_prior_degrees'] = prior
            record['axis_period_degrees'] = period
            planning = obj.metadata.get("planning_pose", {})
            planning["yaw_degrees"] = float(delta)
            obj.metadata["planning_pose"] = planning
            obj.attributes.yaw_degrees = float(delta)
            record.update(applied=True, visual_delta_degrees=float(delta))
        records.append(record)
    target_graph.metadata["yaw_semantics"] = "visual_delta_from_initial_reference_degrees"
    target_graph.metadata["visual_rotation_refinement"] = {
        "source": "fresh_sam_masks", "initial_image": str(initial_image),
        "target_image": str(target_image), "records": records}
    if audit_path is not None:
        import json
        from pathlib import Path
        Path(audit_path).write_text(json.dumps(target_graph.metadata["visual_rotation_refinement"],
                                              ensure_ascii=False, indent=2) + "\n")
    return target_graph


def _angle_error_degrees(actual: float, target: float) -> float:
    return abs((actual - target + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class LayoutTarget:
    object_id: str
    name: str
    image_xy: tuple[float, float]
    world_xy: tuple[float, float]
    yaw_degrees: float | None
    support_id: str | None
    support_name: str | None


class TargetLayoutGoal:
    """Evaluate an arbitrary image-derived target graph in simulator space."""

    def __init__(self, env, target_graph, *, position_tolerance=.025, yaw_tolerance=5.0):
        self.env = env
        self.position_tolerance = float(position_tolerance)
        self.yaw_tolerance = float(yaw_tolerance)
        self.goal_image = None
        self.targets: dict[str, np.ndarray] = {}
        self.layout: dict[str, LayoutTarget] = {}
        def is_static(name):
            actor = getattr(env.task.objects[name], "actor", env.task.objects[name])
            return any("RigidStatic" in type(component).__name__
                       for component in getattr(actor, "get_components", lambda: [])())

        relations = {}
        for relation in target_graph.relations:
            if relation.relation.casefold() not in {"on", "inside", "in", "attached_to"}:
                continue
            source_name = env.object_map.get(relation.source)
            target_name = env.object_map.get(relation.target)
            if source_name not in env.task.objects or target_name not in env.task.objects:
                continue
            # A fixed surface cannot be carried onto a movable object. Correct
            # this common instance-swap error from overlapping target images
            # using physical capability, without naming either object.
            if is_static(source_name) and not is_static(target_name):
                relations[relation.target] = relation.source
            else:
                relations[relation.source] = relation.target
        for obj in target_graph.objects:
            name = env.object_map.get(obj.id)
            target_pose = pose(target_graph, obj.id)
            if name is None or target_pose is None or name not in env.task.objects:
                continue
            if is_static(name):
                continue
            if not (obj.attributes.movable and obj.attributes.portable):
                continue
            image_xy = np.asarray(target_pose["xy"], dtype=float)
            if target_pose["frame"] != "image_normalized" or image_xy.shape != (2,):
                continue
            world_xy = np.r_[image_xy, 1.0] @ env.transform
            support_id = relations.get(obj.id)
            support_name = env.object_map.get(support_id) if support_id else None
            yaw = target_pose.get("yaw_degrees", obj.attributes.yaw_degrees)
            yaw = float(yaw) if yaw is not None and math.isfinite(float(yaw)) else None
            footprint = env.task.object_geometry[name].get("footprint")
            if (footprint is not None and min(footprint) > 0
                    and max(footprint) / min(footprint) < 1.15):
                yaw = None
            if yaw is not None and target_graph.metadata.get("yaw_semantics") == \
                    "visual_delta_from_initial_reference_degrees":
                # The model compares the same instance in two images. Adding
                # that delta to the live initial yaw cancels camera rotation
                # and object-local principal-axis offsets without saved poses.
                yaw = math.degrees(env._object_yaw(name)) - yaw
            item = LayoutTarget(obj.id, name, tuple(image_xy), tuple(world_xy), yaw,
                                support_id, support_name)
            self.layout[name] = item
            self.targets[name] = np.asarray(world_xy, dtype=float)

        expected = {name for name in env.task.objects if not is_static(name)}
        missing = sorted(expected - set(self.layout))
        if missing:
            raise ValueError("Target image graph does not cover movable visual inventory: "
                             + ", ".join(missing))
        missing_yaw = []
        for name, item in self.layout.items():
            footprint = env.task.object_geometry[name].get("footprint")
            if footprint is None or min(footprint) <= 0:
                continue
            if max(footprint) / min(footprint) >= 1.25 and item.yaw_degrees is None:
                missing_yaw.append(name)
        if missing_yaw:
            raise ValueError("Target image graph lacks visual rotation for anisotropic objects: "
                             + ", ".join(sorted(missing_yaw)))

    def _yaw(self, name: str) -> float:
        return math.degrees(self.env._object_yaw(name))

    def object_passed(self, name: str) -> bool:
        item = self.layout[name]
        actor = self.env.task.objects[name]
        position = np.asarray(actor.get_pose().p, dtype=float)
        destination = (np.asarray(self.env.task.objects[item.support_name].get_pose().p[:2])
                       if item.support_name else np.asarray(item.world_xy))
        position_ok = np.linalg.norm(position[:2] - destination) <= self.position_tolerance
        yaw_ok = item.yaw_degrees is None or _angle_error_degrees(
            self._yaw(name), item.yaw_degrees) <= self.yaw_tolerance
        support_ok = True
        if item.support_name:
            support = np.asarray(self.env.task.objects[item.support_name].get_pose().p, dtype=float)
            expected = support[2] + self.env.task.object_geometry[item.support_name]["height"]
            gap = float(position[2] - expected)
            support_ok = 0 <= gap <= .025
        return bool(position_ok and yaw_ok and support_ok)

    def evaluate(self, _env=None) -> dict[str, Any]:
        objects = {}
        for name, item in self.layout.items():
            position = np.asarray(self.env.task.objects[name].get_pose().p, dtype=float)
            destination = (np.asarray(self.env.task.objects[item.support_name].get_pose().p[:2])
                           if item.support_name else np.asarray(item.world_xy))
            error = float(np.linalg.norm(position[:2] - destination))
            yaw_error = (None if item.yaw_degrees is None else
                         _angle_error_degrees(self._yaw(name), item.yaw_degrees))
            objects[name] = {
                "target_xy": destination.tolist(), "actual_xy": position[:2].tolist(),
                "position_error_m": error, "yaw_error_degrees": yaw_error,
                "support": item.support_name or "table", "passed": self.object_passed(name),
            }
        return {"passed": bool(objects) and all(item["passed"] for item in objects.values()),
                "position_tolerance_m": self.position_tolerance, "objects": objects}


class OnlineRearrangementPlanner:
    """Choose one dependency-safe edge from live geometry at a time."""

    def __init__(self, env, target_graph, *, position_tolerance=.025):
        self.env = env
        self.goal = TargetLayoutGoal(env, target_graph,
                                     position_tolerance=position_tolerance)
        self.failed_destinations: set[tuple[str, str, str]] = set()
        self.failed_placement_regions: dict[tuple[str, str], list[np.ndarray]] = {}
        self.last_decision: dict[str, Any] = {}
        self.buffer_uses: dict[str, int] = {}
        self.used_buffers: set[tuple[str, str, str]] = set()
        self.successful_buffer_regions: dict[str, list[np.ndarray]] = {}
        self.failed_arms: dict[str, set[str]] = {}
        self.motion_successes: dict[str, int] = {}
        self.access_dependencies: dict[str, dict[str, np.ndarray]] = {}
        self.access_release_requirements: dict[str, dict[str, float]] = {}
        self.access_collision_regions: dict[str, dict[str, list[np.ndarray]]] = {}
        self.access_corridors: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
        self.access_failure_arms: dict[str, dict[str, set[str]]] = {}
        self.cycle_arm_overrides: dict[str, str] = {}
        self.release_holds: dict[str, set[str]] = {}
        self.pending_buffer_holds: dict[str, set[str]] = {}
        self.failure_signatures: dict[tuple[str, str, tuple[str, ...], str], int] = {}

    def _is_final_action(self, name: str, action: dict[str, Any]) -> bool:
        item = self.goal.layout.get(name)
        pose_data = action.get("pose", {})
        xy = pose_data.get("xy", ()) if isinstance(pose_data, dict) else ()
        if item is None or len(xy) != 2:
            return False
        return (action.get("target") != "table" or
                np.linalg.norm(np.asarray(xy, dtype=float)-item.image_xy) <= 1e-4)

    @staticmethod
    def _failure_stage(message: str) -> str:
        """Classify evidence by the part of the manipulation that failed."""
        destination_tokens = ("transfer", "transport", "lower", "placement",
                              "release", "target pose", "requested target",
                              "correction for held", "release correction")
        if any(token in message for token in destination_tokens):
            return "destination"
        if ("grasp" in message or "pick" in message or
                "side_high" in message or "side_approach" in message):
            return "source"
        return "motion"

    def record_failure(self, action: dict[str, Any], error: Exception | str | None = None) -> None:
        pose_data = action.get("pose", {})
        xy = pose_data.get("xy", []) if isinstance(pose_data, dict) else []
        key = ",".join(f"{float(v):.3f}" for v in xy) if len(xy) == 2 else str(action.get("target"))
        name = self.env.object_map.get(action.get("source_object_id"), action.get("source_object_id", ""))
        self.pending_buffer_holds.pop(str(name), None)
        requested_arm = pose_data.get("arm") if isinstance(pose_data, dict) else None
        arm = (self.env.held[1] if self.env.held and self.env.held[0] == name
               else requested_arm if requested_arm in {"left", "right"}
               else self.env._arm(name))
        message = str(error or "").casefold()
        stage = self._failure_stage(message)
        final_action = self._is_final_action(str(name), action)
        placement_only = "no collision-free ik placement" in message
        # DynamicSimError carries an authoritative blocker field.  An empty
        # tuple means the executor inspected the geometry and found no causal
        # movable object; it must not fall back to scraping incidental object
        # names from the human-readable candidate audit.
        has_structured_blockers = hasattr(error, "blockers")
        structured = set(getattr(error, "blockers", ()) or ())
        structured_corridors = getattr(error, "blocker_corridors", {}) or {}
        identified_blockers = set()
        for blocker, actor in self.env.task.objects.items():
            # Object names must be standalone evidence tokens.  A raw
            # substring check makes e.g. ``cup`` match ``occupied`` and can
            # move an already-correct object for a collision that never
            # happened.
            token = re.compile(r"(?<![a-z0-9_])" + re.escape(blocker.casefold())
                               + r"(?![a-z0-9_])")
            mentioned = (blocker in structured if has_structured_blockers
                         else bool(token.search(message)))
            if blocker != name and mentioned:
                entity = getattr(actor, "actor", actor)
                if any("RigidStatic" in type(component).__name__
                       for component in getattr(entity, "get_components", lambda: [])()):
                    # A fixed support can appear in an IK audit but cannot be
                    # a rearrangement edge. Let arm/grasp selection adapt.
                    continue
                identified_blockers.add(blocker)
                # A source collision proves that the blocker prevents pickup.
                # A collision on a final placement must also be resolved at the
                # fixed goal. A temporary transfer collision only disproves that
                # sampled destination/route and must not create a grasp edge.
                if stage == "source" or final_action:
                    dependencies = self.access_dependencies.setdefault(str(name), {})
                    current = np.asarray(actor.get_pose().p[:2], dtype=float).copy()
                    # Keep the first collision basin stable. Re-observing the
                    # same blocked approach after a small physics drift must
                    # not move the goalpost or inflate the required release.
                    dependencies.setdefault(blocker, current)
                    requirements = self.access_release_requirements.setdefault(str(name), {})
                    radius = requirements.setdefault(
                        blocker, self._base_access_release_distance(blocker))
                    regions = self.access_collision_regions.setdefault(
                        str(name), {}).setdefault(blocker, [dependencies[blocker].copy()])
                    # One dependency may collide at several separated blocker
                    # poses. Preserve each observed basin rather than growing
                    # one scene-specific exclusion radius without bound.
                    if all(np.linalg.norm(current-region) >= .5*radius for region in regions):
                        regions.append(current)
                    corridors = self.access_corridors.setdefault(
                        str(name), {}).setdefault(blocker, [])
                    for segment in structured_corridors.get(blocker, ()):
                        if len(segment) != 2:
                            continue
                        start, end = (np.asarray(point, dtype=float)
                                      for point in segment)
                        if start.shape != (2,) or end.shape != (2,):
                            continue
                        if all(np.linalg.norm(start-old_start) > .005
                               or np.linalg.norm(end-old_end) > .005
                               for old_start, old_end in corridors):
                            corridors.append((start.copy(), end.copy()))
                    if stage == "source":
                        self.access_failure_arms.setdefault(str(name), {}).setdefault(
                            blocker, set()).add(str(arm))
        signature = (str(name), stage, tuple(sorted(identified_blockers)), str(arm))
        self.failure_signatures[signature] = self.failure_signatures.get(signature, 0) + 1
        destination_failure = (placement_only or "during transfer" in message
                               or stage == "destination")
        # A final target remains eligible after a transient route failure, but
        # two identical failures without a movable cause disprove this arm's
        # current route family. Hand it to the other arm instead of replaying
        # the exact collision forever.
        if (final_action and destination_failure and not identified_blockers
                and self.failure_signatures[signature] >= 2):
            self.failed_arms.setdefault(str(name), set()).add(str(arm))
        # A region rejected because of a named movable obstacle becomes valid
        # again after that obstacle moves. Only obstacle-free failures are
        # evidence of an intrinsic local IK basin.
        if destination_failure and not final_action:
            self.failed_destinations.add((str(name), key, str(arm)))
        if destination_failure and not final_action and len(xy) == 2:
            world_xy = np.r_[np.asarray(xy, dtype=float), 1.0] @ self.env.transform
            regions = self.failed_placement_regions.setdefault((str(name), str(arm)), [])
            regions.append(np.asarray(world_xy, dtype=float))
            if placement_only and not identified_blockers and len(regions) >= 2:
                self.failed_arms.setdefault(str(name), set()).add(str(arm))
        # A named external blocker is an environment dependency, not evidence
        # that this arm is intrinsically incapable.  Retry it after the
        # blocker moves; reserve arm handoff for pure IK failures.
        if (not self.env.held and arm in {"left", "right"}
                and stage in {"source", "motion"} and not identified_blockers):
            self.failed_arms.setdefault(str(name), set()).add(str(arm))

    def record_success(self, action: dict[str, Any]) -> None:
        """Retire access evidence disproved by a successful source motion."""
        name = self.env.object_map.get(
            action.get("source_object_id"), action.get("source_object_id", ""))
        name = str(name)
        # Once the upstream object has moved, every temporary blocker it held
        # away may rejoin normal goal planning. A successful buffer move first
        # promotes the live geometric causes captured when it was selected.
        final_action = self._is_final_action(name, action)
        released_blockers = self.release_holds.pop(name, set())
        if not final_action:
            for source in self.pending_buffer_holds.pop(name, set()):
                self.release_holds.setdefault(source, set()).add(name)
        else:
            self.pending_buffer_holds.pop(name, None)
        self.motion_successes[name] = self.motion_successes.get(name, 0) + 1
        if not final_action:
            xy = (action.get("pose") or {}).get("xy", ())
            if len(xy) == 2:
                world_xy = np.r_[np.asarray(xy, dtype=float), 1.0] @ self.env.transform
                regions = self.successful_buffer_regions.setdefault(name, [])
                if all(np.linalg.norm(world_xy-region) > .01 for region in regions):
                    regions.append(np.asarray(world_xy, dtype=float))
        # Keep buffer nodes consumed for the lifetime of the dependency they
        # are resolving. Clearing them after every successful move lets a
        # blocker oscillate between adjacent nodes without ever releasing its
        # upstream grasp corridor. Once the upstream source succeeds (or this
        # object reaches its own final state), the nodes become reusable.
        reusable = set(released_blockers)
        if final_action:
            reusable.add(name)
        if reusable:
            self.used_buffers = {entry for entry in self.used_buffers
                                 if entry[0] not in reusable}
        self.access_dependencies.pop(name, None)
        self.access_release_requirements.pop(name, None)
        self.access_collision_regions.pop(name, None)
        self.access_corridors.pop(name, None)
        self.access_failure_arms.pop(name, None)
        self.cycle_arm_overrides.pop(name, None)
        self.failed_arms.pop(name, None)

    def diagnostics(self) -> dict[str, Any]:
        """Expose only current-run evidence needed to audit replanning."""
        return {
            "access_dependencies": {
                source: {blocker: origin.tolist() for blocker, origin in blockers.items()}
                for source, blockers in self.access_dependencies.items()
            },
            "active_access_blockers": {
                source: self._access_blockers(source)
                for source in self.access_dependencies
            },
            "access_release_requirements": {
                source: dict(requirements)
                for source, requirements in self.access_release_requirements.items()
            },
            "access_collision_regions": {source: {blocker: [region.tolist()
                for region in regions] for blocker, regions in blockers.items()}
                for source, blockers in self.access_collision_regions.items()},
            "access_corridors": {source: {blocker: [[start.tolist(), end.tolist()]
                for start, end in segments] for blocker, segments in blockers.items()}
                for source, blockers in self.access_corridors.items()},
            "failed_arms": {name: sorted(arms) for name, arms in self.failed_arms.items()},
            "access_failure_arms": {source: {blocker: sorted(arms)
                for blocker, arms in blockers.items()}
                for source, blockers in self.access_failure_arms.items()},
            "cycle_arm_overrides": dict(self.cycle_arm_overrides),
            "release_holds": {source: sorted(blockers)
                              for source, blockers in self.release_holds.items()},
            "failure_signatures": {"|".join((name, stage, ",".join(blockers), arm)): count
                for (name, stage, blockers, arm), count in self.failure_signatures.items()},
            "motion_successes": dict(self.motion_successes),
        }

    def _destination_failed(self, name: str, image_xy) -> bool:
        key = ",".join(f"{float(v):.3f}" for v in image_xy)
        arm = (self.env.held[1] if self.env.held and self.env.held[0] == name
               else self._action_arm(name))
        return (name, key, arm) in self.failed_destinations

    def _placement_region_failed(self, name: str, arm: str, world_xy) -> bool:
        """Avoid repeatedly probing the same locally unreachable IK basin."""
        failures = self.failed_placement_regions.get((name, arm), ())
        footprint = self.env.task.object_geometry[name].get("footprint")
        arm_clearance = float(getattr(getattr(self.env, "arm_safety", None),
                                      "minimum_clearance", .05))
        object_scale = 0. if footprint is None else .5 * float(max(footprint))
        radius = float(np.clip(max(1.2 * arm_clearance, object_scale), .04, .12))
        return any(np.linalg.norm(np.asarray(world_xy) - failed) <= radius
                   for failed in failures)

    def _action_arm(self, name: str) -> str:
        selected = None
        if name in self.cycle_arm_overrides:
            override = self.cycle_arm_overrides[name]
            if override not in self.failed_arms.get(name, set()):
                selected = override
            else:
                # An SCC escape arm is only a hypothesis. Once live execution
                # disproves it, return to normal arm selection instead of
                # pinning every retry to the failed arm.
                self.cycle_arm_overrides.pop(name, None)
        natural = self.env._arm(name)
        failed = self.failed_arms.get(name, set())
        if selected is None:
            alternate = "right" if natural == "left" else "left"
            selected = natural if natural not in failed else (
                alternate if alternate not in failed else natural)
        # A source-side arm attempt remains useful evidence after its named
        # blocker moves.  Otherwise the blocker immediately becomes inactive
        # and buffer generation silently selects the same natural arm again,
        # never probing the alternate IK family.  Once both arms have been
        # tried, normal selection resumes so moving live blockers can make a
        # previously rejected arm feasible.
        access_failed = (set().union(*self.access_failure_arms.get(name, {}).values())
                         if self.access_failure_arms.get(name) else set())
        alternate = "right" if selected == "left" else "left"
        if selected in access_failed and alternate not in failed | access_failed:
            return alternate
        return selected

    def _overlap(self, first: str, first_xy, second: str, second_xy, margin=.006) -> bool:
        a = self.env.task.object_geometry[first].get("footprint")
        b = self.env.task.object_geometry[second].get("footprint")
        if a is None or b is None:
            return False
        return self.env._rectangles_overlap(
            first_xy, a, self.env._object_yaw(first), second_xy, b,
            self.env._object_yaw(second), margin)

    def _above_blockers(self, name: str) -> list[str]:
        actor = self.env.task.objects[name]
        top = float(actor.get_pose().p[2] + self.env.task.object_geometry[name]["height"])
        blockers = []
        for other, other_actor in self.env.task.objects.items():
            if other == name:
                continue
            bottom = float(other_actor.get_pose().p[2])
            if bottom + .003 < top:
                continue
            if self._overlap(name, actor.get_pose().p[:2], other, other_actor.get_pose().p[:2]):
                blockers.append(other)
        return blockers

    def _target_blockers(self, name: str) -> list[str]:
        item = self.goal.layout[name]
        if item.support_name:
            return [other for other in self._above_blockers(item.support_name)
                    if other != name]
        return [other for other in self.env._placement_blockers(name, item.world_xy)
                if other in self.goal.layout and other != name]

    def _blocking_objects(self, name: str) -> list[str]:
        """Unify physical, destination and observed access dependencies."""
        return list(dict.fromkeys(self._above_blockers(name)
                                  + self._target_blockers(name)
                                  + self._access_blockers(name)))

    def _access_blockers(self, name: str) -> list[str]:
        active = []
        for blocker, origin in self.access_dependencies.get(name, {}).items():
            current = np.asarray(self.env.task.objects[blocker].get_pose().p[:2], dtype=float)
            release_distance = self._access_release_distance(name, blocker)
            regions = self.access_collision_regions.get(name, {}).get(blocker, [origin])
            blocker_size = self.env.task.object_geometry[blocker].get("footprint")
            blocker_radius = (0. if blocker_size is None else
                              .5 * float(np.linalg.norm(blocker_size)))
            arm_clearance = float(getattr(getattr(self.env, "arm_safety", None),
                                          "minimum_clearance", .05))
            corridors = self.access_corridors.get(name, {}).get(blocker, ())
            inside_corridor = any(
                self._point_segment_distance(current, start, end)
                < blocker_radius + arm_clearance
                for start, end in corridors)
            if (inside_corridor or any(
                    np.linalg.norm(current-region) < release_distance
                    for region in regions)):
                active.append(blocker)
        return active

    def _access_release_distance(self, source: str, blocker: str) -> float:
        return self.access_release_requirements.get(source, {}).get(
            blocker, self._base_access_release_distance(blocker))

    def _base_access_release_distance(self, blocker: str) -> float:
        blocker_size = self.env.task.object_geometry[blocker].get("footprint")
        blocker_radius = (0. if blocker_size is None else
                          .5 * float(np.linalg.norm(blocker_size)))
        arm_distance = float(getattr(getattr(self.env, "arm_safety", None),
                                     "minimum_clearance", 0.0))
        # The recorded origin is the obstacle's pose at contact. Moving its
        # own swept radius plus the robot clearance leaves that local collision
        # basin; adding the source radius overstates the necessary relocation.
        return max(.02, blocker_radius + arm_distance)

    def _access_origin_clear(self, blocker: str, world_xy) -> bool:
        for source, dependencies in self.access_dependencies.items():
            origin = dependencies.get(blocker)
            required = self._access_release_distance(source, blocker)
            regions = self.access_collision_regions.get(source, {}).get(
                blocker, [origin] if origin is not None else [])
            if any(np.linalg.norm(np.asarray(world_xy)-region) < required
                   for region in regions):
                return False
            size = self.env.task.object_geometry[blocker].get("footprint")
            radius = 0. if size is None else .5 * float(np.linalg.norm(size))
            clearance = float(getattr(getattr(self.env, "arm_safety", None),
                                      "minimum_clearance", .05))
            if any(self._point_segment_distance(world_xy, start, end)
                   < radius + clearance
                   for start, end in self.access_corridors.get(
                       source, {}).get(blocker, ())):
                return False
        return True

    @staticmethod
    def _point_segment_distance(point, start, end) -> float:
        point, start, end = (np.asarray(value, dtype=float) for value in
                             (point, start, end))
        delta = end-start
        denominator = float(delta @ delta)
        if denominator <= 1e-12:
            return float(np.linalg.norm(point-start))
        fraction = float(np.clip((point-start) @ delta / denominator, 0., 1.))
        return float(np.linalg.norm(point-(start+fraction*delta)))

    def _blocked_source_corridor_clearance(self, moving: str, world_xy,
                                           blocked_sources: list[str]) -> float:
        """Clearance from a buffer to the object corridors it is meant to free."""
        moving_size = self.env.task.object_geometry[moving].get("footprint")
        moving_radius = 0. if moving_size is None else .5*float(np.linalg.norm(moving_size))
        values = []
        for source in blocked_sources:
            source_size = self.env.task.object_geometry[source].get("footprint")
            source_radius = 0. if source_size is None else .5*float(np.linalg.norm(source_size))
            start = self.env.task.objects[source].get_pose().p[:2]
            end = self.goal.targets[source]
            values.append(self._point_segment_distance(world_xy, start, end)
                          - moving_radius-source_radius)
        return min(values, default=float("inf"))

    def _access_retry_arm_available(self, name: str) -> bool:
        blockers = self._access_blockers(name)
        if not blockers:
            return False
        failed = set().union(*(self.access_failure_arms.get(name, {}).get(
            blocker, set()) for blocker in blockers))
        return bool(failed) and self._action_arm(name) not in failed

    def _active_dependency_graph(self, names) -> dict[str, set[str]]:
        nodes = set(names)
        return {source: {blocker for blocker in self._blocking_objects(source)
                         if blocker in nodes}
                for source in nodes}

    @staticmethod
    def _cyclic_components(graph: dict[str, set[str]]) -> list[set[str]]:
        """Return strongly connected dependency components without a library."""
        index = 0
        indices, low, stack, on_stack, result = {}, {}, [], set(), []
        def visit(node):
            nonlocal index
            indices[node] = low[node] = index
            index += 1
            stack.append(node); on_stack.add(node)
            for neighbour in graph.get(node, ()):
                if neighbour not in indices:
                    visit(neighbour); low[node] = min(low[node], low[neighbour])
                elif neighbour in on_stack:
                    low[node] = min(low[node], indices[neighbour])
            if low[node] == indices[node]:
                component = set()
                while True:
                    member = stack.pop(); on_stack.remove(member); component.add(member)
                    if member == node:
                        break
                if len(component) > 1 or node in graph.get(node, ()):
                    result.append(component)
        for node in graph:
            if node not in indices:
                visit(node)
        return result

    def _set_cycle_escape_arm(self, name: str) -> str:
        natural = self.env._arm(name)
        failed = set(self.failed_arms.get(name, set()))
        if self.access_failure_arms.get(name):
            failed.update(set().union(*self.access_failure_arms[name].values()))
        alternate = "right" if natural == "left" else "left"
        arm = natural if natural not in failed else alternate
        self.cycle_arm_overrides[name] = arm
        return arm

    def _final_ready(self, name: str) -> bool:
        item = self.goal.layout[name]
        # If both direct arm route families have failed at the fixed target,
        # continuing to emit the same final edge cannot change the geometry.
        # Stage the object at a newly sampled free-space pose first; a
        # successful buffer motion clears this current-run evidence and lets
        # the final approach be planned from a different source topology.
        if self.failed_arms.get(name, set()) >= {"left", "right"}:
            return False
        if self._destination_failed(name, item.image_xy):
            return False
        if self._access_blockers(name) and not self._access_retry_arm_available(name):
            return False
        if self._above_blockers(name):
            return False
        if item.support_name:
            if item.support_name in self.goal.layout and not self.goal.object_passed(item.support_name):
                return False
            return not self._target_blockers(name)
        return not self._target_blockers(name) and self.env._clear_table_placement(
            name, np.asarray(item.world_xy), margin=.005)

    def _normalized(self, world_xy) -> list[float]:
        return ((np.asarray(world_xy, dtype=float) - self.env.transform[2])
                @ np.linalg.inv(self.env.transform[:2])).tolist()

    def _action(self, name: str, *, final: bool, image_xy=None, world_xy=None) -> dict[str, Any]:
        item = self.goal.layout[name]
        if final:
            image_xy = list(item.image_xy)
            target = item.support_id or "table"
            kind = "stack" if item.support_id else "move"
            support = item.support_id or "table"
            reason = "target pose is dependency-ready in the live rearrangement graph"
        else:
            image_xy = list(image_xy if image_xy is not None else self._normalized(world_xy))
            target = support = "table"
            kind = "move"
            reason = "sampled free-space node clears a live source or destination dependency"
        action = {"action_type": kind, "source_object_id": item.object_id,
                  "target": target, "pose": {"frame": "image_normalized",
                  "xy": image_xy, "support": support,
                  "arm": self._action_arm(name)}, "reason": reason}
        if final and item.yaw_degrees is not None:
            action["pose"]["yaw_degrees"] = item.yaw_degrees
        return action

    def _reserved_target_clear(self, moving: str, world_xy) -> bool:
        for other, item in self.goal.layout.items():
            if other == moving or item.support_name:
                continue
            if self._overlap(moving, world_xy, other, item.world_xy, margin=.012):
                return False
        return True

    def _buffer_neighbour_clear(self, moving: str, world_xy) -> bool:
        """Leave enough radial space for a later grasp around every object."""
        moving_size = self.env.task.object_geometry[moving].get("footprint")
        moving_radius = (0. if moving_size is None else
                         .5 * float(np.linalg.norm(moving_size)))
        arm_clearance = float(getattr(getattr(self.env, "arm_safety", None),
                                      "minimum_clearance", .05))
        for other, actor in self.env.task.objects.items():
            if other == moving:
                continue
            other_size = self.env.task.object_geometry[other].get("footprint")
            other_radius = (0. if other_size is None else
                            .5 * float(np.linalg.norm(other_size)))
            separation = float(np.linalg.norm(
                np.asarray(world_xy)-np.asarray(actor.get_pose().p[:2])))
            if separation < moving_radius + other_radius + arm_clearance:
                return False
        return True

    def _successful_buffer_blockers(self, moving: str) -> set[str]:
        """Objects currently occupying a free-space node proven in this run."""
        moving_size = self.env.task.object_geometry[moving].get("footprint")
        moving_radius = (0. if moving_size is None else
                         .5 * float(np.linalg.norm(moving_size)))
        clearance = float(getattr(getattr(self.env, "arm_safety", None),
                                  "minimum_clearance", .05))
        blockers = set()
        for region in self.successful_buffer_regions.get(moving, ()):
            for other, actor in self.env.task.objects.items():
                if other == moving:
                    continue
                size = self.env.task.object_geometry[other].get("footprint")
                radius = 0. if size is None else .5 * float(np.linalg.norm(size))
                if np.linalg.norm(np.asarray(actor.get_pose().p[:2])-region) < \
                        moving_radius + radius + clearance:
                    blockers.add(other)
        return blockers

    def _source_failure_count(self, name: str) -> int:
        return sum(count for (source, stage, _blockers, _arm), count
                   in self.failure_signatures.items()
                   if source == name and stage == "source")

    def _buffer_candidates(self, name: str):
        current = np.asarray(self.env.task.objects[name].get_pose().p[:2], dtype=float)
        arm = self._action_arm(name)
        failed_regions = self.failed_placement_regions.get((name, arm), ())
        footprint = self.env.task.object_geometry[name].get("footprint")
        centre_overlap = 0. if footprint is None else .5 * float(min(footprint))
        candidates = []
        blocked_sources = [source for source in self.access_dependencies
                           if name in self._access_blockers(source)]
        prior_buffers = self.successful_buffer_regions.get(name, ())
        arm_clearance = float(getattr(getattr(self.env, "arm_safety", None),
                                      "minimum_clearance", .05))
        buffer_novelty = max(
            arm_clearance,
            0. if footprint is None else .5 * float(min(footprint)))
        # Start with a cheap lattice, then fill its gaps and approach the table
        # boundary. Small objects can have valid placements between coarse
        # samples after several collision basins have been learned. Both
        # lattices are derived from image space and live geometry; no scene
        # coordinate is privileged.
        samples = [(ix / denominator, iy / denominator)
                   for denominator, lower, upper in ((16, 2, 15), (32, 1, 32))
                   for iy in range(lower, upper)
                   for ix in range(lower, upper)]
        seen = set()
        footprint_margin = (.006 if footprint is None else
                            float(np.clip(.15 * min(footprint), .004, .012)))
        for image_x, image_y in samples:
            sample_key = (round(image_x, 6), round(image_y, 6))
            if sample_key in seen:
                continue
            seen.add(sample_key)
            image_xy = np.array([image_x, image_y])
            world_xy = np.r_[image_xy, 1.0] @ self.env.transform
            if ((arm == "right" and world_xy[0] < -centre_overlap)
                    or (arm == "left" and world_xy[0] > centre_overlap)):
                continue
            if not self.env._clear_table_placement(
                    name, world_xy, margin=footprint_margin):
                continue
            if not self._buffer_neighbour_clear(name, world_xy):
                continue
            if not self._reserved_target_clear(name, world_xy):
                continue
            distance = float(np.linalg.norm(world_xy - current))
            if distance < .015 or not self._access_origin_clear(name, world_xy):
                continue
            if blocked_sources and any(
                    np.linalg.norm(world_xy-prior) < buffer_novelty
                    for prior in prior_buffers):
                continue
            target_distance = float(np.linalg.norm(world_xy - self.goal.targets[name]))
            if blocked_sources:
                corridor_clearance = self._blocked_source_corridor_clearance(
                    name, world_xy, blocked_sources)
                # A buffer must not recreate the same obstruction somewhere
                # along the blocked object's source-to-goal corridor.
                if corridor_clearance < .02:
                    continue
                score = distance + .15 * target_distance - .35*corridor_clearance
            else:
                score = distance + .15 * target_distance
            if failed_regions:
                # After live IK rejects a local placement basin, actively
                # diversify the next probe instead of walking across adjacent
                # lattice cells.  This uses only failures from this run and
                # scales continuously with world-space distance, so it does
                # not encode a preferred scene location.
                novelty = min(float(np.linalg.norm(world_xy-failed))
                              for failed in failed_regions)
                score -= 2.0 * novelty
            candidates.append((score, image_xy, world_xy))
        candidates.sort(key=lambda value: value[0])
        return candidates

    def _buffer_action(self, name: str) -> dict[str, Any]:
        for _, image_xy, world_xy in self._buffer_candidates(name):
            key = ",".join(f"{float(v):.3f}" for v in image_xy)
            arm = (self.env.held[1] if self.env.held and self.env.held[0] == name
                   else self._action_arm(name))
            if ((name, key, arm) not in self.failed_destinations
                    and not self._placement_region_failed(name, arm, world_xy)
                    and (name, key, arm) not in self.used_buffers):
                self.used_buffers.add((name, key, arm))
                self.buffer_uses[name] = self.buffer_uses.get(name, 0) + 1
                return self._action(name, final=False, image_xy=image_xy, world_xy=world_xy)
        raise RuntimeError(f"No dynamically sampled free-space placement for {name}")

    def next_action(self) -> tuple[dict[str, Any], dict[str, Any]]:
        unfinished = [name for name in self.goal.layout if not self.goal.object_passed(name)]
        if not unfinished:
            raise StopIteration
        held = self.env.held[0] if self.env.held else None
        if held:
            if held in unfinished and self._final_ready(held):
                action = self._action(held, final=True)
                decision = {"kind": "final", "object": held, "held_recovery": True}
            else:
                action = self._buffer_action(held)
                decision = {"kind": "buffer", "object": held, "held_recovery": True}
            self.last_decision = decision
            return action, decision

        # A collision observed on the current run can require temporarily
        # moving an object that was already at its target. It rejoins the
        # ordinary unfinished set after that buffer move and is restored later.
        authoritative_access = {
            blocker for source in unfinished for blocker in self._access_blockers(source)}

        def protected_completed_support(name):
            item = self.goal.layout.get(name)
            return bool(item is not None and item.support_name
                        and self.goal.object_passed(name)
                        and name not in authoritative_access)

        # Do not dismantle a completed stack merely because conservative
        # footprint geometry predicts overlap at another goal. A supported
        # object can be hard to regrasp by construction. It remains movable
        # when an actual current-run grasp/trajectory collision names it.
        forced_buffers = {
            blocker for name in unfinished for blocker in self._blocking_objects(name)
            if blocker in self.goal.layout and not protected_completed_support(blocker)}
        # A blocker that has moved far enough must stay out of the way until
        # the source action succeeds. Restoring it immediately recreates the
        # same access collision; record_success(source) retires this hold.
        reserved_away = {blocker for source in unfinished
                         for blocker in self.access_dependencies.get(source, {})
                         if blocker in self.goal.layout}
        reserved_away.update(blocker for source in unfinished
                             for blocker in self.release_holds.get(source, set())
                             if blocker in self.goal.layout)
        candidates = list(dict.fromkeys([*unfinished, *sorted(forced_buffers)]))
        # Build causal pressure from inaccessible sources, occupied goals and
        # access contacts learned on this run.
        pressure = {name: 0 for name in candidates}
        for name in unfinished:
            for blocker in self._blocking_objects(name):
                if blocker in pressure:
                    pressure[blocker] += 1
        # A blocker whose own final pose is executable should go directly
        # there: this both clears its current footprint and makes monotonic goal
        # progress. Only an observed access hold requires it to remain away.
        ready = [name for name in unfinished
                 if name not in reserved_away and self._final_ready(name)]
        if ready:
            # Resolve the object that releases the most other dependencies.
            ready.sort(key=lambda name: (-pressure[name],
                float(np.linalg.norm(self.env.task.objects[name].get_pose().p[:2]
                                     - self.goal.targets[name])), name))
            name = ready[0]
            action = self._action(name, final=True)
            decision = {"kind": "final", "object": name, "dependency_pressure": pressure[name]}
        else:
            # Resolve the leaves of the observed blocker graph first. Moving a
            # blocker that is itself still blocked only creates another failed
            # buffer and can exhaust otherwise reusable samples.
            movable_blockers = [name for name in sorted(forced_buffers)
                                if not self._above_blockers(name)
                                and (not self._access_blockers(name)
                                     or self._access_retry_arm_available(name))]
            if not movable_blockers:
                movable_blockers = [name for name in candidates if pressure[name] > 0
                                    and not self._above_blockers(name)
                                    and (not self._access_blockers(name)
                                         or self._access_retry_arm_available(name))]
            cycle_component = None
            if not movable_blockers and forced_buffers:
                graph = self._active_dependency_graph(candidates)
                components = self._cyclic_components(graph)
                if components:
                    # Break the actual SCC, excluding upstream objects that
                    # cannot remove the cycle. Prefer a member with an untried
                    # grasp arm and fewer failed buffer attempts.
                    cycle_component = min(components, key=lambda value: (len(value), sorted(value)))
                    cyclic = [name for name in cycle_component
                              if not self._above_blockers(name)]
                    cyclic.sort(key=lambda name: (
                        self._source_failure_count(name),
                        len(set().union(*self.access_failure_arms.get(name, {}).values()))
                            if self.access_failure_arms.get(name) else 0,
                        self.buffer_uses.get(name, 0), -pressure.get(name, 0), name))
                    movable_blockers = cyclic[:1]
                    if movable_blockers:
                        self._set_cycle_escape_arm(movable_blockers[0])
            pool = movable_blockers or [name for name in candidates
                if name not in reserved_away and not self._blocking_objects(name)]
            if not pool:
                raise RuntimeError("Every unfinished object is blocked by another source")
            pool.sort(key=lambda name: (-pressure[name], self.buffer_uses.get(name, 0), name))
            # The highest-pressure blocker may have no reachable free-space
            # node because another object occupies every viable approach.
            # Broaden the search to other movable objects instead of
            # exhausting the same blocker forever. Every alternative still
            # passes the same live geometry, target reservation and IK filters.
            preferred_space_blockers = set().union(*(
                self._successful_buffer_blockers(item) for item in pool)) if pool else set()
            alternatives = [name for name in candidates
                            if name not in pool and not self._above_blockers(name)
                            and not protected_completed_support(name)]
            alternatives.sort(key=lambda name: (
                name not in preferred_space_blockers,
                -pressure.get(name, 0), self.buffer_uses.get(name, 0), name))
            action = None
            exhausted = []
            for name in [*pool, *alternatives]:
                try:
                    action = self._buffer_action(name)
                    break
                except RuntimeError as exc:
                    if not str(exc).startswith("No dynamically sampled free-space placement"):
                        raise
                    exhausted.append(name)
            if action is None:
                raise RuntimeError(
                    f"No dynamically sampled free-space placement for {exhausted}")
            self.pending_buffer_holds[name] = {
                source for source in unfinished
                if name in self._blocking_objects(source)}
            decision = {"kind": "buffer", "object": name,
                        "dependency_pressure": pressure[name],
                        "exhausted_preferred_buffers": exhausted}
            if cycle_component is not None:
                decision["cycle_component"] = sorted(cycle_component)
                decision["cycle_escape_arm"] = self._action_arm(name)
        self.last_decision = decision
        return action, decision
