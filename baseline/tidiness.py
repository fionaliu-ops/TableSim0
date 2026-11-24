"""Observation-only planar tidying adapters; no goal image enters these planners."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from high_level_planner.interface.vla_interface import Pi05VLAInterface
from high_level_planner.reasoning_graph.execution_feedback import destination_failure_errors
from high_level_planner.reasoning_graph.feasibility import evaluate_skill_feasibility
from high_level_planner.reasoning_graph.geometry import pose, rectangle, placement_checks, workspace
from high_level_planner.reasoning_graph.grounding import action_key
from high_level_planner.reasoning_graph.state import AtomicAction, Subtask
from high_level_planner.reasoning_graph.transition import SceneGraphTransitionModel
from .mcts import MCTSConfig, search


def scene_key(graph):
    """Ignore bookkeeping, retain geometry, identity, capabilities and relations."""
    objects = [{"id": o.id, "state": o.state, "pose": pose(graph, o.id),
                "attributes": {k: v for k, v in o.attributes.to_dict().items()
                               if k not in {"bbox", "confidence"}},
                "identity": o.metadata.get("visual_identity_confirmed")}
               for o in sorted(graph.objects, key=lambda item: item.id)]
    relations = sorted((r.source, r.target, r.relation) for r in graph.relations)
    return hashlib.sha256(json.dumps([objects, relations], sort_keys=True).encode()).hexdigest()


def clean_observation(graph):
    graph = graph.copy()
    # Goal-free baselines must not inherit target policies from cached observations.
    context = graph.metadata.get("planning_context", {})
    graph.metadata = {"planning_context": {k: deepcopy(context[k]) for k in
        ("workspace", "objects", "execution_evidence") if k in context}}
    ids = [o.id for o in graph.objects]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("A nonempty observation with unique object IDs is required")
    for obj in graph.objects:
        p = pose(graph, obj.id)
        if p is None or rectangle(graph, obj.id) is None:
            raise ValueError(f"Missing observed pose/footprint/frame for {obj.id}")
        if p.get("support", "table") != "table" or "held" in obj.state:
            raise ValueError("Goal-free baselines support exposed planar objects on the table only")
        table = workspace(graph, p["frame"])
        if not table or table["basis"] != "supplied_workspace":
            raise ValueError("Supply observed/calibrated workspace bounds; image extent is not table geometry")
        if p["frame"] == "image_normalized":
            rect = rectangle(graph, obj.id)
            obj.metadata["baseline_image_reference"] = {
                "footprint": [rect[2] - rect[0], rect[3] - rect[1]],
                "yaw_degrees": p.get("yaw_degrees", 0)}
    return graph


@dataclass
class TidyingState:
    scene: Any
    visual: Any = None


class HeuristicTidiness:
    """Declared geometric proxy, NOT a human-calibrated learned discriminator.

    Alignment uses observed yaw; grouping uses explicitly supplied category.
    Compactness is normalized by the supplied workspace diagonal. Overlap and
    table bounds are hard score gates. Weights are fixed, never tuned on test.
    """
    def __init__(self, *, alignment_weight=.5, compactness_weight=.25, grouping_weight=.25):
        weights = [alignment_weight, compactness_weight, grouping_weight]
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("Heuristic weights must be finite, nonnegative, with positive sum")
        self.weights = weights

    def components(self, state):
        scene = state.scene
        objects = [o for o in scene.objects if o.attributes.movable and o.attributes.portable]
        if not objects:
            raise ValueError("No movable portable objects in observation")
        if len({pose(scene, o.id)["frame"] for o in scene.objects}) != 1:
            raise ValueError("Tidying requires a common coordinate frame")
        validity = all(all(v is True for v in placement_checks(scene, o.id, pose(scene, o.id)).values())
                       for o in objects)
        table = workspace(scene, pose(scene, objects[0].id)["frame"])
        bounds = table["bounds"]
        diagonal = math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1])
        xy = np.asarray([pose(scene, o.id)["xy"] for o in objects])
        yaws = [o.attributes.yaw_degrees for o in objects]
        alignment = float(np.mean([1 - abs((float(y) + 45) % 90 - 45) / 45
                                  if y is not None else 0.0 for y in yaws]))
        compactness = 1 - float(np.linalg.norm(xy.max(0) - xy.min(0))) / diagonal
        groups = {}
        for obj in objects:
            category = obj.attributes.category
            if category and str(category).lower() not in {"unknown", "other", "others"}:
                groups.setdefault(category, []).append(pose(scene, obj.id)["xy"])
        distances = [math.dist(a, b) / diagonal for points in groups.values()
                     for i, a in enumerate(points) for b in points[i + 1:]]
        grouping = 1 - float(np.mean(distances)) if distances else None
        components = [alignment, compactness, grouping]
        active = [(w, v) for w, v in zip(self.weights, components) if v is not None]
        score = sum(w * v for w, v in active) / sum(w for w, _ in active) if sum(w for w, _ in active) else 0.0
        return {"alignment": alignment, "compactness": compactness, "grouping": grouping,
                "geometry_valid": validity, "score": max(0., min(1., score)) if validity else 0.}

    def score(self, state):
        return self.components(state)["score"]


def move_action(scene, object_id, destination):
    obj = scene.get_object(object_id)
    return AtomicAction("move", object_id, "table",
        instruction=f"Move {obj.name} to {destination['frame']} xy={destination['xy']} "
                    f"yaw={destination.get('yaw_degrees', 0):g}deg.",
        parameters={"pose": deepcopy(destination), "placement_target": {
            "kind": "pose_constraint", "reference": "table", "parameters": {"pose": deepcopy(destination)}}})


def legal_action(scene, action):
    if destination_failure_errors(scene, action):
        return False
    source = scene.get_object(action.source_object_id)
    if source is None or source.metadata.get("visual_identity_confirmed",
            source.attributes.extra.get("visual_identity_confirmed")) is False:
        return False
    # Incoming physical support/obstruction is not removed by an imagined move.
    if any(r.target == source.id and r.relation in {"on", "blocks", "occludes", "attached_to"}
           for r in scene.relations):
        return False
    destination = action.parameters["pose"]
    checks_scene = scene
    # Observed image footprints are axis-aligned. Conservatively enlarge the
    # source footprint for a proposed image rotation before collision checking.
    original = pose(scene, source.id)
    if original["frame"] == "image_normalized":
        size = predicted_image_footprint(scene, source.id, destination)
        checks_scene = scene.copy()
        checks_scene.get_object(source.id).attributes.footprint = size
        sidecar = checks_scene.metadata.get("planning_context", {}).get("objects", {}).get(source.id)
        if sidecar is not None:
            sidecar["footprint"] = size
    if any(v is not True for v in placement_checks(checks_scene, source.id, destination).values()):
        return False
    return evaluate_skill_feasibility(scene, action)["status"] != "infeasible"


def predicted_image_footprint(scene, object_id, destination):
    obj = scene.get_object(object_id)
    current = pose(scene, object_id)
    reference = obj.metadata.get("baseline_image_reference", {})
    rect = rectangle(scene, object_id)
    width, height = reference.get("footprint", [rect[2] - rect[0], rect[3] - rect[1]])
    angle = math.radians(destination.get("yaw_degrees", 0) - reference.get("yaw_degrees", current.get("yaw_degrees", 0)))
    return [abs(math.cos(angle)) * width + abs(math.sin(angle)) * height,
            abs(math.sin(angle)) * width + abs(math.cos(angle)) * height]


class GridProposal:
    def __init__(self, grid_size=5, yaws=(0., 90.), max_candidates=64):
        if not isinstance(grid_size, int) or grid_size < 2 or max_candidates < 1:
            raise ValueError("grid_size >= 2 and max_candidates >= 1 required")
        self.grid_size, self.yaws, self.max_candidates = grid_size, tuple(yaws), max_candidates

    def propose(self, state):
        graph, candidates = state.scene, []
        for obj in graph.objects:
            if not obj.attributes.movable or not obj.attributes.portable:
                continue
            original = pose(graph, obj.id)
            x0, y0, x1, y1 = workspace(graph, original["frame"])["bounds"]
            for x in np.linspace(x0, x1, self.grid_size + 2)[1:-1]:
                for y in np.linspace(y0, y1, self.grid_size + 2)[1:-1]:
                    for yaw in self.yaws:
                        destination = {**original, "xy": [float(x), float(y)],
                                       "yaw_degrees": float(yaw), "support": "table"}
                        if math.dist(destination["xy"], original["xy"]) < 1e-8 and abs(
                            (yaw - original.get("yaw_degrees", 0) + 180) % 360 - 180) < 1e-8:
                            continue
                        action = move_action(graph, obj.id, destination)
                        if legal_action(graph, action):
                            candidates.append((action, 1.0))
        # Evenly subsample the full deterministic list; do not keep only the first object.
        if len(candidates) > self.max_candidates:
            indices = np.linspace(0, len(candidates) - 1, self.max_candidates).astype(int)
            candidates = [candidates[i] for i in indices]
        return candidates


class _TidyingDomain:
    def __init__(self, backend, threshold, blocked, root_key):
        self.backend, self.threshold, self.blocked, self.root_key = backend, threshold, blocked, root_key
        self.transition = SceneGraphTransitionModel()
        self.scores = {}

    def key(self, state):
        return scene_key(state.scene)

    def value(self, state):
        key = self.key(state)
        if key not in self.scores:
            value = float(self.backend.score(state))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Tidiness score must be finite and in [0,1]")
            self.scores[key] = value
        return self.scores[key]

    def terminal(self, state):
        # A visual score does not excuse an overlap or out-of-bounds layout.
        return self.value(state) >= self.threshold and self.geometry_valid(state)

    def geometry_valid(self, state):
        return all(all(v is True for v in placement_checks(state.scene, o.id, pose(state.scene, o.id)).values())
                   for o in state.scene.objects if o.attributes.movable and o.attributes.portable)

    def proposals(self, state):
        return [(a, w) for a, w in self.backend.propose(state)
                if legal_action(state.scene, a) and not (
                    self.key(state) == self.root_key and action_key(a) in self.blocked)]

    def step(self, state, action):
        successor = self.transition.apply(state.scene, action)
        destination = action.parameters["pose"]
        if destination["frame"] == "image_normalized":
            size = predicted_image_footprint(state.scene, action.source_object_id, destination)
            successor.get_object(action.source_object_id).attributes.footprint = size
            sidecar = successor.metadata.get("planning_context", {}).get("objects", {}).get(action.source_object_id)
            if sidecar is not None:
                sidecar["footprint"] = size
        if hasattr(self.backend, "transition_visual"):
            return TidyingState(successor, self.backend.transition_visual(state, action))
        return TidyingState(successor)

    def cost(self, state, action):
        return 1.0


class _HeuristicBackend:
    def __init__(self, scorer, proposal):
        self.scorer, self.proposal = scorer, proposal

    def score(self, state):
        return self.scorer.score(state)

    def propose(self, state):
        return self.proposal.propose(state)


@dataclass
class TidinessOutput:
    next_subtask: Subtask | None
    vla_request: Any
    status: str
    reason: str
    metadata: dict

    def to_dict(self):
        return {"next_subtask": self.next_subtask.to_dict() if self.next_subtask else None,
                "vla_request": self.vla_request.to_dict(), "status": self.status,
                "reason": self.reason, "metadata": self.metadata}


class TidinessMCTSPlanner:
    """Receding-horizon interface: observe -> decide -> feedback -> observe.

    done is a scorer judgment of the CURRENT observation; independent external
    evaluation is required. Predicted rollout completion is never actual success.
    """
    mode = "tsmcts_inspired_heuristic"

    def __init__(self, *, config=None, scorer=None, proposal=None, backend=None,
                 threshold=.85, max_steps=50, scene_constructor=None, scene_updater=None):
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0,1]")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.config = config or MCTSConfig()
        self.backend = backend or _HeuristicBackend(scorer or HeuristicTidiness(), proposal or GridProposal())
        self.threshold, self.max_steps = threshold, max_steps
        self.scene_constructor, self.scene_updater = scene_constructor, scene_updater
        self.vla_interface = Pi05VLAInterface()
        self.reset()

    def reset(self):
        self.history, self.decisions, self._failed = [], 0, {}
        self._pending, self._fatal, self._scene = None, False, None
        self.metrics = {"decisions": 0, "executions": 0, "execution_failures": 0,
                        "planning_seconds": 0., "transitions": 0}

    def plan_initial_scene(self, scene, instruction="整理桌面", *, visual=None):
        self.reset()
        return self.update_from_scene(scene, instruction, visual=visual)

    def update_from_scene(self, scene, instruction="整理桌面", *, visual=None):
        if self._pending is not None:
            raise RuntimeError("Record execution feedback before planning again")
        scene = clean_observation(scene)
        self._scene = scene.copy()
        key = scene_key(scene)
        if self._fatal or self.decisions >= self.max_steps:
            return self._output(None, "blocked", "fatal_execution_failure" if self._fatal else "step_limit", {})
        domain = _TidyingDomain(self.backend, self.threshold, self._failed.get(key, set()), key)
        state = TidyingState(scene, visual)
        outcome = search(domain, state, self.config)
        current_score = domain.value(state)
        self.metrics["planning_seconds"] += outcome.metrics["seconds"]
        self.metrics["transitions"] += outcome.metrics["transitions"]
        metadata = {"search": outcome.metrics, "current_score": current_score,
                    "current_geometry_valid": domain.geometry_valid(state),
                    "threshold": self.threshold, "external_success_verified": False,
                    "instruction": instruction, "root_observation_key": key,
                    "scorer_evaluations": len(domain.scores)}
        if hasattr(self.backend, "provenance"):
            metadata["models"] = self.backend.provenance
        if domain.terminal(state):
            return self._output(None, "done", "current_observation_score_threshold", metadata)
        if outcome.action is None:
            return self._output(None, "blocked", "no_action_within_budget", metadata)
        action = outcome.action
        edge = Subtask(f"tidy_{self.decisions}", f"tidy_{self.decisions + 1}", action.action,
            instruction=action.instruction, object_target={"source": action.source_object_id, "target": action.target},
            action_type=action.action_type, parameters=deepcopy(action.parameters), skill=action.skill,
            constraints=action.constraints, preconditions=action.preconditions, effects=action.effects,
            metadata={"baseline": self.mode, "physical_execution_verified": False})
        self.decisions += 1
        self.metrics["decisions"] = self.decisions
        self._pending = (edge, action_key(action), key)
        return self._output(edge, "action", "most_visited_root_child", metadata)

    def _output(self, edge, status, reason, metadata):
        return TidinessOutput(edge, self.vla_interface.format_subtask(edge), status, reason,
                              {"baseline": self.mode, **metadata, "metrics": deepcopy(self.metrics)})

    def record_execution_result(self, subtask, *, success, details=None):
        if self._pending is None or subtask.to_dict() != self._pending[0].to_dict():
            raise ValueError("Feedback must match the pending action")
        _, action, key = self._pending
        details = deepcopy(details or {})
        self.history.append({"action": subtask.to_dict(), "success": bool(success), "details": details})
        self.metrics["executions"] += 1
        self.metrics["execution_failures"] += int(not success)
        if not success:
            self._failed.setdefault(key, set()).add(action)
        self._fatal |= bool(details.get("fatal", False))
        self._pending = None

    def plan_initial_image(self, image, instruction="整理桌面", *, visual=None):
        self.reset()
        return self.process_observation(image, instruction, update=False, visual=visual)

    def process_observation(self, image, instruction="整理桌面", *, update=True, visual=None):
        if self._pending is not None:
            raise RuntimeError("Record execution feedback before planning again")
        if self.scene_constructor is None:
            raise ValueError("Inject the shared SceneGraphConstructor for image observations")
        if update and self._scene is not None:
            if self.scene_updater is None:
                raise ValueError("Inject the shared SceneGraphUpdater for identity tracking")
            scene = self.scene_updater.update_scene_graph(self._scene, image, instruction=instruction)
        else:
            scene = self.scene_constructor.construct(image, instruction=instruction)
        return self.update_from_scene(scene, instruction, visual=visual)

    update_from_image = process_observation
