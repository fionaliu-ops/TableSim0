"""Ablate incremental reasoning-graph repair with unconditional full replanning."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

from high_level_planner.evidence import EvidenceStore
from high_level_planner.planner.planner import HighLevelPlanner, PlannerOutput
from high_level_planner.reasoning_graph.target_conditioned import generate_between_scenes_with_generator
from high_level_planner.scene_graph.graph import SceneGraph


class FullReplanPlanner(HighLevelPlanner):
    """Same image-first task and execution contract, a new graph every round.

    Perception still maintains instance identity across frames. Only future
    reasoning nodes, edges, cached computations and search cursors are rebuilt.
    Evidence and failed-execution suppression match HighLevelPlanner.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reset_metrics()

    def _reset_metrics(self):
        self._replanning_history: list[dict[str, Any]] = []
        self._closed_loop_decisions = []

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "planning_rounds": len(self._replanning_history),
            "full_replans": sum(r["mode"] == "full_rebuild" for r in self._replanning_history),
            "total_planning_seconds": sum(r["planning_seconds"] for r in self._replanning_history),
        }

    def _annotate(self, output: PlannerOutput, *, mode: str, started: float,
                  evidence_update=None) -> PlannerOutput:
        record = {
            "sequence": len(self._replanning_history) + 1,
            "mode": mode,
            "reason": "initial_plan" if mode == "initial_plan" else "full_replan_every_observation",
            "planning_seconds": time.monotonic() - started,
            "evidence_version": self.evidence_store.version,
            "evidence_update": evidence_update.to_dict() if evidence_update is not None else None,
            "start_state": output.search_result.start_state,
            "goal_state": output.search_result.goal_state,
            "node_count": len(output.reasoning_graph.nodes),
            "edge_count": len(output.reasoning_graph.edges),
            "preserve_execution": False,
        }
        self._replanning_history.append(record)
        output.reasoning_graph.metadata["baseline"] = {
            "name": "full_replan", "latest": deepcopy(record),
            "history": deepcopy(self._replanning_history), "metrics": self.metrics,
            "target_regenerated": False, "incremental_repair_used": False,
        }
        if mode == "full_rebuild":
            output.reasoning_graph.metadata["closed_loop"] = {
                "schema_version": 1,
                "latest": {**deepcopy(record), "decision": {
                    "mode": "full_rebuild", "reason": "full_replan_every_observation"}},
                "history": deepcopy(self._replanning_history),
            }
        return output

    def plan_initial_image(self, image, instruction, *, goal_generator=None,
                           position_tolerance: float = 0.02):
        self._reset_metrics()
        started = time.monotonic()
        output = super().plan_initial_image(
            image, instruction, goal_generator=goal_generator,
            position_tolerance=position_tolerance,
        )
        return self._annotate(output, mode="initial_plan", started=started)

    def plan_initial_scene(self, initial: SceneGraph, target: SceneGraph,
                           instruction: str, *, position_tolerance: float = 0.02):
        """Start with shared cached graphs; do not regenerate the comparison goal."""
        self._reset_metrics()
        started = time.monotonic()
        initial, target = deepcopy(initial), deepcopy(target)
        store = EvidenceStore()
        evidence_update = store.update_from_scene(initial, source="initial_cached_observation")
        store.attach_execution_context(initial)
        graph, generator = generate_between_scenes_with_generator(
            initial, target, instruction, generator=self.reasoning_generator,
            position_tolerance=position_tolerance,
        )
        self.evidence_store = store
        if self._target_update_subscription is not None:
            self.scene_updater.unsubscribe_updates(self._target_update_subscription)
        self._target_reasoning_generator = generator
        self._target_update_subscription = self.scene_updater.subscribe_updates(generator.handle_scene_graph_update)
        self._target_scene_graph = target
        self._target_position_tolerance = float(position_tolerance)
        result = self.graph_planner.plan(graph, start_id=graph.start_id)
        request = self.vla_interface.format_subtask(result.next_subtask)
        self.memory.update(initial, graph, preserve_execution=False)
        output = PlannerOutput(initial, graph, result, result.next_subtask, request)
        return self._annotate(output, mode="initial_plan", started=started,
                              evidence_update=evidence_update)

    def process_observation(self, image, instruction: str, *, update: bool = True):
        if not update or self._target_reasoning_generator is None:
            return self.plan_initial_image(image, instruction)
        return self.update_target_conditioned_image(image, instruction)

    def update_from_image(self, image, instruction: str):
        return self.update_target_conditioned_image(image, instruction)

    def update_target_conditioned_scene(
        self, scene_graph: SceneGraph, instruction: str, *,
        observation_source: str = "camera_observation", coverage: dict[str, Any] | None = None,
    ) -> PlannerOutput:
        if self._target_reasoning_generator is None or self._target_scene_graph is None:
            raise ValueError("No image-first target-conditioned planning session is active")
        if (self.evidence_store.last_execution or {}).get("fatal", False):
            raise RuntimeError("Closed-loop scheduler stopped execution: fatal_execution_failure")
        started = time.monotonic()
        scene = deepcopy(scene_graph)
        evidence_update = self.evidence_store.update_from_scene(
            scene, source=observation_source, coverage=coverage)
        self.evidence_store.attach_execution_context(scene)
        # SceneGraphUpdater publishes before the evidence context is attached.
        # Use this authoritative, evidence-enriched observation, not a queued
        # pre-enrichment snapshot selected by generate().
        self._target_reasoning_generator.consume_scene_graph_update()
        graph = self._full_target_rebuild(scene, instruction)
        result = self.graph_planner.plan(graph, start_id=graph.start_id)
        request = self.vla_interface.format_subtask(result.next_subtask)
        self.memory.update(scene, graph, preserve_execution=False)
        output = PlannerOutput(scene, graph, result, result.next_subtask, request)
        return self._annotate(output, mode="full_rebuild", started=started,
                              evidence_update=evidence_update)
