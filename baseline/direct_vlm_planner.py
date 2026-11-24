"""Single-step visual decision baseline; no graph construction or search."""

from __future__ import annotations

import base64
import io
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from high_level_planner.interface.vla_interface import Pi05VLAInterface, VLARequest
from high_level_planner.reasoning_graph.language_models.base import LanguageModelError
from high_level_planner.reasoning_graph.language_models.openai_compatible import (
    APIConfig, OpenAICompatibleLanguageModel,
)
from high_level_planner.reasoning_graph.state import ATOMIC_ACTION_TYPES, ImageSceneState, Subtask


def image_state(image: str | Path | Image.Image | np.ndarray) -> ImageSceneState:
    """Accept the same local RGB input types as HighLevelPlanner."""
    if isinstance(image, (str, Path)):
        path = Path(image).expanduser().resolve()
        with Image.open(path) as opened:
            opened.verify()
        return ImageSceneState(str(path))
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("RGB array must be uint8 HxWx3")
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a path, PIL image, or RGB numpy array")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return ImageSceneState(
        "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        source_type="data_url", mime_type="image/png",
    )


@dataclass
class DirectVLMPlannerOutput:
    next_subtask: Subtask | None
    vla_request: VLARequest
    status: str
    reason: str
    metadata: dict[str, Any]

    @property
    def task_completed(self) -> bool:
        """VLM completion judgment; an external evaluator must verify success."""
        return self.status == "done"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_graph": None, "reasoning_graph": None, "search_result": None,
            "next_subtask": self.next_subtask.to_dict() if self.next_subtask else None,
            "vla_request": self.vla_request.to_dict(), "status": self.status,
            "reason": self.reason, "metadata": deepcopy(self.metadata),
        }


class DirectVLMPlanner:
    """RGB + instruction + executed-action history -> one Subtask/VLARequest.

    Shared schemas are execution adapters only: their skill contracts are not
    used to infer actions, check scene feasibility, or simulate transitions.
    """

    def __init__(self, *, model=None, api_config: APIConfig | None = None,
                 vla_interface: Pi05VLAInterface | None = None,
                 target_image=None, object_catalog: dict[str, str] | None = None,
                 max_steps: int = 50, history_limit: int = 20):
        for name, value in (("max_steps", max_steps), ("history_limit", history_limit)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.model = model if model is not None else OpenAICompatibleLanguageModel(
            api_config or APIConfig.from_env())
        self.vla_interface = vla_interface or Pi05VLAInterface()
        self.target_image = image_state(target_image) if target_image is not None else None
        self.object_catalog = dict(object_catalog or {})
        if any(not isinstance(k, str) or not k.strip() or not isinstance(v, str)
               for k, v in self.object_catalog.items()):
            raise ValueError("object_catalog must map nonempty object IDs to descriptions")
        self.max_steps, self.history_limit = max_steps, history_limit
        self.reset()

    def reset(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.decision_count = 0
        self.current_node_id = "direct_0"
        self.instruction: str | None = None
        self._pending: Subtask | None = None
        self._last_executed: Subtask | None = None
        self._last_success = False
        self._fatal = False
        self.total_decision_seconds = 0.0
        self._usage_start = {key: getattr(self.model, key, 0) for key in
                             ("request_count", "prompt_tokens", "completion_tokens")}

    @property
    def metrics(self) -> dict[str, Any]:
        """Session counters; transport retries are included in API requests."""
        return {
            "decisions": self.decision_count,
            "executions": len(self.history),
            "execution_failures": sum(not e["success"] for e in self.history),
            "total_decision_seconds": self.total_decision_seconds,
            **{key: getattr(self.model, key, 0) - initial
               for key, initial in self._usage_start.items()},
        }

    def plan_initial_image(self, image, instruction: str) -> DirectVLMPlannerOutput:
        self.reset()
        return self.process_observation(image, instruction, update=True)

    def _prompt(self, instruction: str) -> str:
        return """You are a direct visual tabletop planner. Image 1 is the CURRENT
camera observation. If supplied, image 2 is the fixed GOAL reference.
Decide directly from images, the task, and execution feedback. Return exactly
ONE JSON object, with no markdown. Do not return a plan or a scene graph.
Schema:
{"status":"action|done|blocked", "reason":"brief visible justification",
 "action_type":"move", "instruction":"Move the cup to the clear area.",
 "object_target":{"source":"cup", "target":"table"},
 "parameters":{}, "constraints":[]}
For done/blocked, omit action fields. done means the CURRENT observation
satisfies the task, never merely that an action was issued. blocked means no
action can be selected from available evidence. For action choose exactly one
source object and one atomic manipulation. Use IDs from the catalog when
present, otherwise unambiguous visible labels. Do not manipulate multiple
objects in one instruction. Parameters must describe the chosen action, not
a future plan. For image-derived pose targets use normalized_image coordinates
(xy in [0,1], origin upper left, x right, y down); never invent world coordinates.
A pose may be encoded as parameters.pose={"frame":"normalized_image",
"xy":[0.5,0.5],"yaw_degrees":0}. Other destinations may be object IDs or
executor-supported regions. Physical grounding belongs to the executor.
Allowed action types: """ + ", ".join(sorted(ATOMIC_ACTION_TYPES - {"generic"})) + "\n" + json.dumps({
            "task_instruction": instruction,
            "object_catalog": self.object_catalog,
            "has_goal_image": self.target_image is not None,
            "execution_history": self.history[-self.history_limit:],
        }, ensure_ascii=False)

    def process_observation(self, image, instruction: str, *, update: bool = True):
        if not update:
            self.reset()
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be nonempty")
        if self._pending is not None:
            raise RuntimeError("Record execution feedback before requesting another decision")
        self.instruction = instruction
        if self._fatal or self.decision_count >= self.max_steps:
            return self._empty("blocked", "fatal_execution_failure" if self._fatal else "step_limit")
        images = [image_state(image)]
        if self.target_image is not None:
            images.append(self.target_image)
        started = time.monotonic()
        try:
            answers = self.model.generate(self._prompt(instruction), num_responses=1, images=images)
        finally:
            self.total_decision_seconds += time.monotonic() - started
        self.decision_count += 1
        try:
            if not isinstance(answers, list) or len(answers) != 1:
                raise ValueError("Expected exactly one model answer")
            text = answers[0].strip()
            if text.startswith("```") and text.endswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text, parse_constant=self._reject_constant)
            if not isinstance(data, dict) or data.get("status") not in {"action", "done", "blocked"}:
                raise ValueError("Invalid decision status")
            reason = data.get("reason", "")
            if not isinstance(reason, str):
                raise ValueError("reason must be a string")
            if data["status"] != "action":
                if any(k in data for k in ("action_type", "instruction", "object_target", "parameters")):
                    raise ValueError("Terminal decisions must not include an action")
                return self._empty(data["status"], reason)
            kind, instruction_text = data.get("action_type"), data.get("instruction")
            target, parameters = data.get("object_target"), data.get("parameters", {})
            constraints = data.get("constraints", [])
            if kind not in ATOMIC_ACTION_TYPES - {"generic"}:
                raise ValueError("Unsupported action_type")
            if not isinstance(instruction_text, str) or not instruction_text.strip():
                raise ValueError("Action instruction must be nonempty")
            if not isinstance(target, dict) or not isinstance(target.get("source"), str) or not target["source"].strip():
                raise ValueError("Exactly one source object ID is required")
            if target.get("target") is not None and not isinstance(target["target"], str):
                raise ValueError("target must be a string or null")
            if self.object_catalog and target["source"] not in self.object_catalog:
                raise ValueError("Source not in supplied object catalog")
            if not isinstance(parameters, dict) or not isinstance(constraints, list) or any(not isinstance(c, str) for c in constraints):
                raise ValueError("Invalid parameters or constraints")
            if "source_object_id" in parameters and parameters["source_object_id"] != target["source"]:
                raise ValueError("Conflicting source IDs")
            if "source_object_ids" in parameters and parameters["source_object_ids"] != [target["source"]]:
                raise ValueError("Action must manipulate one source")
            subtask = Subtask(
                from_state=self.current_node_id, to_state=f"direct_{self.decision_count}",
                action=instruction_text.strip(), instruction=instruction_text.strip(),
                action_type=kind, object_target=deepcopy(target), parameters=deepcopy(parameters),
                constraints=list(constraints), metadata={"planner": "direct_vlm", "decision_index": self.decision_count},
            )
            request = self.vla_interface.format_subtask(subtask)
        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
            raise LanguageModelError(f"Invalid Direct VLM decision: {exc}") from exc
        self._pending, self._last_success = subtask, False
        self._last_executed = None
        return DirectVLMPlannerOutput(subtask, request, "action", reason, {
            "planner": "direct_vlm", "decision_index": self.decision_count,
            "decision_seconds": time.monotonic() - started, "verified": False,
            "metrics": self.metrics,
        })

    def _empty(self, status: str, reason: str):
        request = VLARequest("Task completion reported by VLM." if status == "done" else "Planner is blocked.",
                             metadata={"empty": True, "planner": "direct_vlm", "status": status, "verified": False})
        return DirectVLMPlannerOutput(None, request, status, reason, {
            "planner": "direct_vlm", "decision_index": self.decision_count,
            "verified": False, "metrics": self.metrics})

    @staticmethod
    def _reject_constant(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    def update_from_image(self, image, instruction: str):
        return self.process_observation(image, instruction, update=True)

    def update_target_conditioned_image(self, image, instruction: str, *, observation_source="camera_observation"):
        return self.update_from_image(image, instruction)

    def record_execution_result(self, subtask: Subtask, *, success: bool,
                                details: dict[str, Any] | None = None, fatal: bool = False):
        if self._pending is None or subtask.to_dict() != self._pending.to_dict():
            raise ValueError("Execution result must refer to the pending decision")
        if not isinstance(success, bool) or not isinstance(fatal, bool):
            raise TypeError("success and fatal must be booleans")
        if details is not None and not isinstance(details, dict):
            raise TypeError("details must be a dictionary")
        event = {"subtask": subtask.to_dict(), "success": success,
                 "details": deepcopy(details or {}), "fatal": fatal}
        self.history.append(event)
        self._last_success, self._fatal = success and not fatal, fatal
        # Execution feedback resolves the outstanding decision. Cursor advancement
        # is optional and uses the same successful-completion callback as HLP.
        self._last_executed = self._pending
        self._pending = None
        return deepcopy(event)

    def mark_subtask_completed(self, subtask: Subtask | None = None) -> str:
        executed = getattr(self, "_last_executed", None)
        if not self._last_success or executed is None:
            raise ValueError("A successful execution result is required")
        if subtask is not None and subtask.to_dict() != executed.to_dict():
            raise ValueError("Completion callback does not match the executed action")
        self.current_node_id = executed.to_state
        self._last_success = False
        return self.current_node_id
