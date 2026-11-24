"""Execution contracts and bounded sessions for paper-method adaptations."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

from baseline.direct_vlm_planner import DirectVLMPlannerOutput, image_state
from high_level_planner.interface.vla_interface import Pi05VLAInterface
from high_level_planner.reasoning_graph.language_models.base import LanguageModelError
from high_level_planner.reasoning_graph.language_models.openai_compatible import APIConfig, OpenAICompatibleLanguageModel
from high_level_planner.reasoning_graph.state import AtomicAction, Subtask


ACTION_TYPES = frozenset({'move', 'place', 'stack', 'insert', 'align', 'reorient', 'open', 'close', 'attach'})
ACTION_FORMAT = """Return JSON only. A plan is {"status":"plan","actions":[ACTION,...]}.
An ACTION is {"action_type":"move","source_object_id":"known ID","target":"table",
"instruction":"move this object","parameters":{"pose":{"frame":"image_normalized",
"xy":[0.5,0.5],"yaw_degrees":0,"support":"table"}}}.
Each action manipulates one source. A table move/place needs parameters.pose.
For stacking use action_type=stack, target=support ID; optional parameters.relative_pose.
For reorient use parameters.target_yaw_degrees. Coordinates use the supplied image frame,
never world coordinates. An empty plan is {"status":"done","actions":[]} only when
the task is believed complete; inability to plan is {"status":"blocked","actions":[]}.
Allowed actions: move, place, stack, insert, align, reorient, open, close, attach.
Do not invent object IDs. Do not describe predicted states as observed execution.
"""


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def json_value(text):
    if not isinstance(text, str):
        raise ValueError('Model reply must be text')
    if text.strip().startswith('```'):
        text = '\n'.join(text.strip().splitlines()[1:-1])
    def reject(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    result = json.loads(text, parse_constant=reject)
    # JSON permits overflow (1e999); reject it too, including in unused fields.
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Nonfinite number')
        if isinstance(value, dict):
            for item in value.values(): finite(item)
        if isinstance(value, list):
            for item in value: finite(item)
    finite(result)
    return result


def parse_action(data, catalog):
    """Validate message shape/IDs only, without selecting or repairing actions."""
    if not isinstance(data, dict):
        raise ValueError('Action must be an object')
    if data.get('action_type') not in ACTION_TYPES:
        raise ValueError('Unsupported action type')
    source, target = data.get('source_object_id'), data.get('target')
    if not isinstance(source, str) or source not in catalog:
        raise ValueError('Action source must be a supplied object ID')
    if target is not None and (not isinstance(target, str) or target not in {*catalog, 'table'}):
        raise ValueError('Unknown target ID')
    parameters = data.get('parameters', {})
    if not isinstance(parameters, dict):
        raise ValueError('parameters must be an object')
    for key in ('source_object_id', 'source_object_ids'):
        if key in parameters and parameters[key] != (source if key.endswith('_id') else [source]):
            raise ValueError('Conflicting source IDs')
    for key in ('constraints', 'preconditions', 'effects'):
        if not isinstance(data.get(key, []), list) or any(not isinstance(x, str) for x in data.get(key, [])):
            raise ValueError(f'{key} must be a string list')
    if data.get('instruction') is not None and not isinstance(data['instruction'], str):
        raise ValueError('instruction must be a string')
    for key in ('pose', 'relative_pose'):
        if key not in parameters: continue
        pose = parameters[key]
        if not isinstance(pose, dict) or pose.get('frame') != 'image_normalized':
            raise ValueError('Only image_normalized poses are accepted')
        xy = pose.get('xy')
        if not isinstance(xy, list) or len(xy) != 2 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in xy):
            raise ValueError('pose.xy must be two normalized finite coordinates')
        yaw = pose.get('yaw_degrees', 0)
        if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
            raise ValueError('Invalid yaw')
        if pose.get('support', target or 'table') not in {*catalog, 'table'}:
            raise ValueError('Unknown pose support')
    if data['action_type'] in {'move', 'place'} and target in {None, 'table'} and 'pose' not in parameters:
        raise ValueError('Table relocation requires a pose')
    if data['action_type'] == 'reorient':
        yaw = parameters.get('target_yaw_degrees')
        if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
            raise ValueError('Reorient requires a finite target_yaw_degrees')
    # Ignore model-provided metadata: it cannot mark an action verified or change
    # the adapter's execution policy. AtomicAction checks the shared skill schema.
    return AtomicAction(data['action_type'], source, target,
        instruction=data.get('instruction'), parameters=deepcopy(parameters),
        constraints=list(data.get('constraints', [])),
        preconditions=list(data.get('preconditions', [])), effects=list(data.get('effects', [])))


def parse_plan(text, catalog, max_length):
    value = json_value(text)
    if not isinstance(value, dict) or value.get('status') not in {'plan', 'done', 'blocked'}:
        raise ValueError('Expected plan/done/blocked status')
    actions = value.get('actions')
    if not isinstance(actions, list) or len(actions) > max_length:
        raise ValueError('Invalid plan length')
    if (value['status'] == 'plan') != bool(actions):
        raise ValueError('Only plan status may contain actions, and it must be nonempty')
    return value['status'], [parse_action(x, catalog) for x in actions]


@dataclass
class PaperObservation:
    image: object | None = None
    scene: object | None = None


def describe_scene(scene):
    """Text-only passive feedback from sensor-derived scene facts; no sidecar."""
    from high_level_planner.reasoning_graph.geometry import pose
    objects = []
    for obj in scene.objects:
        seen = obj.metadata.get('observed_in_current_frame', True)
        objects.append({'id': obj.id, 'label': obj.name, 'state': obj.state,
            'visibility': obj.attributes.visibility, 'observed_in_current_frame': seen,
            'pose': deepcopy(pose(scene, obj.id)) if seen else None,
            'yaw_degrees': obj.attributes.yaw_degrees if seen else None})
    return {'objects': objects, 'relations': [
        {'source': r.source, 'target': r.target, 'relation': r.relation} for r in scene.relations]}


class PaperPlanner:
    method = 'paper_adaptation'

    def __init__(self, *, model=None, api_config=None, object_catalog,
                 goal_description='', target_image=None, max_steps=50,
                 max_model_calls=200, max_plan_length=32, audit_dir=None):
        if not isinstance(object_catalog, dict) or not object_catalog or any(
            not isinstance(k, str) or not k or not isinstance(v, str) for k, v in object_catalog.items()):
            raise ValueError('object_catalog must map supplied IDs to descriptions')
        self.object_catalog = dict(object_catalog)
        self.model = model if model is not None else OpenAICompatibleLanguageModel(api_config or APIConfig.from_env())
        self.goal_description = str(goal_description)
        self.target_image = image_state(target_image) if target_image is not None else None
        self.max_steps = positive_integer(max_steps, 'max_steps')
        self.max_model_calls = positive_integer(max_model_calls, 'max_model_calls')
        self.max_plan_length = positive_integer(max_plan_length, 'max_plan_length')
        self.audit_dir = Path(audit_dir) if audit_dir is not None else None
        self.vla_interface = Pi05VLAInterface()
        self.reset()

    def reset(self):
        self.history, self.audit, self.plan = [], [], []
        self.pending = None
        self.execution_count = self.failure_count = self.decision_count = 0
        self.model_calls = self.replans = 0
        self.model_seconds = 0.
        self.fatal = self.failed = False
        self.instruction = None
        self.initial_plan_active = False
        self._counter_start = {k: getattr(self.model, k, 0) for k in
                               ('prompt_tokens', 'completion_tokens', 'total_tokens', 'request_count')}

    @property
    def metrics(self):
        data = {'decisions': self.decision_count, 'executions': self.execution_count,
            'execution_failures': self.failure_count, 'model_calls': self.model_calls,
            'model_seconds': self.model_seconds, 'replans': self.replans}
        for key, start in self._counter_start.items():
            now = getattr(self.model, key, None)
            data[key] = now - start if isinstance(now, (int, float)) and isinstance(start, (int, float)) else None
        if data['total_tokens'] is None and all(data[k] is not None for k in ('prompt_tokens', 'completion_tokens')):
            data['total_tokens'] = data['prompt_tokens'] + data['completion_tokens']
        return data

    def context(self):
        return {'instruction': self.instruction, 'goal_description': self.goal_description,
                'object_catalog': self.object_catalog}

    def ask(self, stage, prompt, images=None):
        if self.model_calls >= self.max_model_calls:
            raise LanguageModelError('Paper baseline model-call budget exhausted')
        self.model_calls += 1
        record = {'stage': stage, 'call': self.model_calls, 'prompt': prompt,
                  'images': [x.to_dict() for x in images or []]}
        # In-memory audit preserves image metadata, not repeated base64 content.
        for item in record['images']:
            if item.get('source_type') == 'data_url': item['source'] = '<inline RGB>'
        started = time.perf_counter()
        try:
            reply = self.model.generate(prompt, num_responses=1, images=images)
            if not isinstance(reply, list) or len(reply) != 1 or not isinstance(reply[0], str) or not reply[0].strip():
                raise LanguageModelError('Expected one nonempty model reply')
            record['response'] = reply[0]
            return reply[0]
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            record['seconds'] = time.perf_counter() - started
            self.model_seconds += record['seconds']
            self.audit.append(record)
            if self.audit_dir:
                self.audit_dir.mkdir(parents=True, exist_ok=True)
                (self.audit_dir / f'{self.model_calls:04d}_{stage}.json').write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')

    def load_plan(self, text):
        try:
            return parse_plan(text, self.object_catalog, self.max_plan_length)
        except (ValueError, TypeError, KeyError) as exc:
            raise LanguageModelError(f'Invalid {self.method} plan: {exc}') from exc

    def start(self, observation, instruction, *, initial_plan=None):
        self.reset()
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('instruction must be nonempty')
        self.instruction = instruction
        if initial_plan is not None:
            if not isinstance(initial_plan, list) or not initial_plan:
                raise ValueError('Shared initial_plan must be a nonempty ACTION list')
            _, self.plan = self.load_plan(json.dumps({'status': 'plan', 'actions': initial_plan}, allow_nan=False))
            self.initial_plan_active = True
        return self.step(observation)

    def step(self, observation):
        if self.instruction is None: raise RuntimeError('Start a session first')
        if self.pending is not None: raise RuntimeError('Record execution before another decision')
        if self.fatal: return self.output('blocked', 'fatal_execution_failure')
        if self.decision_count >= self.max_steps: return self.output('blocked', 'step_limit')
        return self.decide(observation)

    def output(self, status, reason='', action=None):
        edge = None
        if action is not None:
            self.decision_count += 1
            edge = Subtask(f'{self.method}_{self.execution_count}', f'{self.method}_{self.execution_count+1}',
                action.instruction or action.action, action_type=action.action_type,
                instruction=action.instruction or action.action,
                object_target={'source': action.source_object_id, 'target': action.target},
                parameters=deepcopy(action.parameters), constraints=list(action.constraints),
                preconditions=list(action.preconditions), effects=list(action.effects), skill=action.skill,
                metadata={'planner': self.method, 'verified': False})
            self.pending = edge
        return DirectVLMPlannerOutput(edge, self.vla_interface.format_subtask(edge), status, reason,
            {'planner': self.method, 'adapted_method': True, 'verified': False, 'metrics': self.metrics})

    def record_execution_result(self, subtask, *, success, details=None, fatal=False):
        if self.pending is None or subtask.to_dict() != self.pending.to_dict():
            raise ValueError('Feedback must identify the pending action')
        if not isinstance(success, bool) or not isinstance(fatal, bool):
            raise TypeError('success/fatal must be boolean')
        if details is not None and not isinstance(details, dict): raise TypeError('details must be a dict')
        event = {'action': {'action_type': subtask.action_type, 'source_object_id': subtask.source_object_id,
                           'target': subtask.object_target.get('target'), 'parameters': deepcopy(subtask.parameters)},
                 'success': success, 'fatal': fatal, 'details': deepcopy(details or {})}
        self.history.append(event)
        self.execution_count += 1
        self.failure_count += int(not success)
        self.failed, self.fatal = not success, fatal
        if success and self.plan: self.plan.pop(0)
        self.pending = None
        return deepcopy(event)
