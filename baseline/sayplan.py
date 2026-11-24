"""SayPlan tabletop adaptation: semantic search and simulator-based plan repair."""
from copy import deepcopy
import json

from high_level_planner.reasoning_graph.feasibility import evaluate_skill_feasibility
from high_level_planner.reasoning_graph.geometry import pose
from high_level_planner.reasoning_graph.operations import is_accessible_relocation, is_accessible_action_target
from high_level_planner.reasoning_graph.transition import SceneGraphTransitionModel

from .paper_common import ACTION_FORMAT, PaperPlanner, describe_scene, json_value, positive_integer


class TabletopHierarchy:
    """Observed table/support/containment hierarchy; no hand-assigned semantic zones."""
    ROOT = '__table__'

    def __init__(self, scene):
        if self.ROOT in {o.id for o in scene.objects}: raise ValueError('Reserved hierarchy root ID')
        self.scene = deepcopy(scene)
        self.parents = {o.id: self.ROOT for o in scene.objects}
        for relation in scene.relations:
            if relation.relation in {'on', 'inside', 'in'} and relation.source in self.parents and relation.target in self.parents:
                self.parents[relation.source] = relation.target
        for key in self.parents:
            seen, current = set(), key
            while current != self.ROOT:
                if current in seen: raise ValueError('Cyclic observed support/containment hierarchy')
                seen.add(current)
                current = self.parents[current]
        self.expanded = set()
        self.memory = []

    def visible_ids(self):
        visible = {self.ROOT}
        changed = True
        while changed:
            old = set(visible)
            visible.update(k for k, parent in self.parents.items() if parent in visible and parent in self.expanded)
            changed = old != visible
        return visible

    def operate(self, operation, node):
        if node not in self.visible_ids(): raise ValueError('Semantic search node is not visible')
        if operation == 'expand':
            self.expanded.add(node)
            if node not in self.memory: self.memory.append(node)
        elif operation == 'contract':
            descendants, frontier = {node}, [node]
            while frontier:
                parent = frontier.pop()
                children = {k for k, p in self.parents.items() if p == parent}
                descendants.update(children)
                frontier.extend(children)
            self.expanded.difference_update(descendants)
        else: raise ValueError('Unknown hierarchy operation')

    def payload(self):
        visible = self.visible_ids()
        # Collapsed entries reveal names, as in semantic search, but not poses,
        # affordances, or relation details until that node is expanded.
        nodes = []
        for key in sorted(visible):
            obj = self.scene.get_object(key)
            entry = {'id': key, 'label': obj.name if obj else 'table',
                     'children': sum(p == key for p in self.parents.values()), 'expanded': key in self.expanded}
            if obj is not None and (key in self.expanded or entry['children'] == 0):
                entry.update(state=obj.state, pose=deepcopy(pose(self.scene, key)),
                    visibility=obj.attributes.visibility,
                    affordances={k: getattr(obj.attributes, k) for k in
                                 ('movable', 'portable', 'container', 'openable', 'stackable')})
            nodes.append(entry)
        return {'nodes': nodes, 'previously_expanded_nodes': list(self.memory),
            'relations': [r for r in describe_scene(self.scene)['relations']
            if r['source'] in visible and r['target'] in visible]}


class TabletopSceneSimulator:
    """Sequential symbolic rollout with explicit failed-action feedback.

    The simulator uses observed graph geometry and the project's transition
    model. It does not generate candidates, search a Reasoning Graph or repair
    actions. Unknown IK/collision checks remain unknown.
    """
    def check(self, scene, plan, target=None, instruction='', position_tolerance=.02):
        current = deepcopy(scene)
        # Never inherit the project's diagnostic 'ignore feasibility' switch.
        current.metadata.pop('graph_validation_ignore_physical_feasibility', None)
        trace = []
        transition = SceneGraphTransitionModel()
        for index, action in enumerate(plan):
            failures = []
            if action.action_type not in {'open', 'close'} and not is_accessible_relocation(current, action):
                failures.append('source_not_accessible_or_not_movable')
            if not is_accessible_action_target(current, action): failures.append('target_not_accessible')
            feasibility = evaluate_skill_feasibility(current, action)
            failures.extend(feasibility['failures'])
            trace.append({'index': index, 'action': action.to_dict(), 'feasibility': feasibility})
            if failures:
                return {'valid': False, 'failed_index': index, 'errors': sorted(set(failures)), 'trace': trace}
            try:
                current = transition.apply(deepcopy(current), action)
            except (ValueError, KeyError, TypeError) as exc:
                return {'valid': False, 'failed_index': index, 'errors': [str(exc)], 'trace': trace}
        if target is not None:
            from high_level_planner.reasoning_graph.target_conditioned import TargetGraphEvaluator
            report = TargetGraphEvaluator(scene, target, position_tolerance).evaluate(current, instruction)
            if not report.terminal:
                return {'valid': False, 'failed_index': None,
                    'errors': ['predicted_goal_not_satisfied'], 'goal_checks': report.goal_checks, 'trace': trace}
        return {'valid': True, 'trace': trace, 'physical_execution_verified': False}


class SayPlanPlanner(PaperPlanner):
    method = 'sayplan'

    def __init__(self, *, target_scene=None, simulator=None, max_search_rounds=24,
                 max_plan_repairs=5, position_tolerance=.02, verify_goal_in_simulator=False, **kwargs):
        self.target_scene = deepcopy(target_scene)
        self.simulator = simulator or TabletopSceneSimulator()
        self.max_search_rounds = positive_integer(max_search_rounds, 'max_search_rounds')
        self.max_plan_repairs = positive_integer(max_plan_repairs, 'max_plan_repairs')
        if not 0 <= position_tolerance < float('inf'): raise ValueError('Invalid position tolerance')
        self.position_tolerance = position_tolerance
        self.verify_goal_in_simulator = bool(verify_goal_in_simulator)
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.simulator_history = []
        self.validated = False

    def semantic_search(self, scene):
        hierarchy = TabletopHierarchy(scene)
        for _ in range(self.max_search_rounds):
            payload = hierarchy.payload()
            reply = self.ask('semantic_search', 'Find the task-relevant subgraph. '
                'Collapsed nodes have labels only; expand a node to inspect its facts and children. '
                'Contract irrelevant subtrees to reduce context. Include blockers and supports needed '
                'to execute the task. Respond with JSON {"operation":"expand|contract|done","node_id":"ID"}.\n'
                + json.dumps({**self.context(), 'hierarchy': payload}, ensure_ascii=False))
            data = json_value(reply)
            if not isinstance(data, dict): raise ValueError('Invalid semantic search response')
            if data.get('operation') == 'done':
                return payload
            hierarchy.operate(data.get('operation'), data.get('node_id'))
        raise RuntimeError('SayPlan semantic-search budget exhausted')

    def decide(self, observation):
        if observation.scene is None: raise ValueError('SayPlan requires an observed scene graph')
        target = self.target_scene if self.verify_goal_in_simulator else None
        # Execute the verified sequence without invoking additional planning on
        # success. Retrying after a task-checker failure is an online extension.
        if self.validated and not self.failed:
            return self.output('action', action=self.plan[0]) if self.plan else self.output('done', 'plan_exhausted')
        if self.plan and not self.failed:
            report = self.simulator.check(observation.scene, self.plan, target,
                                         self.instruction, self.position_tolerance)
            self.simulator_history.append(report)
            if report['valid']:
                self.validated = True
                return self.output('action', action=self.plan[0])
        selected = self.semantic_search(observation.scene)
        feedback = None
        if self.failed:
            # Original validation is simulator feedback. Online retry supplies
            # only a task-checker outcome, never hidden controller geometry.
            feedback = {'execution_failed': True, 'failed_action': self.history[-1]['action']}
        for attempt in range(self.max_plan_repairs):
            prompt = ('Generate a complete ordered task plan over the selected scene graph. '
                'Use scene-simulator feedback to correct the full plan when supplied.\n'
                + ACTION_FORMAT + json.dumps({**self.context(), 'selected_scene_graph': selected,
                    'previous_plan': [a.to_dict() for a in self.plan], 'simulator_feedback': feedback}, ensure_ascii=False))
            status, proposed = self.load_plan(self.ask('plan' if attempt == 0 else 'simulator_replan', prompt))
            if status == 'blocked': return self.output('blocked', 'model_cannot_plan')
            report = self.simulator.check(observation.scene, proposed, target,
                                         self.instruction, self.position_tolerance)
            self.simulator_history.append(report)
            if report['valid']:
                self.plan, self.failed, self.validated = proposed, False, True
                return self.output('action', action=self.plan[0]) if self.plan else self.output('done')
            # Only simulator output drives the next repair. No project graph,
            # UCS or incremental dependency-repair machinery is called.
            self.plan, feedback = proposed, report
            self.replans += 1
        return self.output('blocked', 'simulator_plan_repair_budget_exhausted')
