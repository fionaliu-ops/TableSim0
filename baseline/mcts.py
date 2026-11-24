"""Bounded, deterministic-domain UCT shared by the three MCTS baselines.

The domain supplies observations/predictions, never hidden simulator state.
Proposal probabilities affect expansion/rollout sampling, not the UCT formula.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import random
import time
from typing import Any, Protocol


@dataclass(frozen=True)
class MCTSConfig:
    iterations: int = 256
    max_depth: int = 12
    rollout_depth: int = 4
    exploration: float = 1.0
    rollout_mix: float = 0.3
    cost_weight: float = 0.01
    max_transitions: int = 4096
    time_limit_seconds: float | None = None
    seed: int = 0

    def __post_init__(self):
        for name in ("iterations", "max_depth", "max_transitions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.rollout_depth, bool) or not isinstance(self.rollout_depth, int) or self.rollout_depth < 0:
            raise ValueError("rollout_depth must be a nonnegative integer")
        for name in ("exploration", "cost_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.rollout_mix) or not 0 <= self.rollout_mix <= 1:
            raise ValueError("rollout_mix must be in [0,1]")
        if self.time_limit_seconds is not None and (
            not math.isfinite(self.time_limit_seconds) or self.time_limit_seconds <= 0
        ):
            raise ValueError("time_limit_seconds must be finite and positive")


class Domain(Protocol):
    def key(self, state) -> Any: ...
    def terminal(self, state) -> bool: ...
    def value(self, state) -> float: ...
    def proposals(self, state) -> list[tuple[Any, float]]: ...
    def step(self, state, action): ...
    def cost(self, state, action) -> float: ...


@dataclass
class _Node:
    state: Any
    parent: _Node | None = None
    action: Any = None
    visits: int = 0
    total: float = 0.0
    children: list[_Node] = field(default_factory=list)
    pending: list[tuple[Any, float]] | None = None


@dataclass
class MCTSOutcome:
    action: Any
    goal_path: list[Any]
    goal_state: Any
    goal_cost: float
    metrics: dict


def search(domain: Domain, initial, config: MCTSConfig, *, started_at=None) -> MCTSOutcome:
    started = time.perf_counter() if started_at is None else started_at
    rng = random.Random(config.seed)
    root = _Node(initial)
    counts = {"iterations": 0, "transitions": 0, "value_evaluations": 0,
              "proposal_evaluations": 0, "cycle_rejections": 0}
    visited = []
    best_path, best_state, best_cost = [], None, math.inf

    def exhausted():
        return counts["transitions"] >= config.max_transitions or (
            config.time_limit_seconds is not None
            and time.perf_counter() - started >= config.time_limit_seconds)

    def proposals(state):
        counts["proposal_evaluations"] += 1
        items = list(domain.proposals(state))
        for _, weight in items:
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Proposal weights must be finite and nonnegative")
        return items

    def sample_index(items):
        weights = [item[1] for item in items]
        return rng.choices(range(len(items)), weights=weights if sum(weights) else None)[0]

    def advance(state, action):
        counts["transitions"] += 1
        successor = domain.step(state, action)
        visited.append(str(domain.key(successor)))
        return successor

    def record_goal(state, path, cost):
        nonlocal best_path, best_state, best_cost
        if domain.terminal(state) and cost < best_cost:
            best_path, best_state, best_cost = list(path), state, cost

    def action_cost(state, action):
        result = float(domain.cost(state, action))
        if not math.isfinite(result) or result < 0:
            raise ValueError("MCTS requires finite nonnegative action costs")
        return result

    record_goal(initial, [], 0.0)
    if best_state is None:
        for _ in range(config.iterations):
            if exhausted():
                break
            node, path, chain, depth, path_cost = root, [], [root], 0, 0.0
            seen = {domain.key(initial)}
            while depth < config.max_depth and not domain.terminal(node.state) and not exhausted():
                if node.pending is None:
                    node.pending = proposals(node.state)
                if exhausted():
                    break
                if node.pending:
                    action, _ = node.pending.pop(sample_index(node.pending))
                    successor = advance(node.state, action)
                    key = domain.key(successor)
                    if key in seen:
                        counts["cycle_rejections"] += 1
                        continue
                    path_cost += action_cost(node.state, action)
                    node = _Node(successor, node, action)
                    node.parent.children.append(node)
                    chain.append(node)
                    path.append(action)
                    seen.add(key)
                    depth += 1
                    break
                if not node.children:
                    break
                parent_visits = max(1, node.visits)
                child = max(node.children, key=lambda c: (
                    c.total / c.visits + config.exploration * math.sqrt(
                        2 * math.log(parent_visits + 1) / c.visits)
                    if c.visits else math.inf))
                key = domain.key(child.state)
                if key in seen:
                    counts["cycle_rejections"] += 1
                    break
                path_cost += action_cost(node.state, child.action)
                node = child
                chain.append(node)
                path.append(node.action)
                seen.add(key)
                depth += 1

            record_goal(node.state, path, path_cost)
            counts["value_evaluations"] += 1
            value = float(domain.value(node.state))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Domain value must be finite and in [0,1]")
            rollout, rollout_path, rollout_cost = node.state, list(path), path_cost
            for _ in range(min(config.rollout_depth, config.max_depth - depth)):
                if domain.terminal(rollout) or exhausted():
                    break
                choices = proposals(rollout)
                if not choices or exhausted():
                    break
                action, _ = choices[sample_index(choices)]
                successor = advance(rollout, action)
                key = domain.key(successor)
                if key in seen:
                    counts["cycle_rejections"] += 1
                    break
                rollout_cost += action_cost(rollout, action)
                rollout, rollout_path = successor, [*rollout_path, action]
                seen.add(key)
            record_goal(rollout, rollout_path, rollout_cost)
            reward = ((1 - config.rollout_mix) * value
                      + config.rollout_mix * float(domain.terminal(rollout))
                      - config.cost_weight * rollout_cost)
            for ancestor in chain:
                ancestor.visits += 1
                ancestor.total += reward
            counts["iterations"] += 1

    chosen = max(root.children, key=lambda c: (c.visits, c.total / max(1, c.visits)), default=None)
    metrics = {**counts, "seconds": time.perf_counter() - started,
               "config": asdict(config), "visited": visited,
               "stop_reason": "root_terminal" if domain.terminal(initial) else
                   "budget" if exhausted() else "iterations",
               "root_actions": [{"action": c.action.to_dict() if hasattr(c.action, "to_dict") else str(c.action),
                                 "visits": c.visits, "mean_value": c.total / max(1, c.visits)}
                                for c in root.children]}
    return MCTSOutcome(chosen.action if chosen else None, best_path, best_state, best_cost, metrics)
