"""MCTS on the EXACT same preconstructed ReasoningGraph as UCS."""
from collections import defaultdict
import heapq
import math
import time

from high_level_planner.planner.cost import CostFunction
from high_level_planner.planner.ucs import SearchResult
from .mcts import MCTSConfig, search


def valid_edge(edge):
    return edge.metadata.get("dependencies", {}).get("validity", "valid") == "valid"


class _GraphDomain:
    def __init__(self, graph, cost_fn):
        self.graph, self.cost_fn = graph, cost_fn
        self.adjacency = defaultdict(list)
        for edge in graph.edges:
            if valid_edge(edge):
                self.adjacency[edge.from_state].append(edge)

    def key(self, state):
        return state

    def terminal(self, state):
        return self.graph.is_goal(state)

    def value(self, state):
        if self.terminal(state):
            return 1.0
        checks = self.graph.get_node(state).goal_checks
        return sum(checks.values()) / len(checks) if checks else 0.0

    def proposals(self, state):
        # Uniform priors isolate search from proposal-model differences.
        return [(edge, 1.0) for edge in self.adjacency[state]]

    def step(self, state, edge):
        return edge.to_state

    def cost(self, state, edge):
        return self.cost_fn.edge_cost(edge)


class GoalConditionedMCTSPlanner:
    """Drop-in GraphSearchPlanner; only returns a certified graph goal path.

    Goal paths discovered by rollouts are also retained. Among paths observed
    within budget the cheapest is returned, NOT a claim of global optimality.
    Failure to find a goal is never reported as task completion.
    """
    def __init__(self, cost_fn=None, config=None):
        self.cost_fn = cost_fn or CostFunction()
        self.config = config or MCTSConfig()
        self.metrics = {}

    def plan(self, reasoning_graph, *, start_id=None):
        started = time.perf_counter()
        graph = reasoning_graph
        start = start_id or graph.start_id
        outcome = search(_GraphDomain(graph, self.cost_fn), start, self.config, started_at=started)
        self.metrics = {**outcome.metrics, "mode": "goal_conditioned_mcts",
                        "selection": "cheapest_goal_path_found_within_budget",
                        "graph_nodes": len(graph.nodes), "graph_edges": len(graph.edges)}
        return SearchResult(path=outcome.goal_path, total_cost=outcome.goal_cost,
                            visited=outcome.metrics["visited"], start_state=start,
                            goal_state=outcome.goal_state)


class BudgetedUCSPlanner:
    """UCS with the same cooperative time / transition evaluation limits.

    A transition evaluation is one outgoing valid edge examined. MCTS may
    examine the same edge repeatedly. Graph generation is excluded for both.
    """
    def __init__(self, cost_fn=None, *, max_transitions=4096, time_limit_seconds=None):
        self.cost_fn = cost_fn or CostFunction()
        MCTSConfig(max_transitions=max_transitions, time_limit_seconds=time_limit_seconds)
        self.max_transitions = max_transitions
        self.time_limit_seconds = time_limit_seconds
        self.metrics = {}

    def plan(self, graph, *, start_id=None):
        started = time.perf_counter()
        start = start_id or graph.start_id
        queue, best, visited = [(0.0, 0, start, [])], {start: 0.0}, []
        transitions, counter, result, stop = 0, 1, None, "no_path"
        adjacency = defaultdict(list)
        for edge in graph.edges:
            if valid_edge(edge):
                adjacency[edge.from_state].append(edge)
        if graph.is_goal(start):
            visited = [start]
            result, stop = SearchResult([], 0.0, visited, start, start), "goal"
            queue = []
        while queue:
            if self.time_limit_seconds is not None and time.perf_counter() - started >= self.time_limit_seconds:
                stop = "time_budget"
                break
            cost, _, state, path = heapq.heappop(queue)
            if cost > best[state]:
                continue
            visited.append(state)
            if graph.is_goal(state):
                result = SearchResult(path, cost, visited, state, start)
                stop = "goal"
                break
            interrupted = False
            for edge in adjacency[state]:
                if transitions >= self.max_transitions or (
                    self.time_limit_seconds is not None and
                    time.perf_counter() - started >= self.time_limit_seconds
                ):
                    stop, interrupted = "budget", True
                    break
                transitions += 1
                edge_cost = float(self.cost_fn.edge_cost(edge))
                if not math.isfinite(edge_cost) or edge_cost < 0:
                    raise ValueError("UCS requires finite nonnegative costs")
                new_cost = cost + edge_cost
                if new_cost < best.get(edge.to_state, math.inf):
                    best[edge.to_state] = new_cost
                    heapq.heappush(queue, (new_cost, counter, edge.to_state, [*path, edge]))
                    counter += 1
            if interrupted:
                break
        self.metrics = {"seconds": time.perf_counter() - started, "transitions": transitions,
                        "visited": visited, "stop_reason": stop,
                        "max_transitions": self.max_transitions,
                        "time_limit_seconds": self.time_limit_seconds}
        return result or SearchResult([], math.inf, visited, None, start)
