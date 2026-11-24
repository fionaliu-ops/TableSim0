from __future__ import annotations

import argparse
import base64
import io
import json
import math
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


SEMANTIC_RELATIONS = {"on", "blocks", "occludes", "next_to"}
SPATIAL_RELATIONS = {"left_of", "right_of", "in_front_of", "behind", "behind_image", "in_front_image"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize TableSim planner graph JSON outputs.")
    parser.add_argument("--input-dir", default="planner_outputs/scene_000")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-spatial-edges", type=int, default=32)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def wrap(text: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False)) or ""


def short_id(object_id: str) -> str:
    parts = object_id.split("_")
    if len(parts) <= 3:
        return object_id
    return "_".join(parts[:3])


def object_lookup(scene_graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {obj["id"]: obj for obj in scene_graph.get("objects", [])}


def object_label(obj: dict[str, Any]) -> str:
    attrs = obj.get("attributes", {})
    xy = attrs.get("xy")
    xy_text = f"xy=({xy[0]:.2f},{xy[1]:.2f})" if xy else "xy=?"
    flags = []
    if attrs.get("visibility") == "partially_visible":
        flags.append("partial")
    if not attrs.get("portable", True):
        flags.append("not-portable")
    if attrs.get("z_offset", 0) > 0.01:
        flags.append("stacked")
    flag_text = ", ".join(flags) if flags else "normal"
    return f"{obj['name']}\n{short_id(obj['id'])}\n{obj.get('state', '')}\n{xy_text}\n{flag_text}"


def relation_distance(rel: dict[str, Any], objects: dict[str, dict[str, Any]]) -> float:
    source = objects.get(rel["source"], {})
    target = objects.get(rel["target"], {})
    sxy = source.get("attributes", {}).get("xy")
    txy = target.get("attributes", {}).get("xy")
    if not sxy or not txy:
        return float("inf")
    return float(np.linalg.norm(np.asarray(sxy, dtype=float) - np.asarray(txy, dtype=float)))


def select_scene_relations(
    scene_graph: dict[str, Any],
    *,
    max_spatial_edges: int,
) -> tuple[list[dict[str, Any]], int]:
    objects = object_lookup(scene_graph)
    semantic = []
    spatial_candidates: dict[tuple[str, str], dict[str, Any]] = {}

    for rel in scene_graph.get("relations", []):
        relation = rel.get("relation")
        metadata_source = rel.get("metadata", {}).get("source")
        if relation in SEMANTIC_RELATIONS or metadata_source == "manifest_relationship":
            semantic.append(rel)
            continue

        if relation not in SPATIAL_RELATIONS:
            continue

        pair = tuple(sorted([rel["source"], rel["target"]]))
        existing = spatial_candidates.get(pair)
        if existing is None or relation_distance(rel, objects) < relation_distance(existing, objects):
            spatial_candidates[pair] = rel

    spatial = sorted(spatial_candidates.values(), key=lambda item: relation_distance(item, objects))
    selected = semantic + spatial[:max_spatial_edges]
    omitted = max(0, len(scene_graph.get("relations", [])) - len(selected))
    return selected, omitted


def node_colors(scene_graph: dict[str, Any]) -> dict[str, str]:
    colors = {}
    blockers = set()
    blocked_targets = set()
    for rel in scene_graph.get("relations", []):
        if rel.get("relation") in {"on", "blocks", "occludes"}:
            blockers.add(rel["source"])
            blocked_targets.add(rel["target"])

    for obj in scene_graph.get("objects", []):
        attrs = obj.get("attributes", {})
        if obj["id"] in blocked_targets or attrs.get("visibility") == "partially_visible":
            colors[obj["id"]] = "#ffb3ba"
        elif obj["id"] in blockers:
            colors[obj["id"]] = "#ffd28a"
        elif not attrs.get("portable", True):
            colors[obj["id"]] = "#d4d4d4"
        else:
            colors[obj["id"]] = "#a9d6ff"
    return colors


def draw_scene_network(
    scene_graph: dict[str, Any],
    output_dir: Path,
    *,
    max_spatial_edges: int,
    dpi: int,
) -> None:
    objects = scene_graph.get("objects", [])
    relations, omitted = select_scene_relations(scene_graph, max_spatial_edges=max_spatial_edges)
    colors = node_colors(scene_graph)

    graph = nx.MultiDiGraph()
    for obj in objects:
        graph.add_node(obj["id"], label=object_label(obj))
    for rel in relations:
        graph.add_edge(rel["source"], rel["target"], relation=rel["relation"], confidence=rel.get("confidence", 1.0))

    seed = 7
    xy_positions = {
        obj["id"]: np.asarray(obj.get("attributes", {}).get("xy"), dtype=float)
        for obj in objects
        if obj.get("attributes", {}).get("xy") is not None
    }
    if len(xy_positions) >= max(3, len(objects) // 2):
        pos = {key: np.asarray([value[0], value[1]]) for key, value in xy_positions.items()}
        missing = [obj["id"] for obj in objects if obj["id"] not in pos]
        for index, object_id in enumerate(missing):
            angle = 2.0 * math.pi * index / max(1, len(missing))
            pos[object_id] = np.asarray([0.55 * math.cos(angle), 0.55 * math.sin(angle)])
    else:
        pos = nx.spring_layout(graph, seed=seed, k=1.2)

    fig, ax = plt.subplots(figsize=(22, 16))
    ax.set_title("Scene Graph - selected semantic + nearest spatial relations", fontsize=18, pad=18)
    ax.axis("off")

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=14,
        width=[
            2.6 if data.get("relation") in {"on", "blocks", "occludes"} else 1.2
            for _, _, data in graph.edges(data=True)
        ],
        edge_color=[
            "#d62728" if data.get("relation") == "blocks" else "#9467bd" if data.get("relation") == "on" else "#8a8a8a"
            for _, _, data in graph.edges(data=True)
        ],
        alpha=0.72,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=[colors[node] for node in graph.nodes],
        node_size=3600,
        edgecolors="#303030",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={obj["id"]: object_label(obj) for obj in objects},
        ax=ax,
        font_size=7,
        font_family="DejaVu Sans",
    )

    edge_labels = {}
    for source, target, key, data in graph.edges(keys=True, data=True):
        relation = data["relation"]
        if relation in SEMANTIC_RELATIONS:
            edge_labels[(source, target, key)] = relation
        elif relation in {"left_of", "right_of"}:
            edge_labels[(source, target, key)] = "L/R"
        elif relation in {"in_front_of", "behind"}:
            edge_labels[(source, target, key)] = "F/B"
        else:
            edge_labels[(source, target, key)] = relation

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        ax=ax,
        font_size=7,
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.72},
    )

    relation_counts = Counter(rel["relation"] for rel in scene_graph.get("relations", []))
    note = [
        f"objects: {len(objects)}",
        f"relations total: {len(scene_graph.get('relations', []))}",
        f"relations drawn: {len(relations)}",
        f"relations omitted: {omitted}",
        "",
        "relation counts:",
    ]
    note.extend(f"- {name}: {count}" for name, count in sorted(relation_counts.items()))
    ax.text(
        1.01,
        0.98,
        "\n".join(note),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "fc": "#f7f7f7", "ec": "#cccccc"},
    )

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="blocked / partially visible target", markerfacecolor="#ffb3ba", markeredgecolor="#303030", markersize=12),
        Line2D([0], [0], marker="o", color="w", label="blocker / stacked object", markerfacecolor="#ffd28a", markeredgecolor="#303030", markersize=12),
        Line2D([0], [0], marker="o", color="w", label="regular portable object", markerfacecolor="#a9d6ff", markeredgecolor="#303030", markersize=12),
        Line2D([0], [0], marker="o", color="w", label="not portable / static", markerfacecolor="#d4d4d4", markeredgecolor="#303030", markersize=12),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "scene_graph.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "scene_graph.svg", bbox_inches="tight")
    plt.close(fig)


def draw_scene_spatial(
    scene_graph: dict[str, Any],
    output_dir: Path,
    *,
    max_spatial_edges: int,
    dpi: int,
) -> None:
    objects = scene_graph.get("objects", [])
    object_map = object_lookup(scene_graph)
    colors = node_colors(scene_graph)
    relations, _ = select_scene_relations(scene_graph, max_spatial_edges=max_spatial_edges)

    fig, ax = plt.subplots(figsize=(16, 13))
    ax.set_title("Scene Graph Spatial Layout - tabletop xy view", fontsize=17, pad=16)
    ax.set_xlabel("x (table coordinate)")
    ax.set_ylabel("y (table coordinate)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    xy_values = [obj.get("attributes", {}).get("xy") for obj in objects if obj.get("attributes", {}).get("xy") is not None]
    if xy_values:
        arr = np.asarray(xy_values, dtype=float)
        pad = 0.12
        ax.set_xlim(float(arr[:, 0].min() - pad), float(arr[:, 0].max() + pad))
        ax.set_ylim(float(arr[:, 1].min() - pad), float(arr[:, 1].max() + pad))

    for obj in objects:
        attrs = obj.get("attributes", {})
        xy = attrs.get("xy")
        if xy is None:
            continue
        footprint = attrs.get("footprint") or [0.05, 0.05]
        width = max(0.025, min(0.12, float(footprint[0])))
        height = max(0.025, min(0.12, float(footprint[1])))
        rect = Rectangle(
            (xy[0] - width / 2.0, xy[1] - height / 2.0),
            width,
            height,
            facecolor=colors[obj["id"]],
            edgecolor="#222222",
            linewidth=1.2,
            alpha=0.85,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            xy[0],
            xy[1],
            f"{obj['name']}\n{short_id(obj['id'])}",
            ha="center",
            va="center",
            fontsize=7,
            zorder=4,
        )

    for rel in relations:
        src = object_map.get(rel["source"], {})
        tgt = object_map.get(rel["target"], {})
        sxy = src.get("attributes", {}).get("xy")
        txy = tgt.get("attributes", {}).get("xy")
        if sxy is None or txy is None:
            continue

        relation = rel.get("relation")
        if relation == "blocks":
            color, width, alpha, label = "#d62728", 2.8, 0.9, "blocks"
        elif relation == "on":
            color, width, alpha, label = "#9467bd", 2.8, 0.9, "on"
        elif relation == "next_to":
            color, width, alpha, label = "#2ca02c", 2.0, 0.8, "next_to"
        else:
            color, width, alpha, label = "#9a9a9a", 0.8, 0.22, ""

        arrow = FancyArrowPatch(
            sxy,
            txy,
            arrowstyle="-|>",
            mutation_scale=10 if label else 5,
            color=color,
            linewidth=width,
            alpha=alpha,
            connectionstyle="arc3,rad=0.08",
            zorder=2,
        )
        ax.add_patch(arrow)
        if label:
            mid = ((sxy[0] + txy[0]) / 2.0, (sxy[1] + txy[1]) / 2.0)
            ax.text(mid[0], mid[1], label, color=color, fontsize=9, weight="bold", zorder=5)

    ax.text(
        0.01,
        0.01,
        "All object nodes are drawn at manifest xy coordinates.\n"
        "Semantic relations are emphasized; nearest spatial relations are faint gray arrows.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#cccccc", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(output_dir / "scene_graph_spatial.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "scene_graph_spatial.svg", bbox_inches="tight")
    plt.close(fig)


def summarize_facts(facts: dict[str, Any], max_items: int = 8) -> str:
    items = list(facts.items())
    lines = []
    for key, value in items[:max_items]:
        lines.append(f"{short_id(key)}: {value}")
    if len(items) > max_items:
        lines.append(f"... +{len(items) - max_items} more facts")
    return "\n".join(lines)


def draw_reasoning_graph(reasoning_graph: dict[str, Any], output_dir: Path, *, dpi: int) -> None:
    nodes = reasoning_graph.get("nodes", [])
    edges = reasoning_graph.get("edges", [])
    node_map = {node["id"]: node for node in nodes}
    goal_ids = set(reasoning_graph.get("goal_ids", []))
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(node_map)
    for index, edge in enumerate(edges):
        graph.add_edge(
            edge["from_state"],
            edge["to_state"],
            key=index,
            action=edge.get("action", ""),
        )

    layers: dict[int, list[str]] = {}
    for node_id, node in node_map.items():
        depth = int(node.get("metadata", {}).get("depth", 0))
        layers.setdefault(depth, []).append(node_id)
    for layer in layers.values():
        layer.sort(key=lambda item: (0, int(item[1:])) if item[1:].isdigit() else (1, item))

    max_layer_size = max((len(layer) for layer in layers.values()), default=1)
    max_depth = max(layers, default=0)
    has_image_nodes = any(
        node.get("scene_state", {}).get("representation") == "image"
        for node in nodes
    )
    fig_width = max(14.0, (4.0 if has_image_nodes else 2.5) * (max_depth + 1))
    fig_height = max(9.0, (2.5 if has_image_nodes else 0.62) * max_layer_size + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_title(
        (
            "Reasoning DAG — imagined image states and atomic-action edges"
            if has_image_nodes
            else "Reasoning DAG — Scene Graph states and atomic-action edges"
        ),
        fontsize=17,
        pad=16,
    )
    ax.axis("off")

    positions: dict[str, tuple[float, float]] = {}
    for depth, layer in sorted(layers.items()):
        y_values = np.linspace(
            (len(layer) - 1) / 2.0,
            -(len(layer) - 1) / 2.0,
            len(layer),
        )
        for node_id, y_value in zip(layer, y_values):
            positions[node_id] = (
                float(depth) * (4.0 if has_image_nodes else 2.8),
                float(y_value) * (2.4 if has_image_nodes else 1.0),
            )

    start_id = reasoning_graph.get("start_id")
    colors = []
    labels = {}
    for node_id in graph.nodes:
        node = node_map[node_id]
        metadata = node.get("metadata", {})
        status = metadata.get("search_status", "")
        checks = metadata.get("goal_report", {}).get("goal_checks", {})
        passed = sum(bool(value) for value in checks.values())
        if node_id in goal_ids:
            colors.append("#9be28f")
            suffix = "GOAL"
        elif node_id == start_id:
            colors.append("#8ec5ff")
            suffix = "START"
        elif status == "beam_pruned":
            colors.append("#dedede")
            suffix = "pruned"
        elif status == "frontier":
            colors.append("#ffd27f")
            suffix = "frontier"
        else:
            colors.append("#fff0a8")
            suffix = status or "state"
        labels[node_id] = (
            f"{node_id}\n{passed}/{len(checks)}\n{suffix}"
            if checks
            else f"{node_id}\n{suffix}"
        )

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        arrows=True,
        arrowsize=9,
        width=0.65,
        edge_color="#66717d",
        alpha=0.48,
        connectionstyle="arc3,rad=0.035",
    )
    fallback_nodes = []
    color_by_node = dict(zip(graph.nodes, colors))
    for node_id in graph.nodes:
        image_array = _reasoning_node_image(node_map[node_id])
        if image_array is None:
            fallback_nodes.append(node_id)
            continue
        border = (
            "#2ca02c" if node_id in goal_ids
            else "#1f77b4" if node_id == start_id
            else "#d99a00"
        )
        artist = AnnotationBbox(
            OffsetImage(image_array, zoom=0.115),
            positions[node_id],
            frameon=True,
            bboxprops={"edgecolor": border, "linewidth": 2.0, "boxstyle": "round,pad=0.08"},
            zorder=4,
        )
        ax.add_artist(artist)
        ax.annotate(
            labels[node_id],
            positions[node_id],
            xytext=(0, -54),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": border, "alpha": 0.9},
            zorder=5,
        )
    if fallback_nodes:
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=fallback_nodes,
            ax=ax,
            node_color=[color_by_node[node_id] for node_id in fallback_nodes],
            node_size=720,
            edgecolors="#333333",
            linewidths=0.75,
        )
        nx.draw_networkx_labels(
            graph,
            positions,
            labels={node_id: labels[node_id] for node_id in fallback_nodes},
            ax=ax,
            font_size=5.5 if len(nodes) > 80 else 7,
            font_family="DejaVu Sans",
        )

    # Detailed edge labels remain useful for small graphs but make a real DAG
    # unreadable.  Large graphs keep full action details in the JSON and the
    # companion tabular plan SVG.
    if len(edges) <= 60:
        edge_labels = {
            (edge["from_state"], edge["to_state"], index): wrap(
                edge.get("action", ""), 22
            )
            for index, edge in enumerate(edges)
        }
        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels=edge_labels,
            ax=ax,
            font_size=6,
            rotate=False,
            bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.7},
        )

    metadata = reasoning_graph.get("metadata", {})
    ax.text(
        0.01,
        0.03,
        f"source: {metadata.get('source')}\n"
        f"instruction: {wrap(metadata.get('instruction', ''), 80)}\n"
        f"nodes: {len(nodes)}, edges: {len(edges)}, goals: {', '.join(reasoning_graph.get('goal_ids', []))}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "fc": "#f7f7f7", "ec": "#cccccc"},
    )

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="start", markerfacecolor="#8ec5ff", markeredgecolor="#333333", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="expanded/frontier", markerfacecolor="#ffd27f", markeredgecolor="#333333", markersize=9),
    ]
    if any(
        node.get("metadata", {}).get("search_status") == "beam_pruned"
        for node in nodes
    ):
        legend_items.append(
            Line2D([0], [0], marker="o", color="w", label="beam-pruned leaf", markerfacecolor="#dedede", markeredgecolor="#333333", markersize=9)
        )
    legend_items.append(
        Line2D([0], [0], marker="o", color="w", label="goal", markerfacecolor="#9be28f", markeredgecolor="#333333", markersize=9)
    )
    ax.legend(handles=legend_items, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "reasoning_graph.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "reasoning_graph.svg", bbox_inches="tight")
    plt.close(fig)


def _reasoning_node_image(node: dict[str, Any]) -> np.ndarray | None:
    scene_state = node.get("scene_state") or {}
    if scene_state.get("representation") != "image":
        return None
    image = scene_state.get("image") or {}
    source = str(image.get("source", ""))
    source_type = image.get("source_type", "path")
    try:
        if source_type == "data_url" and "," in source:
            return plt.imread(io.BytesIO(base64.b64decode(source.split(",", 1)[1])))
        if source_type == "path" and Path(source).is_file():
            return plt.imread(source)
    except (OSError, ValueError):
        return None
    return None


def draw_reasoning_plan(
    reasoning_graph: dict[str, Any],
    output_dir: Path,
    *,
    dpi: int,
) -> None:
    """Render the lowest-cost goal path, one atomic action per row."""

    all_edges = reasoning_graph.get("edges", [])
    goal_ids = set(reasoning_graph.get("goal_ids", []))
    start_id = reasoning_graph.get("start_id")
    # Use the planner's composite cost and tie-breaking, so this figure agrees
    # with ucs_path_output/optimal_path.json (motion-only NetworkX paths can differ).
    from high_level_planner.reasoning_graph.graph import ReasoningGraph
    from high_level_planner.planner.ucs import uniform_cost_search

    result = uniform_cost_search(ReasoningGraph.from_dict(reasoning_graph))
    edges = [edge.to_dict() for edge in result.path]
    rows = []
    for index, edge in enumerate(edges, start=1):
        target = edge.get("object_target", {})
        rows.append(
            [
                str(index),
                f"{edge.get('from_state', '')} → {edge.get('to_state', '')}",
                str(edge.get("action_type", edge.get("action", ""))),
                wrap(
                    f"{short_id(str(target.get('source', '')))} → "
                    f"{target.get('target', '')}",
                    28,
                ),
                wrap(edge.get("instruction", edge.get("action", "")), 54),
                "GOAL" if edge.get("to_state") in goal_ids else "",
            ]
        )
    if not rows:
        rows.append([
            "—", str(start_id or "—"), "—", "—",
            "No executable path reaches a goal node in the model graph.", "NO GOAL",
        ])

    fig_height = max(8.0, 0.55 * len(rows) + 3.0)
    fig, ax = plt.subplots(figsize=(20, fig_height))
    ax.axis("off")
    ax.set_title(
        "Selected Goal Path — one atomic action per edge",
        fontsize=18,
        pad=18,
    )
    table = ax.table(
        cellText=rows,
        colLabels=["Step", "State", "Atomic action", "Source → target", "VLA instruction", "After"],
        colWidths=[0.05, 0.09, 0.10, 0.20, 0.47, 0.09],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.25)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#b6b6b6")
        if row == 0:
            cell.set_facecolor("#2f5597")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif rows[row - 1][-1] == "GOAL":
            cell.set_facecolor("#c7f0c2")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f7fb")
        else:
            cell.set_facecolor("white")
        if column in {0, 5}:
            cell.get_text().set_ha("center")

    metadata = reasoning_graph.get("metadata", {})
    fig.text(
        0.02,
        0.02,
        f"model/source: {metadata.get('source')}   "
        f"reasoning mode: {metadata.get('reasoning_mode')}   "
        f"rounds: {len(metadata.get('reasoning_rounds', []))}   "
        f"graph: {len(reasoning_graph.get('nodes', []))} nodes / {len(all_edges)} edges   "
        f"selected path: {len(edges)} edges   "
        f"goal: {', '.join(sorted(goal_ids)) or 'not reached'}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0.01, 0.04, 0.99, 0.98])
    fig.savefig(output_dir / "reasoning_graph_plan.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "reasoning_graph_plan.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_graph_path = input_dir / "scene_graph.json"
    reasoning_graph_path = input_dir / "reasoning_graph.json"
    if not scene_graph_path.exists():
        raise FileNotFoundError(scene_graph_path)
    if not reasoning_graph_path.exists():
        raise FileNotFoundError(reasoning_graph_path)

    scene_graph = load_json(scene_graph_path)
    reasoning_graph = load_json(reasoning_graph_path)

    draw_scene_network(scene_graph, output_dir, max_spatial_edges=args.max_spatial_edges, dpi=args.dpi)
    draw_scene_spatial(scene_graph, output_dir, max_spatial_edges=args.max_spatial_edges, dpi=args.dpi)
    draw_reasoning_graph(reasoning_graph, output_dir, dpi=args.dpi)
    draw_reasoning_plan(reasoning_graph, output_dir, dpi=args.dpi)

    print("wrote:", output_dir / "scene_graph.png")
    print("wrote:", output_dir / "scene_graph.svg")
    print("wrote:", output_dir / "scene_graph_spatial.png")
    print("wrote:", output_dir / "scene_graph_spatial.svg")
    print("wrote:", output_dir / "reasoning_graph.png")
    print("wrote:", output_dir / "reasoning_graph.svg")
    print("wrote:", output_dir / "reasoning_graph_plan.png")
    print("wrote:", output_dir / "reasoning_graph_plan.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
