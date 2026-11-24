"""Conservative packing for the manifest-derived, normalized layout benchmark.

This operates on planar boxes, not robot geometry or image perception. A failed
packing is reported before any paid model calls, never accepted as a goal.
"""
from copy import deepcopy
import math

from high_level_planner.reasoning_graph.geometry import intersects, rectangle


def compact_target(initial, margin=0.015):
    target = deepcopy(initial)
    objects = {o.id: o for o in target.objects}
    for obj in target.objects:
        size = obj.attributes.footprint
        if (not size or len(size) != 2
                or any(not math.isfinite(v) or v <= 0 for v in size)
                or rectangle(target, obj.id) is None):
            raise ValueError(f"Missing or invalid planar geometry for {obj.id}")
    parents = {}
    for relation in target.relations:
        if relation.relation != "on":
            continue
        if relation.source in parents or relation.target not in objects or relation.source not in objects:
            raise ValueError("Support graph must have one known parent per supported object")
        parents[relation.source] = relation.target
    if len(set(parents.values())) != len(parents):
        raise ValueError("Multiple objects on one support require a separate support-surface layout")

    def root(key):
        seen = set()
        while key in parents:
            if key in seen:
                raise ValueError("Cyclic support graph")
            seen.add(key)
            key = parents[key]
        return key

    for key, parent in parents.items():
        root(key)
        child_size = objects[key].attributes.footprint
        parent_size = objects[parent].attributes.footprint
        if any(a + 2 * margin > b for a, b in zip(child_size, parent_size)):
            raise ValueError(f"{key} does not fit on {parent} with the required clearance")

    roots = [o for o in target.objects if o.id not in parents]
    fixed = [o for o in roots if not o.attributes.movable or not o.attributes.portable]
    moving = [o for o in roots if o not in fixed]
    for key in parents:
        obj = objects[key]
        if not obj.attributes.movable or not obj.attributes.portable:
            raise ValueError(f"Cannot relocate support group containing fixed object {key}")
    fixed_boxes = [rectangle(target, o.id) for o in fixed]
    if any(b is None for b in fixed_boxes):
        raise ValueError("Missing geometry for fixed obstacle")
    # Anchor at the front edge of the rear device; without one use rear center.
    anchor = (sum(o.attributes.xy[0] for o in fixed) / len(fixed),
              max(b[3] for b in fixed_boxes) + margin) if fixed else (0.5, 0.15)
    orders = [sorted(moving, key=lambda o: -o.attributes.footprint[0] * o.attributes.footprint[1]),
              sorted(moving, key=lambda o: -max(o.attributes.footprint)), moving]
    chosen = None
    for order in orders:
        occupied = list(fixed_boxes)
        positions = {}
        for obj in order:
            width, height = obj.attributes.footprint
            candidates = []
            for ix in range(101):
                for iy in range(101):
                    x, y = ix / 100, iy / 100
                    box = [x - width / 2, y - height / 2, x + width / 2, y + height / 2]
                    if min(box[:2]) < margin or max(box[2:]) > 1 - margin:
                        continue
                    if any(intersects(box, obstacle, margin) for obstacle in occupied):
                        continue
                    score = (x - anchor[0]) ** 2 + (y - anchor[1]) ** 2
                    candidates.append((score, x, y, box))
            if not candidates:
                break
            _, x, y, box = min(candidates)
            positions[obj.id] = [x, y]
            occupied.append(box)
        if len(positions) == len(moving):
            chosen = positions
            break
    if chosen is None:
        raise ValueError("No collision-free compact target found; review scene geometry/clearance")
    for obj in target.objects:
        key = root(obj.id)
        obj.attributes.xy = list(chosen.get(key, objects[key].attributes.xy))
        support = parents.get(obj.id, "table")
        obj.metadata["planning_pose"] = {
            "xy": list(obj.attributes.xy), "frame": "image_normalized",
            "yaw_degrees": obj.attributes.yaw_degrees or 0.0, "support": support,
        }
        obj.state = "on " + (objects[support].name if support != "table" else "desk")
    # Independent final pair check; supported objects intentionally overlap
    # their own group but must not overlap another support group.
    for i, a in enumerate(target.objects):
        for b in target.objects[i + 1:]:
            if root(a.id) != root(b.id) and intersects(rectangle(target, a.id), rectangle(target, b.id), margin):
                raise ValueError(f"Target occupancy conflict: {a.id}, {b.id}")
    target.metadata.update(terminal_target=True, target_policy="footprint_compact_packing",
                           layout_clearance=margin, layout_validation="planar_proxy_only")
    return target


def planning_budgets(initial, max_actions=None, max_states=None, branches=3):
    movable = sum(o.attributes.movable and o.attributes.portable for o in initial.objects)
    supports = sum(r.relation == "on" for r in initial.relations)
    # Allow a buffer move, final move and articulation adjustment per object,
    # plus clearing/restoring support relations.
    actions = max_actions if max_actions is not None else max(16, 3 * movable + 2 * supports + 4)
    if actions < movable + supports:
        raise ValueError("Action budget too small for moving and restoring this scene")
    states = max_states if max_states is not None else max(128, 4 * branches * actions)
    if states < actions + 1:
        raise ValueError("State budget must accommodate a complete action chain including its initial state")
    return actions, states
