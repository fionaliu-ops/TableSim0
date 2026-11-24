#!/usr/bin/env python3
"""Debug why scence_generator always fails to place the 'on' relationship."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "script"))

import scence_generator as sg  # noqa: E402


class Args:
    min_objects = 9
    max_objects = 11
    min_placed = 9
    render_steps = 4
    width = 512
    height = 384
    allow_missing_on = True
    allow_missing_occlusion = True
    output_dir = ROOT / "image_collectors" / "scenes_new" / "_debug"


def main() -> None:
    args = Args()
    rng = np.random.default_rng(20260920)
    messy = sg.load_messy_config()
    specs = sg.load_specs()
    current_labels = set(messy["objects"])
    label_specs = {
        label: spec_list
        for label, spec_list in sg.build_label_specs(specs).items()
        if label in current_labels and label not in sg.EXCLUDED_DESK_LABELS
    }
    variant_cache = {
        spec["output"]: sg.discover_variants(spec)
        for spec in specs
        if any(label in label_specs for label in spec["objects"])
    }
    label_specs = {
        label: [s for s in spec_list if variant_cache.get(s["output"])]
        for label, spec_list in label_specs.items()
    }
    label_specs = {k: v for k, v in label_specs.items() if v}
    print("available labels:", sorted(label_specs))
    for key in ("pen", "notebook", "keyboard", "laptop"):
        specs_for = label_specs.get(key)
        if not specs_for:
            print(f"!!! label {key!r} has NO usable spec/variants")
            continue
        for s in specs_for:
            print(f"label {key!r}: output={s['output']} kind={s['kind']} "
                  f"variants={[v.display_id for v in variant_cache[s['output']]]}")

    import sapien  # noqa: F401

    for attempt in range(6):
        scene = sapien.Scene()
        try:
            scene.set_timestep(1 / 250)
            sg.add_table_scene(scene, rng)
            available_labels = set(label_specs)
            labels, relations, co_pairs, occ = sg.sample_scene_labels(
                rng, messy, available_labels, args.min_objects, args.max_objects,
            )
            print(f"\n=== attempt {attempt} labels={labels}")
            print("on relations sampled:",
                  [r for r in relations if r["predicate"].lower() == "on"])
            order = [l for l in sg.object_order(labels, relations)
                     if l in sg.REAR_CENTER_DEVICE_LABELS] + \
                    [l for l in sg.object_order(labels, relations)
                     if l not in sg.REAR_CENTER_DEVICE_LABELS]
            loaded, placed, skipped = {}, {}, []
            for label in order:
                choice = sg.choose_object(rng, label, label_specs, variant_cache)
                if choice is None:
                    skipped.append((label, "no mapped asset variant"))
                    continue
                try:
                    obj = sg.load_object(scene, choice)
                except Exception as e:
                    skipped.append((label, str(e)[:120]))
                    continue
                loaded[label] = obj
                obj.yaw = 0.0 if label in sg.REAR_CENTER_DEVICE_LABELS else float(
                    rng.uniform(-np.pi, np.pi))
                if label in sg.REAR_CENTER_DEVICE_LABELS:
                    xy = sg.choose_rear_center_pose(rng, obj, placed)
                    placement = (xy, 0.0, None) if xy is not None else None
                else:
                    rear = next((o for o in placed.values()
                                 if o.label in sg.REAR_CENTER_DEVICE_LABELS
                                 and o.xy is not None), None)
                    fym = float(rear.xy[1] - sg.FRONT_CLEARANCE) if rear else None
                    rel = sg.relation_for_label(label, relations, placed)
                    placement = sg.choose_pose(rng, obj, rel, placed, front_y_max=fym)
                if placement is None:
                    sg.move_offscreen(obj)
                    skipped.append((label, "could not place"))
                    continue
                xy, z, _ = placement
                sg.place_loaded_object(obj, xy, z, obj.yaw)
                placed[label] = obj
            print("skipped:", skipped)
            print("placed:", {k: (v.xy.round(3).tolist(), round(v.z_offset, 4),
                                  round(v.height, 4), round(v.radius, 4))
                              for k, v in placed.items()})
            on_rels = [r for r in relations if r["predicate"].lower() == "on"]
            applied = sg.reapply_on_relations(rng, placed, relations)
            print("reapply applied:", applied)
            for r in on_rels:
                sub = placed.get(r["subject"])
                sup = placed.get(r["object"])
                if sub is None or sup is None:
                    print(f"  on-relation {r}: subject/support NOT PLACED "
                          f"(subject={r['subject'] in placed}, object={r['object'] in placed})")
                    continue
                print(f"  on-relation {r['subject']} on {r['object']}: "
                      f"support xy={sup.xy.round(3).tolist()} h={sup.height:.4f} "
                      f"r={sup.radius:.4f} fp={sup.footprint.round(3).tolist()}")
                # probe collision outcome at the exact candidate used by reapply
                z_offset = sup.z_offset + sup.height + 0.006
                fails = 0
                for _ in range(60):
                    xy = sup.xy + rng.normal(
                        0.0, min(0.018, max(0.006, sup.radius * 0.08)), size=2)
                    xy = sg.clip_xy_for_table(sub, xy)
                    if sg.collides(sub, xy, placed,
                                   allow_overlap_with=sup.label, z_offset=z_offset):
                        fails += 1
                print(f"    collision fails: {fails}/60  "
                      f"sub h={sub.height:.4f} fp={sub.footprint.round(3).tolist()}")
        finally:
            scene.clear()


if __name__ == "__main__":
    main()
