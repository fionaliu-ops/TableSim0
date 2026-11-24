"""Render dynamic simulation trace into inspectable GIF/MP4 artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size=20):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _scene_panel(path: Path, title: str, active: str | None = None) -> Image.Image:
    data = json.loads(path.read_text(encoding="utf-8"))
    objects = data.get("objects", {})
    if isinstance(objects, list):
        objects = {str(o.get("id", o.get("object_id", i))): o for i, o in enumerate(objects)}
    W, H = 720, max(420, 80 + 42 * len(objects))
    im = Image.new("RGB", (W, H), "#101820")
    d = ImageDraw.Draw(im)
    d.text((18, 12), title, fill="white", font=_font(24))
    y = 58
    for oid, obj in objects.items():
        pose = obj.get("observed_pose") or obj.get("placement") or {}
        xy = pose.get("xy", [0.5, 0.5])
        x = int(40 + float(xy[0]) * 640)
        yy = int(40 + float(xy[1]) * 300)
        color = "#ffcc66" if oid == active else "#66ccff"
        d.ellipse((x - 8, yy - 8, x + 8, yy + 8), fill=color)
        d.text((18, y), f"{oid}: {obj.get('state', obj.get('category', 'visible'))}",
               fill=color, font=_font(16))
        y += 38
    relations = data.get("relations", [])
    d.text((400, 58), "relations", fill="#aaaaaa", font=_font(16))
    for i, rel in enumerate(relations[:14]):
        if isinstance(rel, dict):
            s = rel.get("source", rel.get("subject", "?")); t = rel.get("target", "?")
            r = rel.get("relation", rel.get("predicate", "?"))
            d.text((400, 84 + i * 22), f"{s} -{r}-> {t}", fill="#dddddd", font=_font(12))
    return im


def _reasoning_panel(path: Path, title: str, current: str | None) -> Image.Image:
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    if isinstance(nodes, dict): nodes = list(nodes.values())
    W, H = 920, max(440, 80 + 52 * min(len(nodes), 32))
    im = Image.new("RGB", (W, H), "#17151f"); d = ImageDraw.Draw(im)
    d.text((18, 12), title, fill="white", font=_font(24))
    pos = {}
    for i, n in enumerate(nodes[:32]):
        x = 50 + (i % 4) * 215; y = 65 + (i // 4) * 55; pos[n.get("id", str(i))] = (x, y)
        nid = n.get("id", str(i)); fill = "#e09045" if nid == current else "#4b5875"
        d.rounded_rectangle((x, y, x + 170, y + 32), radius=7, fill=fill)
        d.text((x + 6, y + 8), nid, fill="white", font=_font(14))
    for e in data.get("edges", []):
        a, b = e.get("from_state"), e.get("to_state")
        if a in pos and b in pos: d.line((pos[a][0] + 85, pos[a][1] + 32, pos[b][0] + 85, pos[b][1]), fill="#777777", width=2)
    return im


def _write_video(frames, out: Path, fps=2):
    import imageio.v2 as imageio
    if not frames: return
    imageio.mimsave(out, [f.convert("RGB") for f in frames], fps=fps, codec="libx264", quality=7)


def main(root: str):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    trace = json.loads((root / "dynamic_trace.json").read_text(encoding="utf-8"))
    steps = trace.get("steps", [])
    sg_dir = root / "sam3_scene_graph_frames"; rg_dir = root / "reasoning_graph_frames"; sim_dir = root / "simulation_frames"
    for p in (sg_dir, rg_dir, sim_dir): p.mkdir(exist_ok=True)
    sg_frames = []; rg_frames = []; sim_frames = []
    motion_files = sorted((root / "motion_frames").glob("motion_*_head_camera.png"))
    front_files = sorted((root / "motion_frames").glob("motion_*_front_camera.png"))
    if motion_files:
        for i, obs in enumerate(motion_files):
            f = Image.open(obs).convert("RGB").resize((640, 480))
            ImageDraw.Draw(f).text((12, 12), f"robot motion frame {i:03d} ({obs.stem.split('_')[2]})", fill="yellow", font=_font(20))
            f.save(sim_dir / f"frame_{i:03d}.png"); sim_frames.append(f)
    for i in range(len(steps) + 1):
        sg = root / f"scene_graph_{i:03d}.json"; rg = root / f"reasoning_graph_{i:03d}.json"
        if not sg.exists(): sg = root / "scene_graph_000.json"
        if not rg.exists(): rg = root / "reasoning_graph_000.json"
        active = None
        if i and i <= len(steps): active = steps[i-1].get("request", {}).get("object_target", {}).get("source")
        f = _scene_panel(sg, f"SAM3 scene graph observation {i:03d}", active); f.save(sg_dir / f"frame_{i:03d}.png"); sg_frames.append(f)
        current = steps[i-1].get("reasoning_to") if i else None
        f = _reasoning_panel(rg, f"Reasoning graph checkpoint {i:03d}", current); f.save(rg_dir / f"frame_{i:03d}.png"); rg_frames.append(f)
        if motion_files:
            continue
        obs = root / ("observation_000.png" if i == 0 else str(steps[i-1].get("observation", "")))
        if not obs.exists(): obs = root / "observation_000.png"
        if obs.exists():
            f = Image.open(obs).convert("RGB").resize((640, 480)); ImageDraw.Draw(f).text((12, 12), f"simulation observation {i:03d}", fill="yellow", font=_font(20)); f.save(sim_dir / f"frame_{i:03d}.png"); sim_frames.append(f)
    _write_video(sg_frames, root / "sam3_scene_graph_dynamic.mp4"); _write_video(rg_frames, root / "reasoning_graph_dynamic.mp4"); _write_video(sim_frames, root / "simulation_effect.mp4")
    if front_files:
        front_frames = []
        for i, obs in enumerate(front_files):
            f = Image.open(obs).convert("RGB").resize((640, 480))
            ImageDraw.Draw(f).text((12, 12), f"gripper/front-camera frame {i:03d}", fill="yellow", font=_font(20))
            front_frames.append(f)
        _write_video(front_frames, root / "simulation_gripper_front_camera.mp4")
    import imageio.v2 as imageio
    for name, frames in (("sam3_scene_graph_dynamic.gif", sg_frames), ("reasoning_graph_dynamic.gif", rg_frames), ("simulation_effect.gif", sim_frames)):
        if frames: imageio.mimsave(root / name, [f.convert("RGB") for f in frames], duration=0.7, loop=0)
    if front_files:
        imageio.mimsave(root / "simulation_gripper_front_camera.gif", [f.convert("RGB") for f in front_frames], duration=0.5, loop=0)
    (root / "visualization_manifest.json").write_text(json.dumps({"steps": len(steps), "sam3_frames": len(sg_frames), "reasoning_frames": len(rg_frames), "simulation_frames": len(sim_frames)}, indent=2) + "\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("root"); main(p.parse_args().root)
