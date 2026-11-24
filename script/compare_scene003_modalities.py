"""Compare Kimi action plans from two images versus two SAM3 scene graphs.

Both routes use live head-camera input, the same supplied terminal image,
identical action schema and one execution adapter. Neither asks Kimi to
produce an image or a target scene graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from high_level_planner.perception.sam3_client import Sam3PerceptionClient
from high_level_planner.reasoning_graph.language_models.openai_compatible import OpenAICompatibleLanguageModel
from high_level_planner.reasoning_graph.parser import ReasoningGraphParser
from high_level_planner.reasoning_graph.state import ImageSceneState, SimpleSceneGraph
from high_level_planner.reasoning_graph.target_conditioned import prompt_graph
from high_level_planner.scene_graph.constructor import SceneGraphConstructor
from script.dynamic_sim import build_environment, DynamicSimError
from script.goal_config import dynamic_reasoning_config, load_api_settings

INSTRUCTION = ("Organize the desktop compactly near the monitor, straighten and stack "
               "the book and folder neatly, align the keyboard, phone and mouse, and preserve all objects.")
COMMON = """You are planning an ordered sequence of physical tabletop actions from the initial state to the supplied terminal state. Return strict JSON: {"actions": [{"action_type":"move|place|stack|align|reorient|open|close", "source_object_id":"initial ID", "target":"table or another initial ID", "pose":{"frame":"image_normalized","xy":[x,y],"support":"table or target ID"}, "reason":"short observable reason"}],"uncertainties":["..."]}. At most 10 actions. Each action moves one object. For a table move, pose.xy is the center in the TERMINAL image/graph, divided by image width and height. Do not supply real-world dimensions. Clear objects above an item before moving that item. Do not invent a missing object's terminal pose. Include a support target for stacking. Return only actions supported by the input; state uncertainty when identity or support is unclear. Do not claim physical success.\n"""


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_image(source, destination):
    """Encode the same visual evidence compactly for the model transport."""
    image = Image.open(source).convert("RGB")
    image.thumbnail((960, 720))
    image.save(destination, format="JPEG", quality=88, optimize=True)
    return ImageSceneState(str(destination.resolve()), mime_type="image/jpeg")


def install_recording(env):
    original_reset, original_close = env.reset, env.close
    writers = {}
    state = {"ticks": 0, "frames": 0, "stride": 1}
    def capture():
        env.task._update_render()
        env.task.cameras.update_picture()
        rgb = env.task.cameras.get_rgb()
        for name in ("head_camera", "front_camera"):
            if name not in rgb: continue
            if name not in writers:
                writers[name] = imageio.get_writer(str(env.output_dir / f"operation_{name}.mp4"),
                    fps=1.0 / (env.task.scene.get_timestep() * state["stride"]),
                    codec="libx264", quality=8, macro_block_size=1)
            writers[name].append_data(rgb[name]["rgb"])
        state["frames"] += 1
    def reset(seed=0):
        obs = original_reset(seed)
        state["stride"] = max(1, round(1.0 / (25 * env.task.scene.get_timestep())))
        original_sync = env.task._sync_attached_actor
        def sync():
            original_sync()
            state["ticks"] += 1
            if state["ticks"] % state["stride"] == 0: capture()
        env.task._sync_attached_actor = sync
        capture()
        return obs
    def close():
        try:
            if writers:
                capture()
            for writer in writers.values(): writer.close()
            write(env.output_dir / "video_manifest.json", {
                "frames":state["frames"], "control_ticks":state["ticks"],
                "actual_simulation_frames":True,
                "has_robot_control_steps":state["ticks"]>0 and not getattr(env, 'object_motion_validation', False),
                "has_object_motion_steps":state["ticks"]>0 and getattr(env, 'object_motion_validation', False),
                "cameras":list(writers)})
        finally: original_close()
    env.reset, env.close = reset, close
    return env


def match_book_instances(initial, task):
    """Execution-only mapping from SAM3 book instances to known scene actors."""
    mapping = {}
    books = sorted((o for o in initial.objects if o.name == "book" and o.attributes.xy),
                   key=lambda o:o.attributes.xy[0])
    if len(books) == 2 and {"folder", "book"} <= set(task.objects):
        mapping[books[0].id] = "folder"
        mapping[books[1].id] = "book"
    return mapping


def parse_plan(text, initial_ids):
    data = ReasoningGraphParser._json_value(text)
    if not isinstance(data, dict) or not isinstance(data.get("actions"), list):
        raise ValueError("Model did not return an actions array")
    if len(data["actions"]) > 10: raise ValueError("Model exceeded action budget")
    for index, action in enumerate(data["actions"]):
        if not isinstance(action, dict): raise ValueError(f"Action {index} is not an object")
        if action.get("action_type") not in {"move","place","stack","align","reorient","open","close"}:
            raise ValueError(f"Action {index} has unsupported type")
        if action.get("source_object_id") not in initial_ids:
            raise ValueError(f"Action {index} references unknown source {action.get('source_object_id')!r}")
        if action.get("target") not in ({"table",None} | initial_ids):
            raise ValueError(f"Action {index} references unknown target")
        if action["action_type"] not in {"open","close"}:
            pose = action.get("pose")
            if not isinstance(pose, dict) or pose.get("frame") != "image_normalized":
                raise ValueError(f"Action {index} lacks image-normalized pose")
            xy = np.asarray(pose.get("xy"), dtype=float)
            if xy.shape != (2,) or not np.isfinite(xy).all() or not np.all((0 <= xy) & (xy <= 1)):
                raise ValueError(f"Action {index} has invalid image pose")
    return data


def request_from_action(action):
    return {"instruction": action.get("reason") or "Move object toward terminal layout",
            "object_target":{"source":action["source_object_id"],"target":action.get("target")},
            "metadata":{"reasoning_edge":{"action_type":action["action_type"],
                "parameters":{"pose":action.get("pose",{})}}}}


def run(mode, root, goal, settings, sam3_endpoint):
    out = root / mode
    out.mkdir(parents=True, exist_ok=True)
    record = {"mode":mode,"status":"starting","goal_image":str(goal.resolve()),
              "goal_sha256":digest(goal),"model_image_generation_called":False,
              "kimi_target_graph_extraction_called":False,"actions_executed":0,"steps":[]}
    started=time.monotonic()
    env = install_recording(build_environment("scene_003","scene_003",out))
    try:
        obs = env.reset(0)
        initial_image = obs["image"]
        record.update(initial_image=str(initial_image.resolve()), initial_sha256=digest(initial_image))
        constructor = SceneGraphConstructor(perception_backend=Sam3PerceptionClient(endpoint=sam3_endpoint))
        initial = constructor.construct(initial_image, instruction=INSTRUCTION)
        target = constructor.construct(goal, instruction=INSTRUCTION)
        initial.to_json(out / "initial_sam3_scene_graph.json")
        target.to_json(out / "terminal_sam3_scene_graph.json")
        initial_ids = {o.id for o in initial.objects}
        target_ids = {o.id for o in target.objects}
        record.update(initial_object_ids=sorted(initial_ids),terminal_object_ids=sorted(target_ids),
                      missing_target_ids=sorted(initial_ids-target_ids),
                      extra_target_ids=sorted(target_ids-initial_ids),
                      terminal_sam3_complete=initial_ids==target_ids)
        mapping = match_book_instances(initial, env.task)
        write(out / "execution_identity_map.json",mapping)
        env.bind_graph(initial,mapping)
        model_config = dynamic_reasoning_config(settings,out / "model_audit")
        model_config = replace(model_config,max_retries=1)
        model = OpenAICompatibleLanguageModel(model_config)
        legend = [{"id":o.id,"label":o.name} for o in initial.objects]
        task = (COMMON + "Instruction: " + INSTRUCTION + "\nUse only these source IDs: "
                + json.dumps(legend,ensure_ascii=False) + "\n")
        if mode == "images":
            task += "Image 1 is the actual initial robot head-camera image. Image 2 is the supplied terminal tidy desk image. Infer actions by comparing them. No scene graph is supplied."
            images = [model_image(initial_image,out / "model_initial.jpg"),
                      model_image(goal,out / "model_terminal.jpg")]
            record["model_input"]="two_images_plus_initial_id_legend"
            record["model_image_transport"]="JPEG quality 88, max 960x720, same visual content"
        else:
            task += ("Use only these SAM3 scene graphs as scene evidence. A missing target object means its goal is unknown. INPUT:\n"
                +json.dumps({"initial":prompt_graph(initial),
                            "terminal":prompt_graph(target)},ensure_ascii=False))
            images = None
            record["model_input"]="two_sam3_scene_graphs_only"
        (out / "model_prompt.txt").write_text(task)
        record["status"]="requesting_action_plan"
        write(out / "result.json",record)
        reply = model.generate(task,images=images)[0]
        (out / "model_response.txt").write_text(reply)
        record["model_elapsed_seconds"]=round(time.monotonic()-started,3)
        plan = parse_plan(reply,initial_ids)
        write(out / "action_plan.json",plan)
        record.update(status="executing",action_count=len(plan["actions"]),uncertainties=plan.get("uncertainties",[]))
        write(out / "result.json",record)
        for index, action in enumerate(plan["actions"]):
            step={"index":index,"action":action,"status":"pending"}
            record["steps"].append(step)
            try:
                observation=env.step(request_from_action(action))
                step.update(status="executed",observed_image=str(observation["image"]),
                            object_poses=observation["object_poses"])
                record["actions_executed"]+=1
            except Exception as exc:
                step.update(status="failed",error=f"{type(exc).__name__}: {exc}")
                record.update(status="execution_failed",error=step["error"])
                break
            finally: write(out / "result.json",record)
        else:
            record["status"]="plan_executed"
        record["fixed_scene_success"] = bool(env.evaluate())
        final = env.observe()["image"]
        final_graph=constructor.construct(final,instruction=INSTRUCTION)
        final_graph.to_json(out / "final_sam3_scene_graph.json")
        final_by_id={o.id:o for o in final_graph.objects}
        target_by_id={o.id:o for o in target.objects}
        checks={}
        for oid in sorted(initial_ids | target_ids):
            a,b=final_by_id.get(oid),target_by_id.get(oid)
            if a is None or b is None or a.attributes.xy is None or b.attributes.xy is None:
                checks[oid]=None
            else:
                checks[oid]=float(np.linalg.norm(np.asarray(a.attributes.xy)-np.asarray(b.attributes.xy)))<=.06
        record["visible_goal_checks_0p06"]=checks
        record["all_initial_instances_accounted_for"] = bool(initial_ids <= target_ids)
        record["certified_visual_terminal"] = bool(initial_ids <= target_ids and checks and all(v is True for v in checks.values()))
    except Exception as exc:
        record.update(status="planning_or_setup_failed",error=f"{type(exc).__name__}: {exc}")
    finally:
        record["elapsed_seconds"]=round(time.monotonic()-started,3)
        try: env.close()
        except Exception as exc: record["close_error"]=f"{type(exc).__name__}: {exc}"
        write(out / "result.json",record)
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--goal-image",type=Path,required=True)
    parser.add_argument("--api-config",type=Path,required=True)
    parser.add_argument("--sam3-endpoint",default="http://127.0.0.1:8003")
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    settings=load_api_settings(args.api_config)
    # The image input uses the reasoning credential; no image generator is constructed.
    results=[]
    for mode in ("images","scene_graphs"):
        result=run(mode,args.output_dir,args.goal_image,settings,args.sam3_endpoint)
        results.append(result)
        write(args.output_dir / "comparison.json",{"results":results})
        print(json.dumps({"mode":mode,"status":result["status"],"actions_executed":result["actions_executed"],
                          "error":result.get("error")},ensure_ascii=False),flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
