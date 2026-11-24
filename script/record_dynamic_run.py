"""Record actual control-step camera frames without altering dynamic planning."""
import json
from pathlib import Path

import imageio.v2 as imageio

from script import dynamic_sim


def main():
    original_build = dynamic_sim.build_environment

    def build(*args, **kwargs):
        env = original_build(*args, **kwargs)
        original_reset, original_close = env.reset, env.close
        writers = {}
        frames = 0
        ticks = 0
        stride = 1
        telemetry = None

        def capture():
            nonlocal frames
            env.task._update_render()
            env.task.cameras.update_picture()
            rgb = env.task.cameras.get_rgb()
            for name in ("head_camera", "front_camera"):
                if name not in rgb:
                    continue
                if name not in writers:
                    writers[name] = imageio.get_writer(
                        str(env.output_dir / f"operation_{name}.mp4"),
                        fps=1.0 / (env.task.scene.get_timestep() * stride),
                        codec="libx264", quality=8, macro_block_size=1)
                writers[name].append_data(rgb[name]["rgb"])
            telemetry.write(json.dumps({"frame": frames, "control_tick": ticks,
                "action_index": env.step_index,
                "objects": {k: v.get_pose().p.tolist()
                            for k, v in env.task.objects.items()}}) + "\n")
            frames += 1

        def reset(seed=0):
            nonlocal stride, telemetry
            obs = original_reset(seed)
            stride = max(1, round(1.0 / (25 * env.task.scene.get_timestep())))
            telemetry = (env.output_dir / "video_frames.jsonl").open("w")
            original_sync = env.task._sync_attached_actor

            def sync():
                nonlocal ticks
                original_sync()
                ticks += 1
                if ticks % stride == 0:
                    capture()

            env.task._sync_attached_actor = sync
            capture()
            return obs

        def close():
            try:
                if telemetry is not None:
                    capture()
                    telemetry.close()
                for writer in writers.values():
                    writer.close()
                (env.output_dir / "video_manifest.json").write_text(json.dumps({
                    "frames": frames, "control_ticks": ticks,
                    "recording": "actual simulation control-step frames",
                    "has_robot_control_steps": ticks > 0,
                    "cameras": list(writers),
                    "ai_video_generated": False,
                    "planning_and_execution_modified": False}, indent=2) + "\n")
            finally:
                original_close()

        env.reset, env.close = reset, close
        return env

    dynamic_sim.build_environment = build
    return dynamic_sim.main()


if __name__ == "__main__":
    raise SystemExit(main())
