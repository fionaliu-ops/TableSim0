"""Instrumentation at the live episode boundary; restored after every run."""
from functools import wraps
import time

from high_level_planner.evaluation import EpisodeMetrics


def evaluated_closed_loop(function):
    @wraps(function)
    def run(env, planner, instruction, **kwargs):
        # Preserve the original validation errors for malformed invocations.
        if env is None or planner is None:
            return function(env, planner, instruction, **kwargs)
        metrics = EpisodeMetrics(method=type(planner).__name__, instruction=instruction,
                                 output_dir=env.output_dir)
        metrics.context = {k: kwargs[k] for k in ("max_steps", "position_tolerance", "graph_validation", "online_rearrangement") if k in kwargs}
        metrics.context.update(getattr(env, "evaluation_context", {}))
        original_metrics = getattr(env, "_episode_metrics", None)
        env._episode_metrics = metrics
        restored = []
        status = "episode_error"

        def instrument(owner, name, wrapper):
            if owner is None:
                return
            original = getattr(owner, name, None)
            if callable(original):
                own = name in vars(owner)
                restored.append((owner, name, original, own))
                setattr(owner, name, wrapper(original))

        def timed(name):
            def wrapper(original):
                def call(*args, **kw):
                    with metrics.phase(name):
                        return original(*args, **kw)
                return call
            return wrapper

        def execution(original):
            def call(*args, **kw):
                tick = time.perf_counter()
                try:
                    result = original(*args, **kw)
                except Exception as exc:
                    from script.dual_arm_safety import ArmClearanceError, TrajectoryClearanceError
                    metrics.execution(success=False, seconds=time.perf_counter()-tick,
                        details=getattr(exc, "details", {}),
                        fatal=isinstance(exc, ArmClearanceError) and not isinstance(exc, TrajectoryClearanceError))
                    raise
                metrics.execution(success=True, seconds=time.perf_counter()-tick)
                return result
            return call

        def observation(original):
            def call(*args, **kw):
                try:
                    with metrics.phase("perception"):
                        scene = original(*args, **kw)
                except Exception as exc:
                    metrics.invalidate_observation(exc)
                    raise
                if metrics.target is not None:
                    try:
                        metrics.observe(scene, payload_released=env.held is None)
                    except Exception as exc:
                        metrics.invalidate_observation(f"metric observation: {exc}")
                return scene
            return call

        def disturbance(original):
            def call(*args, **kw):
                result = original(*args, **kw)
                metrics.disturbance(result if isinstance(result, dict) else {"event": args[0]})
                return result
            return call

        def revision(original):
            def call(*args, **kw):
                result = original(*args, **kw)
                metrics.target_revision_count += 1
                return result
            return call

        def initial_planning(original):
            def call(*args, **kw):
                with metrics.phase("initial_planning_pipeline"), metrics.phase("planning"):
                    return original(*args, **kw)
            return call

        def verified(original):
            def call(*args, **kw):
                result = original(*args, **kw)
                metrics.verify_action(kw.get("success"))
                return result
            return call

        try:
            instrument(env, "step", execution)
            instrument(env, "observe", timed("camera_capture"))
            instrument(env, "inject_event", disturbance)
            instrument(getattr(planner, "scene_updater", None), "update_scene_graph", observation)
            instrument(getattr(planner, "scene_constructor", None), "construct", timed("perception"))
            instrument(getattr(planner, "graph_planner", None), "plan", timed("planning"))
            generator = getattr(planner, "reasoning_generator", None)
            for name in ("generate", "update_reasoning_graph"):
                instrument(generator, name, timed("planning"))
            instrument(planner, "update_target_conditioned_scene", timed("planning"))
            instrument(planner, "plan_initial_image", initial_planning)
            instrument(planner, "revise_terminal_target", revision)
            instrument(planner, "record_execution_result", verified)
            instrument(kwargs.get("goal_generator"), "generate", timed("planning"))
            result = function(env, planner, instruction, **kwargs)
            status = result.status
            return result
        except Exception as exc:
            metrics.errors.append(f"episode: {type(exc).__name__}: {exc}")
            raise
        finally:
            for owner, name, original, own in reversed(restored):
                if own:
                    setattr(owner, name, original)
                else:
                    delattr(owner, name)
            env._episode_metrics = original_metrics
            report = metrics.write(status)
            print("自动评估：", {"task_success": report["task_success_int"],
                "goal_fraction": report["goal_progress"]["final_fraction"],
                "actions": report["efficiency"]["attempted_actions"],
                "total_seconds": round(report["efficiency"]["total_seconds"], 3),
                "report": str(env.output_dir / "evaluation_metrics.json")})
    return run
