"""Shared-executor callback runner, keeping actual success separate from scores."""
import time

from high_level_planner.evaluation import EpisodeMetrics


def run_tidiness_episode(planner, *, observe, execute, evaluate, instruction="整理桌面", max_steps=50,
                         output_dir=None, target_scene=None, position_tolerance=.02):
    """observe()->TidyingState; execute(VLARequest)->{success:bool, details:dict}.

    evaluate(TidyingState)->{success:bool,...} must be an independent evaluator
    of the current physical observation. Never pass planner scores as success.
    Simulator truth may be used by evaluate, but must not enter observe/planner.
    """
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be positive")
    planner.reset()
    started, records = time.perf_counter(), []
    evaluation_metrics = EpisodeMetrics(method=type(planner).__name__, instruction=instruction,
                                        output_dir=output_dir)
    evaluation_metrics.context = {"max_steps": max_steps, "position_tolerance": position_tolerance}
    status, external = "step_limit", {"success": False}
    try:
        for index in range(max_steps + 1):
            with evaluation_metrics.phase("perception"):
                observed = observe()
            external = evaluate(observed)
            if not isinstance(external.get("success"), bool):
                raise ValueError("External evaluator must return a boolean success")
            if index == 0 and target_scene is not None:
                evaluation_metrics.configure(observed.scene, target_scene, position_tolerance)
            evaluation_metrics.observe(observed.scene, evaluation=external,
                                       payload_released=external.get("payload_released"))
            if external["success"]:
                status = "external_success"
                break
            if index == max_steps:
                break
            with evaluation_metrics.phase("planning"):
                output = planner.update_from_scene(observed.scene, instruction, visual=observed.visual)
            record = {"step": index, "decision": output.to_dict(), "external_before": external}
            records.append(record)
            if output.status != "action":
                status = "scorer_false_positive" if output.status == "done" else "planner_blocked"
                break
            execution_start = time.perf_counter()
            try:
                feedback = execute(output.vla_request)
            except Exception as exc:
                evaluation_metrics.execution(success=False, seconds=time.perf_counter()-execution_start,
                                             details={"error_type": type(exc).__name__})
                raise
            if not isinstance(feedback.get("success"), bool):
                raise ValueError("Executor must report boolean success, not a predicted outcome")
            record.update(execution_seconds=time.perf_counter() - execution_start, feedback=feedback)
            evaluation_metrics.execution(success=feedback["success"], seconds=record["execution_seconds"],
                details=feedback.get("details"), fatal=feedback.get("fatal", False))
            if feedback.get("disturbance") is not None:
                evaluation_metrics.disturbance(feedback["disturbance"])
            planner.record_execution_result(output.next_subtask, success=feedback["success"],
                                            details=feedback.get("details", {}))
            if feedback.get("fatal", False):
                status = "safety_stop"
                break
    except Exception as exc:
        status = "episode_error"
        evaluation_metrics.errors.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        evaluation_report = evaluation_metrics.write(status)
    return {"status": status, "external_evaluation": external, "steps": records,
            "metrics": planner.metrics, "evaluation_metrics": evaluation_report,
            "total_seconds": time.perf_counter() - started}
