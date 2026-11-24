"""Compare one-shot planning and graph search against one frozen graph contract."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from high_level_planner.planner.cost import CostFunction
from high_level_planner.planner.ucs import uniform_cost_search
from high_level_planner.reasoning_graph import APIConfig, OpenAICompatibleLanguageModel
from high_level_planner.reasoning_graph.generator import ReasoningGraphGenerator
from high_level_planner.reasoning_graph.graph import ReasoningGraph
from high_level_planner.reasoning_graph.object_attribute_query import query_object_attributes
from high_level_planner.reasoning_graph.operation_context import OperationContext
from high_level_planner.reasoning_graph.operations import ValidateActions, ScoreActions, LanguageModelActionProposer
from high_level_planner.reasoning_graph.parser import ReasoningGraphParser
from high_level_planner.reasoning_graph.state import AtomicAction, SceneStatePayload, SimpleSceneGraph
from high_level_planner.reasoning_graph.target_conditioned import TargetGraphEvaluator, TargetGraphPrompter
from high_level_planner.reasoning_graph.target_identity import merge_target_goals
from high_level_planner.reasoning_graph.thought import Thought
from high_level_planner.reasoning_graph.transition import SceneGraphTransitionModel
from high_level_planner.scene_graph.graph import SceneGraph
from high_level_planner.ucs_path_output import write_ucs_path_output
from script.goal_config import load_api_settings

ROOT = Path(__file__).resolve().parent
INSTRUCTION = (
    "Reproduce the supplied confirmed terminal Scene Graph. Preserve object identities. "
    "Use only the supplied graph facts and constraints; do not reinterpret images or invent goals."
)
COMMON_RULES = """The initial and terminal Scene Graphs are the sole task evidence. No images are supplied.
The confirmed_goal_constraints are authoritative, including support, articulation, visibility,
relative relations and position tolerance. Do not add tidiness rules or unprovided orientation goals.
Use one atomic skill per action. Clear incoming on/blocks/occludes/overlaps blockers before
moving, opening or closing an object. Close an open book before stacking it. Preserve fixed objects.
Each action must have action_type, source_object_id, target, instruction, parameters.
For a table move use target="table", parameters.pose and parameters.placement_target with
kind="pose_constraint", reference="table", parameters={"pose": <the same pose>}.
Use supplied placement candidates for temporary buffers. Target table poses come from the terminal graph.
For stack or supported place, copy the complete parameters (including relative_pose and
placement_target) from the matching grounded candidate. Never strip support, frame or candidate
fields. A non-stackable object requires place with support_placement=true, not stack.
For a direct plan, future support_action_templates supply these same parameter shapes, but are
NOT READY actions: first clear blockers and move the support to its terminal pose.
An occupied terminal table pose is not an executable move. Relocate the occupying object first,
using a supplied buffer if needed. Remove objects ON a support before moving that support,
then restore them afterwards. Cover every unsatisfied object; do not truncate the task to fit.
For open/close use target=null; for reorient provide parameters.target_yaw_degrees.
Do not repeat already satisfied goals. Image-normalized graph coordinates are not robot coordinates.
Return strict JSON only. Do not provide hidden reasoning.
"""


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def clean_scene(scene):
    scene = deepcopy(scene)
    for key in ("planning_context", "task_goals", "task_policy", "goal_rules"):
        scene.metadata.pop(key, None)
    return scene


def prepared_initial(initial, evaluator):
    # Exactly the projection used by ReasoningGraphGenerator.generate(json).
    state = ReasoningGraphGenerator._project_scene_with_context(deepcopy(initial), {}, use_regions=False)
    state.instruction = INSTRUCTION
    evaluator.prepare(state, INSTRUCTION)
    return state


def make_contract(initial, target, tolerance):
    evaluator = TargetGraphEvaluator(initial, target, tolerance)
    return {
        "input_mode": "initial_terminal_scene_graphs", "instruction": INSTRUCTION,
        "position_tolerance": tolerance,
        "initial_graph_sha256": digest(initial.to_dict()),
        "terminal_graph_sha256": digest(target.to_dict()),
        "goal_predicates": [g.to_dict() for g in evaluator.goals],
        "target_identity_and_relation_checks": evaluator.evaluate(target, INSTRUCTION).goal_checks,
        "target_scene_graph": SimpleSceneGraph.from_scene_graph(target).to_dict(),
        "physical_execution_verified": False,
        "unprovided_properties": "No image interpretation or additional orientation target is inferred.",
    }


class ComparisonPrompter(TargetGraphPrompter):
    """Same evidence and action rules; only the requested planning strategy differs."""

    def __init__(self, initial, target, contract, max_actions):
        super().__init__(initial, target)
        self.contract = contract
        self.max_actions = max_actions
        self.placement_provider = TargetGraphEvaluator(
            initial, target, contract["position_tolerance"]
        ).goal_compiler.placement_provider

    def payload(self, scene_graph, goal_report):
        original = super().generation_prompt(scene_graph, INSTRUCTION, goal_report, num_branches=3)
        data = json.loads(original.rsplit("INPUT:\n", 1)[1])
        data["confirmed_goal_constraints"] = self.contract
        data["max_plan_actions"] = self.max_actions
        data["support_action_templates"] = self.support_templates()
        return data

    def support_templates(self):
        # Future-state templates are evidence for the one-shot plan, not an
        # exemption from the current-state grounding/feasibility gate.
        from high_level_planner.reasoning_graph.geometry import pose
        from high_level_planner.reasoning_graph.placement import PlacementTarget
        templates = []
        for relation in self.target.relations:
            if relation.relation != "on":
                continue
            obj = self.target.get_object(relation.source)
            destination = pose(self.target, relation.source)
            if obj is None or destination is None:
                continue
            parameters = {"relative_pose": destination}
            action_type = "stack" if obj.attributes.stackable else "place"
            if action_type == "place":
                parameters["support_placement"] = True
            parameters["placement_target"] = PlacementTarget(
                "support_surface", relation.target, deepcopy(parameters)
            ).to_dict()
            templates.append({"action_type": action_type, "source_object_id": obj.id,
                              "target": relation.target, "parameters": parameters,
                              "ready": False, "requires": "clear blockers and position support first"})
        return templates

    def generation_prompt(self, scene_graph, instruction, goal_report, *, num_branches, scene_state=None):
        strategy = (
            f"Return a JSON array of at most {num_branches} READY candidate actions for the current state. "
            "They are alternative next actions, not consecutive steps. Return only one when only one "
            "action advances the blocker chain. Do not output the whole plan.\n"
        )
        return COMMON_RULES + strategy + "INPUT:\n" + json.dumps(self.payload(scene_graph, goal_report), ensure_ascii=False)

    def direct_prompt(self, scene_graph, goal_report):
        strategy = (
            f"Return a JSON array containing one complete ordered plan, at most {self.max_actions} actions. "
            "These are consecutive steps. Check prerequisites against the predicted result of preceding "
            "steps. You receive no intermediate tool feedback or repair calls.\n"
        )
        return COMMON_RULES + strategy + "INPUT:\n" + json.dumps(self.payload(scene_graph, goal_report), ensure_ascii=False)


def replay_plan(initial, target, actions, tolerance):
    """Shared post-hoc verifier; rejects at the first invalid step, never repairs a plan."""
    evaluator = TargetGraphEvaluator(initial, target, tolerance)
    state = prepared_initial(initial, evaluator)
    transition = SceneGraphTransitionModel()
    prompter = ComparisonPrompter(initial, target, make_contract(initial, target, tolerance), max(1, len(actions)))
    context = OperationContext(
        instruction=INSTRUCTION, proposer=None, prompter=None, parser=None, llm=None,
        attribute_query=query_object_attributes, transition_model=transition,
        goal_evaluator=evaluator, reasoning_graph=ReasoningGraph(), source="language_model",
    )
    states = [SimpleSceneGraph.from_scene_graph(state).to_dict()]
    steps, total_cost, failed = [], 0.0, False
    for i, raw in enumerate(actions, 1):
        step = {"step": i, "action": raw, "status": "not_replayed_after_failure"}
        steps.append(step)
        if failed:
            continue
        try:
            # Replay the saved answer through the SAME proposal-grounding gate as graph search.
            # This callback returns saved JSON only; it is not a model call or a repair.
            saved_answer = SimpleNamespace(generate=lambda *a, **kw: [json.dumps([raw])])
            action = LanguageModelActionProposer(saved_answer, prompter=prompter).propose(
                state, INSTRUCTION, evaluator.evaluate(state, INSTRUCTION), num_branches=1
            )[0]
            thought = Thought(
                f"replay_{i}", SimpleSceneGraph.from_scene_graph(state),
                OperationContext.symbolic_snapshot(state), candidate_action=action,
                scene_state=SceneStatePayload("json", state),
                goal_report=evaluator.evaluate(state, INSTRUCTION),
            )
            validated = ValidateActions(keep_invalid=True).execute([thought], context)[0]
            step["validation_errors"] = validated.metadata["validation_errors"]
            step["feasibility"] = validated.metadata.get("feasibility")
            if not validated.valid:
                raise ValueError("; ".join(step["validation_errors"]))
            scored = ScoreActions().execute([validated], context)
            if not scored:
                raise ValueError("Shared scorer could not predict this transition")
            costs = scored[0].metadata["cost_metadata"]
            cost = CostFunction().edge_cost(SimpleNamespace(metadata=costs, cost=1.0))
            successor = transition.apply(state, action)
            verification = successor.metadata.get("reasoning_transition", {})
            if not verification.get("skill_verification", {}).get("satisfied", False):
                raise ValueError("Skill local goal was not satisfied")
            state = successor
            total_cost += cost
            states.append(SimpleSceneGraph.from_scene_graph(state).to_dict())
            step.update(status="passed", cost=cost, cost_components=costs,
                        transition_verification=verification,
                        goal_report=evaluator.evaluate(state, INSTRUCTION).to_dict())
        except (ValueError, KeyError, TypeError) as exc:
            failed = True
            step.update(status="failed", error=str(exc))
    report = evaluator.evaluate(state, INSTRUCTION).to_dict()
    return {
        "status": "invalid_plan" if failed else "passed" if report["terminal"] else "goal_not_reached",
        "goal_reached": not failed and report["terminal"],
        "action_count": len(actions), "valid_prefix_length": len(states) - 1,
        "total_cost": None if failed else total_cost, "valid_prefix_cost": total_cost,
        "steps": steps, "states": states, "final_goal_report": report,
        "physical_execution_verified": False,
    }


def raw_action(edge):
    return AtomicAction(edge.action_type, edge.object_target["source"], edge.object_target.get("target"),
                        instruction=edge.instruction, parameters=edge.parameters, skill=edge.skill).to_dict()


def audit_summary(directory):
    audits = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("audit_*.json"))]
    successful = [a for a in audits if a.get("complete") and a.get("http_status") == 200]
    return {
        "request_count": len(audits), "successful_requests": len(successful),
        "all_response_models_match": bool(successful) and all(a.get("response_model") == "kimi-k3" for a in successful),
        "prompt_tokens": sum(a.get("usage", {}).get("prompt_tokens", 0) or 0 for a in audits),
        "completion_tokens": sum(a.get("usage", {}).get("completion_tokens", 0) or 0 for a in audits),
        "api_seconds": sum(a.get("elapsed_seconds", 0) for a in audits),
    }


def run_method(method, output, initial, target, contract, config, args):
    output.mkdir(parents=True, exist_ok=True)
    provider = OpenAICompatibleLanguageModel(replace(config, audit_dir=str(output / "model_audit")))
    prompter = ComparisonPrompter(initial, target, contract, args.max_actions)
    evaluator = TargetGraphEvaluator(initial, target, args.position_tolerance)
    started = time.monotonic()
    result = {"method": method, "contract_sha256": digest(contract), "model": "kimi-k3",
              "input_mode": "initial_terminal_scene_graphs", "rule_fallback_used": False}
    try:
        if method == "direct":
            state = prepared_initial(initial, evaluator)
            prompt = prompter.direct_prompt(state, evaluator.evaluate(state, INSTRUCTION))
            (output / "prompt.txt").write_text(prompt, encoding="utf-8")
            answers = provider.generate(prompt, num_responses=1)
            write_json(output / "raw_answers.json", answers)
            if len(answers) != 1:
                raise ValueError("Direct baseline requires exactly one answer")
            actions = ReasoningGraphParser._json_value(answers[0])
            if not isinstance(actions, list) or any(not isinstance(a, dict) for a in actions):
                raise ValueError("Expected a JSON array of action objects")
            write_json(output / "actions.json", actions)
            if len(actions) > args.max_actions:
                raise ValueError("Direct plan exceeded the common action limit")
        else:
            generator = ReasoningGraphGenerator(
                model=provider, backend="api", goal_evaluator=evaluator, prompter=prompter,
                max_depth=args.max_actions, max_states=args.max_states,
                beam_width=args.beam_width, num_branches=args.num_branches,
                fallback_on_model_error=False, state_representation="json",
            )
            graph = generator.generate(deepcopy(initial), INSTRUCTION)
            merge_target_goals(graph, target, evaluator, INSTRUCTION)
            graph.metadata.update(reasoning_mode="initial_terminal_graphs", contract_sha256=digest(contract))
            graph.to_json(output / "reasoning_graph.json")
            search = uniform_cost_search(graph)
            write_ucs_path_output(graph, search, output)
            result.update(node_count=len(graph.nodes), edge_count=len(graph.edges),
                          goal_count=len(graph.goal_ids),
                          ucs_cost=search.total_cost if search.goal_state else None,
                          state_budget_exhausted=graph.metadata.get("state_budget_exhausted", False))
            actions = [raw_action(e) for e in search.path]
            write_json(output / "actions.json", actions)
            if search.goal_state is None:
                raise ValueError("No goal path found within the graph search budget")
        replay = replay_plan(initial, target, actions, args.position_tolerance)
        write_json(output / "replay.json", replay)
        for i, state in enumerate(replay["states"]):
            write_json(output / "states" / f"S{i:02d}.json", state)
        result.update({k: v for k, v in replay.items() if k not in {"states", "steps"}})
        if method == "graph" and replay["total_cost"] is not None:
            result["replay_cost_matches_ucs"] = abs(replay["total_cost"] - result["ucs_cost"]) < 1e-8
            if not result["replay_cost_matches_ucs"]:
                result.update(status="verification_mismatch", goal_reached=False)
    except Exception as exc:
        # Preserve failures rather than repairing or replacing the model's answer.
        message = str(exc)
        if config.resolved_token:
            message = message.replace(config.resolved_token, "[REDACTED]")
        result.update(status="error", error=f"{type(exc).__name__}: {message}", goal_reached=False)
    result.update(elapsed_seconds=round(time.monotonic() - started, 3),
                  api=audit_summary(output / "model_audit"), physical_execution_verified=False)
    if result.get("goal_reached") and not result["api"]["all_response_models_match"]:
        result.update(status="model_identity_unverified", goal_reached=False)
    write_json(output / "summary.json", result)
    return result


LABELS = {"cup_sam3": "杯子", "book_sam3": "深蓝书", "book_sam3_2": "浅蓝书",
          "keyboard_sam3": "键盘", "mouse_sam3": "鼠标", "phone_sam3": "手机", "stapler_sam3": "订书机"}


def action_label(action):
    verbs = {"move": "移动", "place": "放置", "stack": "叠放", "close": "合上", "reorient": "旋转"}
    text = verbs.get(action.get("action_type"), str(action.get("action_type"))) + " " + LABELS.get(action.get("source_object_id"), str(action.get("source_object_id")))
    if action.get("target"):
        text += " → " + LABELS.get(action["target"], action["target"])
    pose = action.get("parameters", {}).get("pose") or action.get("parameters", {}).get("relative_pose")
    if pose:
        text += " " + str(pose.get("xy"))
    return text


def write_comparison(output, manifest, results):
    write_json(output / "comparison.json", {"experiment": manifest, "results": results})
    lines = ["# 同目标 Scene Graph 规划对比", "", "输入是初态和终态 scene graph；两组均配置为调用外部 kimi-k3，不输入图片。实际调用状态见下表。", "",
             "一次完整计划 vs 逐状态候选生成＋graph 搜索。共用约束、技能、校验、状态转移和代价。",
             "调用次数不相同；这是两种完整流程的比较，不是等 token 预算实验。未验证物理执行。", "",
             "目标、输入哈希及模型参数见 manifest.json、inputs/confirmed_goal_constraints.json。", "",
             "| 指标 | 直接生成 | Graph 搜索 |", "|---|---|---|"]
    pending = any(r.get("status") in {"prepared_not_run", "missing_credentials"} for r in results.values())
    if pending:
        lines[2] += " 当前尚未调用模型，没有生成真实实验动作或结论；需配置 API 凭据后运行。"
    def value(method, key):
        result = results.get(method, {})
        for part in key.split("."):
            result = result.get(part, {}) if isinstance(result, dict) else None
        return "—" if result is None or result == {} else str(result)
    for label, key in [("状态", "status"), ("目标通过", "goal_reached"), ("动作数", "action_count"),
                       ("总代价", "total_cost"), ("有效前缀长度", "valid_prefix_length"),
                       ("API 请求数", "api.request_count"), ("输入 token", "api.prompt_tokens"),
                       ("输出 token", "api.completion_tokens"), ("总耗时（秒）", "elapsed_seconds")]:
        lines.append(f"| {label} | {value('direct', key)} | {value('graph', key)} |")
    plans = {}
    for method in ("direct", "graph"):
        path = output / method / "actions.json"
        plans[method] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    lines += ["", "| 步骤 | 直接生成原始顺序 | Graph UCS 顺序 |", "|---|---|---|"]
    for i in range(max(map(len, plans.values()), default=0)):
        cells = [action_label(plans[m][i]) if i < len(plans[m]) else "—" for m in ("direct", "graph")]
        lines.append(f"| {i+1} | {cells[0]} | {cells[1]} |")
    lines += ["", "| 目标检查 | 直接生成 | Graph 搜索 |", "|---|---|---|"]
    checks = {m: results.get(m, {}).get("final_goal_report", {}).get("goal_checks", {}) for m in ("direct", "graph")}
    for key in sorted(set(checks["direct"]) | set(checks["graph"])):
        lines.append(f"| {key} | {checks['direct'].get(key, '—')} | {checks['graph'].get(key, '—')} |")
    for method, result in results.items():
        if result.get("error"):
            lines += ["", f"{method}: {result['error']}"]
        path = output / method / "replay.json"
        if path.exists():
            replay = json.loads(path.read_text(encoding="utf-8"))
            for step in replay["steps"]:
                if step["status"] == "failed":
                    lines += ["", f"{method} 第 {step['step']} 步校验失败：{step.get('error', '')}"]
    lines += ["", "每组 actions.json 是待验证动作；replay.json 给出每步校验、失败位置和未验证物理条件。",
              "states/ 是有效前缀的预测状态，不是实际执行观测。失败后不跳步、不补动作、不修复答案。",
              "model_audit/ 保存实际文本请求、可见回答、返回模型、用量和结束状态，不保存认证头。"]
    text = "\n".join(lines) + "\n"
    (output / "comparison.md").write_text(text, encoding="utf-8")
    # Simple local report: no remote scripts, images or model interpretation.
    body, table_open = [], False
    for line in lines:
        if line.startswith("|---"):
            continue
        if line.startswith("|"):
            if not table_open:
                body.append("<table>")
                table_open = True
            body.append("<tr>" + "".join("<td>" + html.escape(c.strip()) + "</td>" for c in line.strip("|").split("|")) + "</tr>")
        else:
            if table_open:
                body.append("</table>")
                table_open = False
            body.append("<p>" + html.escape(line.lstrip("# ")) + "</p>")
    if table_open:
        body.append("</table>")
    links = '<p><a href="inputs/confirmed_goal_constraints.json">共同目标约束</a> · <a href="manifest.json">实验参数</a>'
    for method in results:
        for name, label in (("actions.json", "动作"), ("replay.json", "逐步校验"), ("summary.json", "统计"), ("reasoning_graph.json", "完整搜索图")):
            if (output / method / name).exists():
                links += f' · <a href="{method}/{name}">{method} {label}</a>'
    links += '</p>'
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Kimi 规划对比</title>'
        '<style>body{max-width:1200px;margin:30px auto;font:16px/1.6 system-ui;padding:20px;color:#183047}'
        'table{border-collapse:collapse;width:100%;margin:20px 0}td{border:1px solid #ccd5df;padding:9px;overflow-wrap:anywhere}'
        'tr:first-child{background:#eaf2f8;font-weight:bold}</style>' + links + "\n".join(body) + "</html>", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, default=ROOT / "inputs/initial_scene_graph.json")
    parser.add_argument("--terminal", type=Path, default=ROOT / "inputs/terminal_scene_graph.json")
    parser.add_argument("--config", type=Path, help="Local config with reasoning section; token is never exported")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--position-tolerance", type=float, default=0.04)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--max-states", type=int, default=128)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--num-branches", type=int, choices=(3, 4, 5), default=3)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=None, help="Per API request timeout in seconds")
    parser.add_argument("--prepare-only", action="store_true", help="Export shared inputs/prompts without any API request")
    args = parser.parse_args(argv)
    if min(args.max_actions, args.max_states, args.beam_width, args.max_output_tokens) < 1:
        parser.error("Budgets must be positive")
    settings = load_api_settings(args.config).get("reasoning", {})
    if settings.get("model", "kimi-k3") != "kimi-k3":
        parser.error("This controlled comparison requires kimi-k3 for both methods")
    config = replace(APIConfig.from_env(model="kimi-k3", base_url=settings.get("url")),
                     api_token=settings.get("api_token"), api_mode="chat_completions",
                     timeout=args.timeout if args.timeout is not None else settings.get("timeout", 600), temperature=settings.get("temperature", 1.0),
                     reasoning_effort=settings.get("reasoning_effort", "low"), stream=True,
                     max_output_tokens=args.max_output_tokens)
    output = args.output_dir or ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    initial, target = (clean_scene(SceneGraph.from_json(p)) for p in (args.initial, args.terminal))
    contract = make_contract(initial, target, args.position_tolerance)
    write_json(output / "inputs/initial_scene_graph.json", initial.to_dict())
    write_json(output / "inputs/terminal_scene_graph.json", target.to_dict())
    write_json(output / "inputs/confirmed_goal_constraints.json", contract)
    evaluator = TargetGraphEvaluator(initial, target, args.position_tolerance)
    state = prepared_initial(initial, evaluator)
    prompter = ComparisonPrompter(initial, target, contract, args.max_actions)
    report = evaluator.evaluate(state, INSTRUCTION)
    for method, prompt in (("direct", prompter.direct_prompt(state, report)),
                           ("graph", prompter.generation_prompt(state, INSTRUCTION, report, num_branches=args.num_branches))):
        (output / f"{method}_initial_prompt.txt").write_text(prompt, encoding="utf-8")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT.parent)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "input_mode": "initial_terminal_scene_graphs",
        "contract_sha256": digest(contract), "git_revision": revision.stdout.strip(),
        "model": config.model, "endpoint": config.base_url, "temperature": config.temperature,
        "reasoning_effort": config.reasoning_effort, "max_output_tokens_per_call": config.max_output_tokens,
        "request_timeout_seconds": config.timeout,
        "max_actions": args.max_actions, "max_states": args.max_states, "beam_width": args.beam_width,
        "num_branches": args.num_branches, "images_supplied": False, "repair_enabled": False,
        "shared_rules_sha256": digest(COMMON_RULES), "physical_execution_verified": False,
        "comparison_scope": "same model/evidence/constraints/validator; different strategy and total API budget",
        "source_sha256": {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), *sorted((ROOT.parent / "high_level_planner/reasoning_graph").glob("*.py")),
                                    *sorted((ROOT.parent / "high_level_planner/planner").glob("*.py"))]},
    }
    write_json(output / "manifest.json", manifest)
    if args.prepare_only or not config.resolved_token:
        status = "prepared_not_run" if args.prepare_only else "missing_credentials"
        results = {m: {"status": status, "api_called": False, "goal_reached": None} for m in ("direct", "graph")}
        write_comparison(output, manifest, results)
        print(json.dumps({"output": str(output), "status": status}, ensure_ascii=False))
        return 0 if args.prepare_only else 2
    results = {}
    for method in ("direct", "graph"):
        print(f"Running {method} with external kimi-k3...", flush=True)
        results[method] = run_method(method, output / method, initial, target, contract, config, args)
        write_comparison(output, manifest, results)
        print(json.dumps({"method": method, "status": results[method]["status"], "output": str(output)}, ensure_ascii=False), flush=True)
    return 0 if all(r.get("goal_reached") for r in results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
