"""Create a compact aggregate report for the constrained-v2 comparison batch."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

def error_category(error: str | None) -> str:
    if not error:
        return ""
    e = error.lower()
    if "timed out" in e or "timeout" in e or "request budget" in e:
        return "API timeout/budget"
    if "ssl" in e or "eof" in e or "connection" in e:
        return "API network/SSL"
    if "support placement" in e or "grounded support" in e:
        return "support placement validation"
    if "no goal path" in e:
        return "no goal path"
    if "json" in e:
        return "JSON parse/format"
    return error.split(":", 1)[0]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    rows = []
    for p in sorted(args.output.glob("scene_*/comparison/comparison.json")):
        d = json.loads(p.read_text())
        row = {"scene": p.parts[-3]}
        for method, result in d.get("results", {}).items():
            api = result.get("api", {})
            row[method] = {"status": result.get("status"), "goal_reached": result.get("goal_reached", False),
                           "error": result.get("error"), "requests": api.get("request_count", 0),
                           "prompt_tokens": api.get("prompt_tokens", 0), "completion_tokens": api.get("completion_tokens", 0),
                           "elapsed_seconds": result.get("elapsed_seconds", 0)}
        rows.append(row)
    counts = {}
    for method in ("direct", "graph"):
        vals = [r.get(method, {}) for r in rows]
        counts[method] = {"scenes": len(vals), "goal_reached": sum(v.get("goal_reached", False) for v in vals),
                         "success": sum(v.get("goal_reached") is True for v in vals),
                         "errors": sum(v.get("status") == "error" for v in vals),
                         "requests": sum(v.get("requests", 0) for v in vals),
                         "prompt_tokens": sum(v.get("prompt_tokens", 0) or 0 for v in vals),
                         "completion_tokens": sum(v.get("completion_tokens", 0) or 0 for v in vals),
                         "elapsed_seconds": round(sum(v.get("elapsed_seconds", 0) or 0 for v in vals), 1),
                         "error_types": dict(Counter(error_category(v.get("error")) for v in vals if v.get("error")))}
    payload = {"scenes": rows, "aggregate": counts}
    (args.output / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    exp = {}
    first = next(iter(args.output.glob("scene_*/comparison/comparison.json")), None)
    if first:
        exp = json.loads(first.read_text()).get("experiment", {})
    lines = ["# scenes_constrained_v2 批量对比报告", "", "## 测试范围与方法", "", "测试对象为 `image_collectors/scenes_constrained_v2` 的 scene_000–scene_009，共10个场景。两条路径均调用 kimi-k3：direct 一次性生成动作链，reasoning graph 逐步扩展状态并搜索动作链。两者使用同一份初始/终态 Scene Graph、同一验证器和约束；本批次未执行真实机器人动作。", "", "输入图由每个场景的 manifest 视觉布局数据转换为兼容 SAM 的简化 Scene Graph。终态采用统一的紧凑收纳策略：保留后方显示设备等固定物，将可移动物体收拢至其附近。该终态是可重复的基准目标，不是人工逐场景标注。", "", f"运行参数：`max_states={exp.get('max_states', '见场景记录')}`，`max_actions={exp.get('max_actions', '见场景记录')}`，`beam_width={exp.get('beam_width', '见场景记录')}`，`num_branches={exp.get('num_branches', '见场景记录')}`，单次请求上限={exp.get('request_timeout_seconds', '历史记录未保存')}秒，最大输出 token={exp.get('max_output_tokens_per_call', '未记录')}。不同场景的预算以各自 manifest 为准；耗时为各场景累计值，非并行批次墙钟时间。", "", "## 汇总结果", "", "| 方法 | 目标达成 | success | invalid_plan | error | 请求数 | prompt tokens | completion tokens | 总耗时(s) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in ("direct", "graph"):
        c = counts[m]
        invalid = sum(v.get("status") == "invalid_plan" for v in [r.get(m, {}) for r in rows])
        lines.append(f"| {m} | {c['goal_reached']}/{len(rows)} | {c['success']} | {invalid} | {c['errors']} | {c['requests']} | {c['prompt_tokens']} | {c['completion_tokens']} | {c['elapsed_seconds']} |")
    lines += ["", "错误分类：", ""]
    for m in ("direct", "graph"):
        c = counts[m]
        parts = [f"`{k}` {v}个" for k, v in c["error_types"].items()]
        lines.append(f"- {m}: " + ("，".join(parts) if parts else "无异常终止（动作有效性另见 invalid_plan）"))
    lines += ["", "## 逐场景结果", "", "| 场景 | direct 状态 | direct 请求/耗时(s) | graph 状态 | graph 请求/耗时(s) |", "|---|---|---:|---|---:|"]
    for r in rows:
        d, g = r.get("direct", {}), r.get("graph", {})
        ds = d.get("status", "missing")
        gs = g.get("status", "missing")
        lines.append(f"| {r['scene']} | {ds} | {d.get('requests', 0)}/{d.get('elapsed_seconds', 0):.1f} | {gs} | {g.get('requests', 0)}/{g.get('elapsed_seconds', 0):.1f} |")
    lines += ["", "## 结论与后续", ""]
    for method in ("direct", "graph"):
        c = counts[method]
        lines.append(f"{method}：目标达成 {c['goal_reached']}/{len(rows)}，异常终止 {c['errors']} 次。失败分类见上表。")
    lines += ["", "结果只反映该批次输入、资源预算及平面验证器下的表现；接口异常、动作校验失败和搜索未找到路径需分别分析，不能据此断言真实机器人执行失败。终态来自 manifest 的模拟坐标投影，不是图像检测结果。"]

    (args.output / "analysis_zh.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
