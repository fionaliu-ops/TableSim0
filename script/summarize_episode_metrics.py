"""Aggregate automatic episode reports by method: python -m script.summarize_episode_metrics RUNS_DIR."""
import argparse
import json
from pathlib import Path

from high_level_planner.evaluation import aggregate_reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    paths = set()
    for path in args.paths:
        paths.update([path.resolve()] if path.is_file() else
                     (p.resolve() for p in path.rglob("evaluation_metrics.json")))
    grouped = {}
    for path in sorted(paths):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schema") != "tablesim.episode_evaluation.v1":
            parser.error(f"Unsupported episode report: {path}")
        grouped.setdefault(report["method"], []).append(report)
    if not grouped:
        parser.error("No evaluation_metrics.json reports found")
    result = {method: aggregate_reports(reports) for method, reports in grouped.items()}
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
