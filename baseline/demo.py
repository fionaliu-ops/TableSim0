"""Run one Direct VLM decision: python -m baseline.demo --help."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

from high_level_planner.reasoning_graph.language_models.openai_compatible import APIConfig
from .direct_vlm_planner import DirectVLMPlanner


def main():
    parser = argparse.ArgumentParser(description="Direct VLM Planner: one visual decision, no execution")
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", default="整理桌面")
    parser.add_argument("--target-image")
    parser.add_argument("--object-catalog", help="JSON mapping object IDs to visible descriptions")
    parser.add_argument("--config", help="Existing API JSON configuration; uses reasoning section")
    parser.add_argument("--model")
    parser.add_argument("--api-base")
    parser.add_argument("--output-dir", default="planner_outputs/baseline/direct_vlm")
    args = parser.parse_args()
    settings = {}
    if args.config:
        raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
        settings = dict(raw.get("reasoning", {}))
        if "url" in settings:
            settings["base_url"] = settings.pop("url")
    defaults = APIConfig.from_env()
    allowed = {f.name for f in fields(APIConfig)}
    config = {f.name: getattr(defaults, f.name) for f in fields(APIConfig)}
    config.update({k: v for k, v in settings.items() if k in allowed})
    if args.model:
        config["model"] = args.model
    if args.api_base:
        config["base_url"] = args.api_base
    catalog = json.loads(Path(args.object_catalog).read_text(encoding="utf-8")) if args.object_catalog else None
    planner = DirectVLMPlanner(api_config=APIConfig(**config), target_image=args.target_image,
                               object_catalog=catalog)
    result = planner.plan_initial_image(args.image, args.instruction)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "planner_output.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "vla_request.json").write_text(
        json.dumps(result.vla_request.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{result.status}: {output / 'planner_output.json'}")


if __name__ == "__main__":
    main()
