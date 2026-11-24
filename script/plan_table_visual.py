"""Detect the tabletop and draw center reference axes without any model calls."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from high_level_planner.visual_planner import TableVisualPlanner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--table-corners", type=Path,
                        help="Optional JSON array of four normalized xy corner pairs")
    parser.add_argument("--output-dir", type=Path, default=Path("planner_outputs/visual_plans"))
    args = parser.parse_args()
    corners = json.loads(args.table_corners.read_text(encoding="utf-8-sig")) if args.table_corners else None
    output = args.output_dir.resolve() / uuid4().hex
    reference = TableVisualPlanner(corners=corners).save(args.image, output)
    print("status:", reference["status"])
    print("reference:", output / "table_reference.json")
    print("overlay:", output / "table_axes.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
