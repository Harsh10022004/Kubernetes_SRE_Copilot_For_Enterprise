"""Compare two Ragas evaluation JSON result files and print a markdown delta table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.reporting import METRIC_KEYS


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff two Ragas eval result files")
    parser.add_argument("before", type=Path, help="Baseline eval JSON file")
    parser.add_argument("after", type=Path, help="Comparison eval JSON file")
    args = parser.parse_args()

    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))

    b_agg = before.get("aggregate") or {}
    a_agg = after.get("aggregate") or {}

    print(
        f"\n## Eval Diff: `{before.get('profile', args.before.name)}` -> "
        f"`{after.get('profile', args.after.name)}`\n"
    )
    print("| Metric | Before | After | Delta |")
    print("|--------|--------|-------|-------|")

    for key in METRIC_KEYS:
        bv = b_agg.get(key)
        av = a_agg.get(key)
        b_str = f"{bv:.3f}" if isinstance(bv, (int, float)) else "—"
        a_str = f"{av:.3f}" if isinstance(av, (int, float)) else "—"
        if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
            delta = av - bv
            d_str = f"{delta:+.3f}"
        else:
            d_str = "—"
        print(f"| {key} | {b_str} | {a_str} | {d_str} |")

    b_viol = b_agg.get("forbidden_violations", 0)
    a_viol = a_agg.get("forbidden_violations", 0)
    print(f"| forbidden_violations | {b_viol} | {a_viol} | {a_viol - b_viol:+d} |")


if __name__ == "__main__":
    main()
