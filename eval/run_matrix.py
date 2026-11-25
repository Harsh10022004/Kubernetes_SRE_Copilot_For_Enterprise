"""Run the progressive RAG evaluation matrix across profiles and summarize results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.profiles import PROFILES
from eval.run_ragas import run_profile

DEFAULT_PROGRESSION = [
    "naive",
    "sparse_only",
    "hybrid",
    "hybrid+rerank",
    "hybrid+rerank+hyde",
    "hybrid+rerank+crag",
    "hybrid+rerank+self_rag",
    "hybrid+rerank+hyde+crag",
    "all",
]


def _find_latest_profile_result(profile: str, results_dir: Path) -> Path | None:
    if not results_dir.exists():
        return None
    matches = sorted(
        results_dir.glob(f"*_{profile}.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _fmt_metric(val: object) -> str:
    return f"{val:.3f}" if isinstance(val, (int, float)) else "—"


def _print_matrix_summary(summaries: list[dict]) -> None:
    print("\n## Progressive Evaluation Matrix Summary\n")
    print(
        "| profile | evaluated | skipped | faith | ctx_prec | ctx_recall | ans_rel | violations |"
    )
    print(
        "|---------|-----------|---------|-------|----------|------------|---------|------------|"
    )
    for item in summaries:
        agg = item.get("aggregate") or {}
        print(
            f"| {item['profile']} | {len(item.get('rows') or [])} | {len(item.get('skipped') or [])} | "
            f"{_fmt_metric(agg.get('faithfulness'))} | "
            f"{_fmt_metric(agg.get('context_precision'))} | "
            f"{_fmt_metric(agg.get('context_recall'))} | "
            f"{_fmt_metric(agg.get('answer_relevancy'))} | "
            f"{agg.get('forbidden_violations', 0)} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run progressive RAG evaluation profiles across the golden dataset"
    )
    parser.add_argument(
        "--profiles",
        default=None,
        help="Comma-separated list of profiles to run (default: all 9 progressive profiles)",
    )
    parser.add_argument(
        "--scorer",
        default="local",
        choices=["local", "ragas"],
        help="Scoring backend (default: local)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of golden questions per profile",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Delay in seconds between questions (default: 1.5)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip any profile that already has a result file in eval/results/",
    )
    args = parser.parse_args()

    if args.profiles:
        profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    else:
        profiles = list(DEFAULT_PROGRESSION)

    invalid = [p for p in profiles if p not in PROFILES]
    if invalid:
        print(f"Unknown profiles: {invalid}. Valid: {list(PROFILES.keys())}", file=sys.stderr)
        sys.exit(1)

    results_dir = Path("eval/results")
    summaries: list[dict] = []

    for profile in profiles:
        if args.resume:
            existing = _find_latest_profile_result(profile, results_dir)
            if existing is not None:
                print(f"[resume] Skipping {profile}; using existing {existing}")
                summaries.append(json.loads(existing.read_text(encoding="utf-8")))
                continue

        payload, _ = run_profile(
            profile=profile,
            scorer=args.scorer,
            limit=args.limit,
            delay=args.delay,
        )
        summaries.append(payload)

    _print_matrix_summary(summaries)


if __name__ == "__main__":
    main()
