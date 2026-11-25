from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

from eval.invokers import ServiceInvoker, SkippedIntent
from eval.local_scorer import score_rows as score_local
from eval.post_checks import forbidden_keywords_check, keyword_recall_check, source_overlap
from eval.profiles import PROFILES
from eval.ragas_adapter import run as run_ragas
from eval.reporting import aggregate, print_table
from eval.schema import load_goldens


def run_profile(
    profile: str,
    questions: str = "eval/seed_questions.yaml",
    feature_filter: str | None = None,
    mode: str = "service",
    output: str | None = None,
    scorer: str = "local",
    limit: int | None = None,
    delay: float = 0.0,
) -> tuple[dict, Path]:
    """Execute one profile over golden questions, score results, and write JSON."""
    flags = PROFILES[profile]
    goldens = load_goldens(questions)

    if feature_filter:
        goldens = [
            g
            for g in goldens
            if g.demonstrates_feature in (feature_filter, "baseline")
        ]
    if limit is not None and limit > 0:
        goldens = goldens[:limit]

    if mode == "service":
        invoker = ServiceInvoker()
    else:
        raise ValueError("API mode not yet implemented (Phase B).")

    rows: list[dict] = []
    skipped: list[dict] = []

    for idx, g in enumerate(goldens):
        if idx > 0 and delay > 0:
            time.sleep(delay)
        try:
            resp, chunks = invoker.invoke(g.question, flags, g.intent)
        except SkippedIntent as e:
            skipped.append({"id": g.id, "reason": str(e)})
            continue
        except Exception as e:
            skipped.append({"id": g.id, "reason": f"error: {e}"})
            continue

        rows.append(
            {
                "id": g.id,
                "demonstrates_feature": g.demonstrates_feature,
                "intent": g.intent,
                "question": g.question,
                "answer": resp.answer,
                "contexts": [c.text for c in chunks],
                "chunk_sources": [c.source for c in chunks],
                "ground_truth": ", ".join(g.golden_answer_keywords),
                "golden_answer_keywords": g.golden_answer_keywords,
                "actual_sources": resp.sources,
                "golden_sources": g.golden_sources,
                "forbidden_keywords": g.forbidden_keywords,
            }
        )

    if rows:
        metrics = score_local(rows) if scorer == "local" else run_ragas(rows)
    else:
        metrics = []

    for row, m in zip(rows, metrics):
        row["ragas_metrics"] = m
        row["forbidden_check"] = forbidden_keywords_check(
            row["answer"], row["forbidden_keywords"]
        )
        row["source_overlap"] = source_overlap(
            row["actual_sources"], row["golden_sources"]
        )
        row["keyword_check"] = keyword_recall_check(
            row["answer"], row.get("golden_answer_keywords") or []
        )
        row.pop("chunk_sources", None)

    timestamp = datetime.datetime.now(datetime.timezone.utc)
    out_path = (
        Path(output)
        if output
        else Path(f"eval/results/{timestamp:%Y%m%dT%H%M%SZ}_{profile}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "profile": profile,
        "scorer": scorer,
        "flags": flags,
        "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "filter": feature_filter,
        "mode": mode,
        "rows": rows,
        "skipped": skipped,
        "aggregate": aggregate(rows),
    }

    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print_table(payload)
    print(f"\nWrote: {out_path}")
    return payload, out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run eval harness")
    ap.add_argument(
        "--profile",
        required=True,
        choices=list(PROFILES.keys()),
        help="Flag profile to evaluate",
    )
    ap.add_argument(
        "--questions",
        default="eval/seed_questions.yaml",
        help="Path to seed questions YAML",
    )
    ap.add_argument(
        "--filter",
        default=None,
        help="Only run goldens with demonstrates_feature == FILTER (plus baseline)",
    )
    ap.add_argument(
        "--mode",
        default="service",
        choices=["service", "api"],
        help="Invocation mode",
    )
    ap.add_argument(
        "--scorer",
        default="local",
        choices=["local", "ragas"],
        help="Scoring backend: 'local' (IR + Jina embeddings, 0 Groq judge tokens) or 'ragas'",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of golden questions to evaluate",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay in seconds between golden questions",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: eval/results/<timestamp>_<profile>.json)",
    )
    args = ap.parse_args()

    try:
        run_profile(
            profile=args.profile,
            questions=args.questions,
            feature_filter=args.filter,
            mode=args.mode,
            output=args.output,
            scorer=args.scorer,
            limit=args.limit,
            delay=args.delay,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
