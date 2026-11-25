
from __future__ import annotations

import os


def forbidden_keywords_check(answer: str, forbidden: list[str]) -> dict:
    answer_lower = answer.lower()
    hits = [kw for kw in forbidden if kw.lower() in answer_lower]
    return {"passed": not hits, "hits": hits}


def normalize_source(s: str) -> str:
    """Normalize a source identifier so URLs, SQL labels, and doc filenames align."""
    cleaned = s.strip().lower()
    if cleaned.startswith(("http://", "https://")):
        return "tavily_web"
    if cleaned in {"database query", "query_results", "query_result"}:
        return "query_result"
    name = os.path.basename(cleaned)
    if "__" in name:
        name = name.rsplit("__", 1)[-1]
    stem = os.path.splitext(name)[0]
    return stem.rstrip("s") if len(stem) > 3 else stem


def source_overlap(actual: list[str], golden: list[str]) -> dict:
    actual_set = {normalize_source(s) for s in actual if s}
    golden_set = {normalize_source(s) for s in golden if s}
    overlap = actual_set & golden_set

    return {
        "overlap_pct": round(len(overlap) / max(len(golden_set), 1), 3),
        "matched": sorted(overlap),
        "missed": sorted(golden_set - actual_set),
    }


def keyword_recall_check(answer: str, keywords: list[str]) -> dict:
    """Check case-insensitive substring recall of golden keywords in an answer."""
    answer_lower = answer.lower()
    matched = [kw for kw in keywords if kw.lower() in answer_lower]
    missed = [kw for kw in keywords if kw.lower() not in answer_lower]
    return {
        "hit_rate": round(len(matched) / max(len(keywords), 1), 3),
        "matched": matched,
        "missed": missed,
    }