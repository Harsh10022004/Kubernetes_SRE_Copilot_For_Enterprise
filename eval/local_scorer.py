"""Local zero-Groq-token evaluation scorer using IR metrics + Jina embeddings."""

from __future__ import annotations

import re

import numpy as np

from app.services.embedding_service import embed_texts
from eval.post_checks import keyword_recall_check, normalize_source, source_overlap

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_ABSTENTION_PHRASES = (
    "don't have enough information",
    "do not have enough information",
    "not mentioned in the provided",
    "no information",
    "cannot answer",
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _split_sentences(answer: str) -> list[str]:
    stripped = answer.strip()
    if not stripped:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(stripped) if p.strip()]
    long_parts = [p for p in parts if len(p) >= 15][:12]
    return long_parts if long_parts else [stripped]


def _compute_context_recall(row: dict) -> float:
    contexts = [c for c in (row.get("contexts") or []) if c and c.strip()]
    if not contexts:
        return 0.0

    src_overlap = source_overlap(
        row.get("actual_sources") or [],
        row.get("golden_sources") or [],
    )["overlap_pct"]

    keywords = row.get("golden_answer_keywords") or []
    if keywords:
        joined_ctx = "\n".join(contexts).lower()
        kw_hits = sum(1 for kw in keywords if kw.lower() in joined_ctx)
        ctx_kw_recall = kw_hits / len(keywords)
    else:
        ctx_kw_recall = src_overlap

    return round(0.6 * src_overlap + 0.4 * ctx_kw_recall, 4)


def _compute_context_precision(row: dict) -> float:
    contexts = row.get("contexts") or []
    if not contexts:
        return 0.0

    chunk_sources = row.get("chunk_sources") or row.get("actual_sources") or []
    golden_norm_set = {
        normalize_source(s) for s in (row.get("golden_sources") or []) if s
    }
    keywords = row.get("golden_answer_keywords") or []
    is_sql_intent = row.get("intent") == "sql"

    chunk_flags: list[tuple[bool, bool, bool]] = []
    for idx, ctx in enumerate(contexts):
        src = chunk_sources[idx] if idx < len(chunk_sources) else ""
        norm_src = normalize_source(src)
        is_gold_src = norm_src in golden_norm_set
        is_sql = is_sql_intent or norm_src == "query_result"
        ctx_lower = (ctx or "").lower()
        has_kw = any(kw.lower() in ctx_lower for kw in keywords) if keywords else True
        chunk_flags.append((is_gold_src, has_kw, is_sql))

    any_gold_with_kw = any(
        is_gold and (has_kw or is_sql) for is_gold, has_kw, is_sql in chunk_flags
    )

    relevant = [
        (is_gold and (has_kw or is_sql)) if any_gold_with_kw else is_gold
        for is_gold, has_kw, is_sql in chunk_flags
    ]

    hits = 0
    precisions: list[float] = []
    for rank, is_rel in enumerate(relevant, start=1):
        if is_rel:
            hits += 1
            precisions.append(hits / rank)

    return round(sum(precisions) / hits, 4) if hits else 0.0


def _compute_answer_relevancy(row: dict, emb_map: dict[str, np.ndarray]) -> float:
    answer = (row.get("answer") or "").strip()
    if not answer:
        return 0.0

    question = (row.get("question") or "").strip()
    q_vec = emb_map.get(question)
    a_vec = emb_map.get(answer)
    q_sim = max(0.0, _cosine(q_vec, a_vec)) if q_vec is not None and a_vec is not None else 0.0

    keywords = row.get("golden_answer_keywords") or []
    kw_hit = keyword_recall_check(answer, keywords)["hit_rate"] if keywords else q_sim

    answer_lower = answer.lower()
    if kw_hit == 0.0 and any(phrase in answer_lower for phrase in _ABSTENTION_PHRASES):
        q_sim *= 0.3

    return round(min(1.0, 0.5 * q_sim + 0.5 * kw_hit), 4)


def _compute_faithfulness(row: dict, emb_map: dict[str, np.ndarray]) -> float:
    contexts = [c.strip() for c in (row.get("contexts") or []) if c and c.strip()]
    sentences = _split_sentences(row.get("answer") or "")
    if not contexts or not sentences:
        return 0.0

    ctx_vecs = [emb_map[c] for c in contexts if c in emb_map]
    if not ctx_vecs:
        return 0.0

    scores = [
        max(max(0.0, _cosine(emb_map[sent], c_vec)) for c_vec in ctx_vecs)
        if sent in emb_map
        else 0.0
        for sent in sentences
    ]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _build_embedding_map(rows: list[dict]) -> dict[str, np.ndarray]:
    unique_texts: list[str] = []
    seen: set[str] = set()

    def _add(text: str | None) -> None:
        if not text:
            return
        cleaned = text.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_texts.append(cleaned)

    for row in rows:
        _add(row.get("question"))
        _add(row.get("answer"))
        for sent in _split_sentences(row.get("answer") or ""):
            _add(sent)
        for ctx in row.get("contexts") or []:
            _add(ctx)

    if not unique_texts:
        return {}

    vectors = embed_texts(unique_texts)
    return {
        text: np.asarray(vec, dtype=np.float32)
        for text, vec in zip(unique_texts, vectors)
    }


def score_rows(rows: list[dict]) -> list[dict]:
    """Compute faithfulness, context_precision, context_recall, and answer_relevancy."""
    if not rows:
        return []

    emb_map = _build_embedding_map(rows)
    scored: list[dict] = []
    for row in rows:
        scored.append(
            {
                "faithfulness": _compute_faithfulness(row, emb_map),
                "context_precision": _compute_context_precision(row),
                "context_recall": _compute_context_recall(row),
                "answer_relevancy": _compute_answer_relevancy(row, emb_map),
            }
        )
    return scored
