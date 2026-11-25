from __future__ import annotations

import threading
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.models import RetrievedChunk



VECTOR_SIZE = settings.embedding_dimension
_SPARSE_INDEX = None
_SPARSE_LOCK = threading.RLock()


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=30)

def ensure_collection() -> None:
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if settings.qdrant_collection in existing:
        info = client.get_collection(collection_name=settings.qdrant_collection)
        vectors_cfg = info.config.params.vectors
        current_size = getattr(vectors_cfg, "size", None)
        if current_size is None and isinstance(vectors_cfg, dict) and vectors_cfg:
            first_cfg = next(iter(vectors_cfg.values()))
            current_size = getattr(first_cfg, "size", None)
        if current_size == VECTOR_SIZE:
            return
        client.delete_collection(collection_name=settings.qdrant_collection)

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

def upsert_chunks(chunks: list[RetrievedChunk], embeddings: list[list[float]]) -> None:
    global _SPARSE_INDEX
    ensure_collection()
    client  = get_client()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": chunk.text, "source": chunk.source},
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    with _SPARSE_LOCK:
        _SPARSE_INDEX = None

def search(query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
    client = get_client()
    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        RetrievedChunk(
            text=p.payload.get("text", ""),
            source=p.payload.get("source", ""),
            score=float(p.score),
        )
        for p in results
    ]


def _build_sparse_index():
    global _SPARSE_INDEX
    with _SPARSE_LOCK:
        if _SPARSE_INDEX is not None:
            return _SPARSE_INDEX

        from app.services.sparse_vector_service import SparseVectorIndex

        client = get_client()
        all_points = []
        next_page = None
        while True:
            points, next_page = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=1000,
                offset=next_page,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(points)
            if next_page is None:
                break

        documents = [
            {
                "text": point.payload.get("text", "") if point.payload else "",
                "source": point.payload.get("source", "") if point.payload else "",
                "id": str(point.id),
            }
            for point in all_points
        ]
        sparse_index = SparseVectorIndex()
        sparse_index.fit(documents)
        _SPARSE_INDEX = sparse_index
        return _SPARSE_INDEX

def sparse_search(query_text: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Pure sparse search using TF-IDF (no dense embeddings, no fusion)."""
    sparse_index = _build_sparse_index()
    return sparse_index.search(query_text, top_k=top_k)


def hybrid_search(
    query_embedding: list[float],
    query_text: str,
    top_k: int = 5,
    rrf_k: int = 60,
    sparse_top_k: int = 20,
) -> list[RetrievedChunk]:

    from app.services.sparse_vector_service import fuse_rrf
    dense_results = search(query_embedding, top_k=sparse_top_k)
    sparse_index = _build_sparse_index()
    sparse_results = sparse_index.search(query_text, top_k=sparse_top_k)
    fused = fuse_rrf([dense_results, sparse_results], rrf_k=rrf_k)
    return fused[:top_k]
