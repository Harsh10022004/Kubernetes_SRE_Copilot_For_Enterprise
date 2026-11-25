from openai import OpenAI

from app.config import settings
from app.services.query_cache_service import query_cache


openai_client = OpenAI(
    api_key=settings.jina_api_key,
    base_url=settings.embedding_base_url,
)

def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    if model is None:
        model = settings.embedding_model

    results: list[list[float] | None] = [None] * len(texts)
    miss_indices: list[int] = []
    miss_texts: list[str] = []

    for i, text in enumerate(texts):
        cached = query_cache.get_embedding(text)
        if cached is not None:
            results[i] = cached

        else:
            miss_indices.append(i)
            miss_texts.append(text)
        
    if miss_texts:
        batch_size = 64
        for start in range(0, len(miss_texts), batch_size):
            batch_texts = miss_texts[start : start + batch_size]
            response = openai_client.embeddings.create(
                input=batch_texts,
                model=model,
                encoding_format="float",
            )
            for idx_in_batch, item in enumerate(response.data):
                miss_idx = start + idx_in_batch
                original_idx = miss_indices[miss_idx]
                vector = item.embedding
                results[original_idx] = vector
                query_cache.set_embedding(miss_texts[miss_idx], vector)

    return [r for r in results if r is not None]

