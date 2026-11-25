from __future__ import annotations

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig


from app.config import settings


METRICS = [
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy,
]



def _get_ragas_llm():
    """Create a Ragas-compatible LLM using app settings (Groq)."""
    lc_llm = ChatOpenAI(
        model=settings.llm_model_grader,
        api_key=settings.groq_api_key,
        base_url=settings.llm_base_url,
        temperature=0.0,
    )
    return LangchainLLMWrapper(lc_llm)

def _get_ragas_embeddings():
    """Create Ragas-compatible embeddings using app settings (Jina)."""
    lc_emb = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.jina_api_key,
        base_url=settings.embedding_base_url,
        check_embedding_ctx_length=False,
    )
    return LangchainEmbeddingsWrapper(lc_emb)

def build_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "user_input": [r["question"] for r in rows],
            "response": [r["answer"] for r in rows],
            "retrieved_contexts": [r["contexts"] for r in rows],
            "reference": [r["ground_truth"] for r in rows],
        }
    )

def run(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    ds = build_dataset(rows)
    result = evaluate(
        ds,
        metrics=METRICS,
        llm=_get_ragas_llm(),
        embeddings=_get_ragas_embeddings(),
        run_config=RunConfig(max_workers=1, max_retries=6, timeout=180),
        show_progress=False,
    )
    return result.to_pandas().to_dict(orient="records")



