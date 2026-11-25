import logging
import re
import time

from openai import APIStatusError, OpenAI

from app.config import settings
from app.security.input_restructuring import truncate_text

logger = logging.getLogger(__name__)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

openai_client = OpenAI(
    api_key=settings.groq_api_key,
    base_url=settings.llm_base_url,
)


def _strip_think_tags(text: str) -> str:
    """Remove optional <think>...</think> reasoning blocks from model outputs."""
    return _THINK_RE.sub("", text).strip()


def _parse_retry_after(exc: APIStatusError) -> float | None:
    """Extract numeric retry-after seconds from an APIStatusError response."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _build_candidate_models(primary_model: str) -> list[str]:
    """Return unique candidate chat models for rate-limit rotation."""
    candidates: list[str] = []
    for candidate in (
        primary_model,
        settings.llm_model_grader,
        settings.llm_model_answer,
        "qwen/qwen3.8-27b",
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _call_chat_completion(
    system_prompt: str,
    user_message: str,
    model: str,
    temperature: float,
    default_user_message: str,
    response_format: dict | None = None,
) -> dict:
    """Call Groq chat completions with token truncation, model rotation, and 413/429 backoff."""
    if not user_message or not user_message.strip():
        user_message = default_user_message
    else:
        user_message, _ = truncate_text(user_message, max_tokens=4500)
        if not user_message.strip():
            user_message = default_user_message

    candidates = _build_candidate_models(model)
    max_rounds = 5
    last_exc: APIStatusError | None = None
    response = None

    for round_idx in range(max_rounds):
        retry_after_hint: float | None = None
        for candidate in candidates:
            kwargs: dict = {
                "model": candidate,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": temperature,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format

            try:
                response = openai_client.chat.completions.create(**kwargs)
                break
            except APIStatusError as exc:
                if exc.status_code not in (413, 429):
                    raise
                last_exc = exc
                parsed = _parse_retry_after(exc)
                if parsed is not None:
                    retry_after_hint = parsed if retry_after_hint is None else min(retry_after_hint, parsed)
                logger.warning(
                    "Groq %s on model=%s (round %d/%d); trying next candidate",
                    exc.status_code,
                    candidate,
                    round_idx + 1,
                    max_rounds,
                )

        if response is not None:
            break

        if round_idx < max_rounds - 1:
            delay = min(25.0, max(retry_after_hint or 0.0, 5.0 * (round_idx + 1)))
            logger.info("All Groq candidates rate-limited; sleeping %.1fs before retry", delay)
            time.sleep(delay)

    if response is None:
        assert last_exc is not None
        raise last_exc

    text = _strip_think_tags(response.choices[0].message.content or "")
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }
    return {"text": text, "usage": usage}


def generate(system_prompt: str, user_message: str, model: str | None = None, temperature: float = 0.0) -> dict:
    if model is None:
        model = settings.llm_model_answer
    return _call_chat_completion(
        system_prompt=system_prompt,
        user_message=user_message,
        model=model,
        temperature=temperature,
        default_user_message="Please respond.",
    )


def generate_with_json(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    if model is None:
        model = settings.llm_model_grader
    return _call_chat_completion(
        system_prompt=system_prompt,
        user_message=user_message,
        model=model,
        temperature=temperature,
        default_user_message="Please respond in JSON.",
        response_format={"type": "json_object"},
    )