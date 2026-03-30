from functools import lru_cache
import re

from openai import OpenAI

from config import GROQ_API_KEY

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found. Set environment variable GROQ_API_KEY "
        "(or Streamlit Cloud secret) before running."
    )

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

MODEL_ALIASES = {
    "qwen-2.5-32b": [
        "qwen/qwen3-32b",
        "qwen-2.5-32b",
    ],
    "deepseek-r1-distill-qwen-32b": [
        "deepseek-r1-distill-qwen-32b",
        "deepseek/deepseek-r1-distill-qwen-32b",
    ],
}


@lru_cache(maxsize=1)
def list_available_models():
    models = client.models.list()
    return {m.id for m in models.data if getattr(m, "id", None)}


def _resolve_model_id(requested_model):
    candidates = [requested_model] + MODEL_ALIASES.get(requested_model, [])

    try:
        available = list_available_models()
    except Exception:
        # If model listing fails (permissions/network), fall back to requested value.
        return requested_model

    for candidate in candidates:
        if candidate in available:
            return candidate

    if "qwen" in requested_model.lower():
        qwen_candidates = [m for m in available if "qwen" in m.lower()]
        if qwen_candidates:
            return qwen_candidates[0]

    if "deepseek" in requested_model.lower():
        deepseek_candidates = [m for m in available if "deepseek" in m.lower()]
        if deepseek_candidates:
            return deepseek_candidates[0]

    raise ValueError(
        f"Model '{requested_model}' is not available for this API key/project. "
        "Check model permissions or update config.MODELS."
    )


def _strip_leading_think_block(text):
    if not text:
        return text
    # Remove only the first leading <think>...</think> section if present.
    return re.sub(r"^\s*<think>.*?</think>\s*", "", text, count=1, flags=re.IGNORECASE | re.DOTALL)


def generate(model, prompt):
    resolved_model = _resolve_model_id(model)
    response = client.chat.completions.create(
        model=resolved_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        timeout=30,
    )
    content = response.choices[0].message.content
    if "qwen" in model.lower() or "deepseek" in model.lower():
        content = _strip_leading_think_block(content)
    return {
        "requested_model": model,
        "resolved_model": resolved_model,
        "text": content,
    }
