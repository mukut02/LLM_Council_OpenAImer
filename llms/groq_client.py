from functools import lru_cache
import re

from openai import OpenAI

from config import API_KEYS, DEFAULT_API_SLOT

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


def get_available_api_slots():
    return [slot for slot, key in API_KEYS.items() if key]


def _resolve_api_slot(api_slot=None):
    requested_slot = api_slot or DEFAULT_API_SLOT
    if requested_slot not in API_KEYS or not API_KEYS[requested_slot]:
        available_slots = get_available_api_slots()
        if not available_slots:
            raise RuntimeError("No Groq API keys are configured.")
        if requested_slot == DEFAULT_API_SLOT:
            return available_slots[0]
        raise ValueError(
            f"API slot '{requested_slot}' is not available. "
            f"Available slots: {', '.join(available_slots)}"
        )
    return requested_slot


@lru_cache(maxsize=8)
def _get_client(api_slot):
    resolved_slot = _resolve_api_slot(api_slot)
    return OpenAI(
        api_key=API_KEYS[resolved_slot],
        base_url="https://api.groq.com/openai/v1",
    )


@lru_cache(maxsize=16)
def list_available_models(api_slot=None):
    client = _get_client(api_slot)
    models = client.models.list()
    return {m.id for m in models.data if getattr(m, "id", None)}


def _resolve_model_id(requested_model, api_slot=None):
    candidates = [requested_model] + MODEL_ALIASES.get(requested_model, [])

    try:
        available = list_available_models(api_slot)
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


def generate(model, prompt, api_slot=None):
    resolved_slot = _resolve_api_slot(api_slot)
    client = _get_client(resolved_slot)
    resolved_model = _resolve_model_id(model, resolved_slot)
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
        "api_slot": resolved_slot,
        "requested_model": model,
        "resolved_model": resolved_model,
        "text": content,
    }
