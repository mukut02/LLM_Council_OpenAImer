from concurrent.futures import ThreadPoolExecutor, as_completed

from config import MODELS
from llms.groq_client import generate


def _generate_one(model, prompt):
    try:
        result = generate(model, prompt)
        if isinstance(result, dict):
            return {
                "model": model,
                "requested_model": result.get("requested_model", model),
                "resolved_model": result.get("resolved_model", model),
                "text": result.get("text", ""),
            }
        return {
            "model": model,
            "requested_model": model,
            "resolved_model": model,
            "text": str(result),
        }
    except Exception as e:
        return {
            "model": model,
            "requested_model": model,
            "resolved_model": model,
            "text": f"Error: {str(e)}",
        }


def generate_one(model, prompt):
    return _generate_one(model, prompt)


def generate_all(prompt):
    if not MODELS:
        return []

    # Keep enough parallelism for network-bound API calls without oversubscribing threads.
    max_workers = min(len(MODELS), 8)
    outputs_by_model = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {
            executor.submit(_generate_one, model, prompt): model for model in MODELS
        }
        for future in as_completed(future_to_model):
            result = future.result()
            outputs_by_model[result["model"]] = result

    # Preserve config model order in the final output.
    return [outputs_by_model[model] for model in MODELS if model in outputs_by_model]
