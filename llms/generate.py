from concurrent.futures import ThreadPoolExecutor, as_completed

from config import MODELS
from llms.groq_client import generate


def _generate_one(model, prompt):
    try:
        text = generate(model, prompt)
        return {"model": model, "text": text}
    except Exception as e:
        return {"model": model, "text": f"Error: {str(e)}"}


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
