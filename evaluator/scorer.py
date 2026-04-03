from statistics import mean

from evaluator.metrics import semantic_similarity, bertscore
from config import WEIGHTS


def score_response(response, references):
    if not references:
        return 0.0

    semantic_scores = [(ref, semantic_similarity(response, ref)) for ref in references]

    # Run heavier BERTScore only on top semantic candidates for speed.
    top_k = min(3, len(semantic_scores))
    top_refs = sorted(semantic_scores, key=lambda x: x[1], reverse=True)[:top_k]

    best = 0.0
    for ref, sim in top_refs:
        bert = bertscore(response, ref)
        score = WEIGHTS["semantic"] * sim + WEIGHTS["bert"] * bert
        best = max(best, score)

    return best


def _normalize_history(conversation_history):
    if not isinstance(conversation_history, dict):
        return []

    turns = {}
    for key, value in conversation_history.items():
        if not isinstance(key, str):
            continue
        lower_key = key.strip().lower()
        if lower_key.startswith("user_msg"):
            idx = lower_key.replace("user_msg", "")
            if idx.isdigit():
                turns.setdefault(int(idx), {"user": "", "assistant": ""})
                turns[int(idx)]["user"] = str(value).strip()
        elif lower_key.startswith("ai_msg"):
            idx = lower_key.replace("ai_msg", "")
            if idx.isdigit():
                turns.setdefault(int(idx), {"user": "", "assistant": ""})
                turns[int(idx)]["assistant"] = str(value).strip()

    return [turns[idx] for idx in sorted(turns.keys())]


def score_generated_history(reference_history, generated_history):
    reference_turns = _normalize_history(reference_history)
    generated_turns = _normalize_history(generated_history)

    per_turn = []
    for idx, ref_turn in enumerate(reference_turns, start=1):
        ref_ai = str(ref_turn.get("assistant", "")).strip()
        if not ref_ai:
            continue

        gen_ai = ""
        if idx - 1 < len(generated_turns):
            gen_ai = str(generated_turns[idx - 1].get("assistant", "")).strip()

        semantic = semantic_similarity(gen_ai, ref_ai) if gen_ai else 0.0
        bert = bertscore(gen_ai, ref_ai) if gen_ai else 0.0
        weighted = (WEIGHTS["semantic"] * semantic) + (WEIGHTS["bert"] * bert)

        per_turn.append(
            {
                "turn": idx,
                "reference_ai": ref_ai,
                "generated_ai": gen_ai,
                "semantic": semantic,
                "bert": bert,
                "weighted": weighted,
            }
        )

    if not per_turn:
        return {
            "matched_turns": 0,
            "avg_semantic": 0.0,
            "avg_bert": 0.0,
            "final_score": 0.0,
            "per_turn": [],
        }

    return {
        "matched_turns": len(per_turn),
        "avg_semantic": mean(item["semantic"] for item in per_turn),
        "avg_bert": mean(item["bert"] for item in per_turn),
        "final_score": mean(item["weighted"] for item in per_turn),
        "per_turn": per_turn,
    }
