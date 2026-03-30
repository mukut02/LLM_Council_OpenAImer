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
