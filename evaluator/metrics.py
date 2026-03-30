from functools import lru_cache

from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")
bert_scorer = BERTScorer(
    lang="en",
    model_type="distilbert-base-uncased",
    batch_size=8,
    idf=False,
    rescale_with_baseline=False,
)


@lru_cache(maxsize=2048)
def _encode_text(text):
    return model.encode(text, convert_to_tensor=True, normalize_embeddings=True)


@lru_cache(maxsize=4096)
def semantic_similarity(a, b):
    emb1 = _encode_text(a)
    emb2 = _encode_text(b)
    return util.cos_sim(emb1, emb2).item()


@lru_cache(maxsize=4096)
def bertscore(a, b):
    _, _, f1 = bert_scorer.score([a], [b])
    return f1.mean().item()
