from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import WEIGHTS
from evaluator.metrics import bertscore, semantic_similarity
from llms.generate import generate_all
from utils.prompt import build_prompt


app = FastAPI(title="Mental AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    num_examples: int = Field(default=3, ge=1, le=10)


class EvaluateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model_answer: str = Field(..., min_length=1)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    few_shot_prompt = build_prompt(req.prompt, num_examples=req.num_examples)
    outputs = generate_all(few_shot_prompt)
    return {"outputs": outputs}


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest):
    semantic = semantic_similarity(req.prompt, req.model_answer)
    bert_f1 = bertscore(req.prompt, req.model_answer)
    weighted = (WEIGHTS["semantic"] * semantic) + (WEIGHTS["bert"] * bert_f1)
    return {
        "semantic_similarity": semantic,
        "bertscore_f1": bert_f1,
        "weighted_score": weighted,
    }
