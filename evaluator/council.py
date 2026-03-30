import json
import re
from statistics import median

from llms.generate import generate_all
from utils.loader import load_examples


def _clamp_01(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, x))


def _extract_json_block(text):
    if not text:
        return {}
    cleaned = text.strip().strip("`")
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _history_text(conversation_history, max_turns=6):
    turns = _normalize_history(conversation_history)
    if not turns:
        return "No previous conversation turns in this session."
    turns = turns[-max_turns:]
    lines = []
    for i, turn in enumerate(turns, start=1):
        u = turn.get("user", "").strip()
        a = turn.get("assistant", "").strip()
        lines.append(f"Turn {i} User: {u}")
        lines.append(f"Turn {i} Assistant: {a}")
    return "\n".join(lines)


def _normalize_history(conversation_history):
    """
    Accepts either:
    1) list[{"user": "...", "assistant": "..."}]
    2) dict with keys like user_msg1, AI_msg1, user_msg2, AI_msg2...
    """
    if not conversation_history:
        return []

    if isinstance(conversation_history, list):
        normalized = []
        for turn in conversation_history:
            if not isinstance(turn, dict):
                continue
            normalized.append(
                {
                    "user": str(turn.get("user", "")),
                    "assistant": str(turn.get("assistant", "")),
                }
            )
        return normalized

    if isinstance(conversation_history, dict):
        by_idx = {}
        for key, value in conversation_history.items():
            if not isinstance(key, str):
                continue
            k = key.strip().lower()
            match = re.match(r"^(user|ai)_msg(\d+)$", k)
            if not match:
                continue
            role, idx_str = match.groups()
            idx = int(idx_str)
            if idx not in by_idx:
                by_idx[idx] = {"user": "", "assistant": ""}
            if role == "user":
                by_idx[idx]["user"] = str(value)
            else:
                by_idx[idx]["assistant"] = str(value)

        turns = [by_idx[i] for i in sorted(by_idx.keys())]
        return turns

    return []


def _ground_truth_text(max_examples=4):
    examples = load_examples()[:max_examples]
    if not examples:
        return "No ground-truth examples provided."
    lines = []
    for i, ex in enumerate(examples, start=1):
        category = ex.get("category", "General")
        u = ex.get("user", "").strip()
        a = ex.get("ai", "").strip()
        lines.append(f"Example {i} Category: {category}")
        lines.append(f"Example {i} User: {u}")
        lines.append(f"Example {i} Preferred AI: {a}")
    return "\n".join(lines)


def _build_council_prompt(user_prompt, participant_answer, conversation_history):
    history = _history_text(conversation_history)
    ground_truth = _ground_truth_text()

    return f"""
You are an evaluator in a 3-LLM council.
Score the participant answer from 0 to 1 on these four parameters:
1) inference: how well it reads between the lines and understands the user's internal state
2) memory: how well it uses/retains context from the session history
3) ground_truth: how aligned it is with the provided ground-truth style/context
4) actionability: how actionable and useful the suggestions are

User prompt:
{user_prompt}

Participant answer:
{participant_answer}

Conversation history:
{history}

Ground truth context:
{ground_truth}

Return ONLY valid JSON:
{{
  "inference": <0 to 1>,
  "memory": <0 to 1>,
  "ground_truth": <0 to 1>,
  "actionability": <0 to 1>
}}
"""


def evaluate_with_council(user_prompt, participant_answer, conversation_history):
    judge_prompt = _build_council_prompt(
        user_prompt=user_prompt,
        participant_answer=participant_answer,
        conversation_history=conversation_history,
    )

    raw_judgments = generate_all(judge_prompt)
    parsed = []

    for item in raw_judgments:
        model = item.get("model", "unknown")
        text = item.get("text", "")
        data = _extract_json_block(text)
        parsed.append(
            {
                "model": model,
                "inference": _clamp_01(data.get("inference")),
                "memory": _clamp_01(data.get("memory")),
                "ground_truth": _clamp_01(data.get("ground_truth")),
                "actionability": _clamp_01(data.get("actionability")),
                "raw": text,
            }
        )

    if not parsed:
        return {
            "median_inference": 0.0,
            "median_memory": 0.0,
            "median_ground_truth": 0.0,
            "median_actionability": 0.0,
            "final_score": 0.0,
            "judges": [],
        }

    med_inf = median([p["inference"] for p in parsed])
    med_mem = median([p["memory"] for p in parsed])
    med_gt = median([p["ground_truth"] for p in parsed])
    med_act = median([p["actionability"] for p in parsed])

    final_score = (
        0.3 * med_inf
        + 0.25 * med_mem
        + 0.2 * med_gt
        + 0.15 * med_act
    )

    return {
        "median_inference": med_inf,
        "median_memory": med_mem,
        "median_ground_truth": med_gt,
        "median_actionability": med_act,
        "final_score": final_score,
        "judges": parsed,
    }
