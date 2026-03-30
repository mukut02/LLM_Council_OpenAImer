import json
import re
from html import escape
from statistics import median
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

from config import MODELS
from evaluator.council import evaluate_with_council
from llms.generate import generate_one
from utils.prompt import build_sequential_turn_prompt

st.set_page_config(page_title="OpenAImer LLM Council")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
      html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
      }
      /* Hide Streamlit top chrome/menu bar */
      header[data-testid="stHeader"] {
        display: none;
      }
      [data-testid="stToolbar"] {
        display: none;
      }
      [data-testid="stDecoration"] {
        display: none;
      }
      #MainMenu, footer {
        visibility: hidden;
      }
      .stApp {
        background:
          radial-gradient(1200px 500px at 10% -10%, rgba(255, 86, 48, 0.22), transparent 60%),
          radial-gradient(900px 420px at 95% 8%, rgba(255, 166, 0, 0.16), transparent 55%),
          radial-gradient(700px 350px at 50% 115%, rgba(255, 43, 87, 0.18), transparent 60%),
          linear-gradient(145deg, #060b16 0%, #0a1222 45%, #090f1c 100%);
        background-attachment: fixed;
      }
      .block-container {
        padding-top: 1.3rem;
        max-width: 1100px;
      }
      .hero {
        background: linear-gradient(130deg, rgba(255, 124, 67, 0.18), rgba(20, 36, 62, 0.72));
        border: 1px solid rgba(255, 144, 77, 0.4);
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 0 0 1px rgba(255, 121, 63, 0.15), 0 22px 45px rgba(2, 8, 20, 0.45);
        backdrop-filter: blur(8px);
      }
      .hero h1 {
        margin: 0;
        color: #ffe8d7;
        font-size: 1.75rem;
        text-shadow: 0 0 20px rgba(255, 126, 66, 0.25);
      }
      .hero p {
        margin: 8px 0 0;
        color: #ffd9bf;
      }
      .section-title {
        color: #ffbd8b;
        font-weight: 700;
        margin: 4px 0 10px;
        letter-spacing: 0.02em;
      }
      .chat-window {
        background: linear-gradient(180deg, rgba(11, 19, 36, 0.92), rgba(7, 12, 25, 0.96));
        border: 1px solid rgba(111, 145, 255, 0.28);
        border-radius: 12px;
        padding: 12px;
        height: 420px;
        overflow: auto;
        box-shadow: inset 0 0 0 1px rgba(93, 129, 255, 0.18), 0 14px 28px rgba(0, 0, 0, 0.35);
      }
      .chat-row {
        display: flex;
        margin: 8px 0;
      }
      .chat-row.user {
        justify-content: flex-end;
      }
      .chat-row.ai {
        justify-content: flex-start;
      }
      .bubble {
        max-width: 85%;
        padding: 10px 12px;
        border-radius: 12px;
        line-height: 1.5;
        white-space: pre-wrap;
        font-size: 0.96rem;
      }
      .bubble.user {
        background: linear-gradient(130deg, #ff6b3d 0%, #ff934a 100%);
        color: #ffffff;
        border-bottom-right-radius: 4px;
        box-shadow: 0 0 18px rgba(255, 112, 56, 0.35);
      }
      .bubble.ai {
        background: linear-gradient(130deg, #16253f 0%, #1a2e4f 100%);
        color: #f3f6fb;
        border: 1px solid rgba(109, 142, 255, 0.45);
        border-bottom-left-radius: 4px;
        box-shadow: 0 0 16px rgba(77, 118, 255, 0.22);
      }
      [data-testid="stButton"] button {
        border-radius: 10px !important;
        border: 1px solid rgba(255, 124, 64, 0.42) !important;
        background: linear-gradient(130deg, rgba(255, 114, 55, 0.14), rgba(255, 153, 71, 0.08)) !important;
        color: #ffe7d3 !important;
      }
      [data-testid="stButton"] button:hover {
        box-shadow: 0 0 22px rgba(255, 121, 63, 0.24);
      }
      [data-testid="stMetricValue"] {
        color: #ffe6d1;
      }
      [data-testid="stMetricLabel"] {
        color: #ffbf95;
      }
      [data-testid="stMarkdownContainer"], [data-testid="stText"] {
        color: #f6f8ff;
      }
    </style>
    <div class="hero">
      <h1>OpenAImer LLM Council</h1>
      <p>Upload conversation JSON, generate model answers with few-shot memory learning, and score with the 3-LLM council.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "generated_outputs" not in st.session_state:
    st.session_state.generated_outputs = []
if "generated_meta" not in st.session_state:
    st.session_state.generated_meta = {
        "eval_prompt": "",
        "history_sig": "",
    }
if "council_cache" not in st.session_state:
    st.session_state.council_cache = {}

MAX_GENERATED_TURNS = 5


def parse_history_json_obj(history_obj):
    if not isinstance(history_obj, dict):
        return []
    idx_map = {}
    for key, value in history_obj.items():
        if not isinstance(key, str):
            continue
        m = re.match(r"^(user|ai)_msg(\d+)$", key.strip(), flags=re.IGNORECASE)
        if not m:
            continue
        role, idx = m.group(1).lower(), int(m.group(2))
        if idx not in idx_map:
            idx_map[idx] = {"user": "", "assistant": ""}
        if role == "user":
            idx_map[idx]["user"] = str(value)
        else:
            idx_map[idx]["assistant"] = str(value)
    return [idx_map[i] for i in sorted(idx_map.keys())]


def next_turn_index(history_obj):
    if not isinstance(history_obj, dict):
        return 1
    max_idx = 0
    for key in history_obj.keys():
        if not isinstance(key, str):
            continue
        m = re.match(r"^(user|ai)_msg(\d+)$", key.strip(), flags=re.IGNORECASE)
        if m:
            max_idx = max(max_idx, int(m.group(2)))
    return max_idx + 1


def get_latest_pending_user(history_obj):
    if not isinstance(history_obj, dict):
        return None, None
    max_idx = 0
    for key in history_obj.keys():
        if not isinstance(key, str):
            continue
        m = re.match(r"^(user|ai)_msg(\d+)$", key.strip(), flags=re.IGNORECASE)
        if m:
            max_idx = max(max_idx, int(m.group(2)))
    if max_idx == 0:
        return None, None

    user_val = history_obj.get(f"user_msg{max_idx}", history_obj.get(f"User_msg{max_idx}", ""))
    ai_val = history_obj.get(f"AI_msg{max_idx}", history_obj.get(f"ai_msg{max_idx}", ""))
    user_text = str(user_val).strip()
    ai_text = str(ai_val).strip()
    if user_text and not ai_text:
        return max_idx, user_text
    return None, None


def get_latest_user_any(history_obj):
    if not isinstance(history_obj, dict):
        return None, None
    max_idx = 0
    for key in history_obj.keys():
        if not isinstance(key, str):
            continue
        m = re.match(r"^(user|ai)_msg(\d+)$", key.strip(), flags=re.IGNORECASE)
        if m:
            max_idx = max(max_idx, int(m.group(2)))
    if max_idx == 0:
        return None, None
    user_val = history_obj.get(f"user_msg{max_idx}", history_obj.get(f"User_msg{max_idx}", ""))
    user_text = str(user_val).strip()
    if not user_text:
        return None, None
    return max_idx, user_text


def append_json_turn(history_obj, user_msg, ai_msg):
    base = dict(history_obj) if isinstance(history_obj, dict) else {}
    idx = next_turn_index(base)
    base[f"user_msg{idx}"] = user_msg
    base[f"AI_msg{idx}"] = ai_msg
    return base


def upsert_json_turn(history_obj, user_msg, ai_msg, fill_pending=False):
    base = dict(history_obj) if isinstance(history_obj, dict) else {}
    if fill_pending:
        pending_idx, pending_user = get_latest_pending_user(base)
        if pending_idx is not None:
            if not user_msg:
                user_msg = pending_user
            base[f"user_msg{pending_idx}"] = user_msg
            base[f"AI_msg{pending_idx}"] = ai_msg
            return base
    return append_json_turn(base, user_msg, ai_msg)


def get_latest_ai_from_history_json(history_obj):
    if not isinstance(history_obj, dict):
        return ""
    direct_ai = str(history_obj.get("AI_msg", history_obj.get("ai_msg", ""))).strip()
    if direct_ai:
        return direct_ai
    last_idx = 0
    for key in history_obj.keys():
        if not isinstance(key, str):
            continue
        m = re.match(r"^ai_msg(\d+)$", key.strip(), flags=re.IGNORECASE)
        if m:
            last_idx = max(last_idx, int(m.group(1)))
    if last_idx == 0:
        return ""
    return str(history_obj.get(f"AI_msg{last_idx}", history_obj.get(f"ai_msg{last_idx}", "")))


def json_signature(obj):
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(obj)


def normalize_history_json_obj(history_obj, max_turns=MAX_GENERATED_TURNS):
    turns = parse_history_json_obj(history_obj)
    normalized = {}
    for idx, turn in enumerate(turns[: max(0, max_turns)], start=1):
        user_text = str(turn.get("user", "")).strip()
        ai_text = str(turn.get("assistant", "")).strip()
        if user_text:
            normalized[f"user_msg{idx}"] = user_text
        if ai_text:
            normalized[f"AI_msg{idx}"] = ai_text
    return normalized


def extract_generated_history_json(raw_text, fallback_history, user_prompt, fill_pending=False):
    parsed_obj = None
    text = str(raw_text or "").strip()
    if text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                parsed_obj = parsed
        except Exception:
            fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
            candidate = fenced.group(1) if fenced else None
            if not candidate:
                raw_obj = re.search(r"(\{[\s\S]*\})", text)
                candidate = raw_obj.group(1) if raw_obj else None
            if candidate:
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        parsed_obj = parsed
                except Exception:
                    parsed_obj = None

    if isinstance(parsed_obj, dict):
        has_numbered_keys = any(
            isinstance(key, str) and re.match(r"^(user|ai)_msg\d+$", key.strip(), flags=re.IGNORECASE)
            for key in parsed_obj.keys()
        )
        if has_numbered_keys:
            return normalize_history_json_obj(parsed_obj, max_turns=MAX_GENERATED_TURNS)

        assistant_text = str(
            parsed_obj.get("AI_msg", parsed_obj.get("ai_msg", parsed_obj.get("response", "")))
        ).strip()
        if assistant_text:
            fallback = upsert_json_turn(
                fallback_history,
                user_prompt,
                assistant_text,
                fill_pending=fill_pending,
            )
            return normalize_history_json_obj(fallback, max_turns=MAX_GENERATED_TURNS)

    fallback = upsert_json_turn(
        fallback_history,
        user_prompt,
        text.strip(),
        fill_pending=fill_pending,
    )
    return normalize_history_json_obj(fallback, max_turns=MAX_GENERATED_TURNS)


def to_pretty_json_text(obj):
    if not isinstance(obj, dict):
        return "{}"
    return json.dumps(obj, ensure_ascii=False, indent=2)


def build_user_only_history(history_obj, max_turns=MAX_GENERATED_TURNS):
    turns = parse_history_json_obj(history_obj)
    user_only = {}
    for idx, turn in enumerate(turns[: max(0, max_turns)], start=1):
        user_text = str(turn.get("user", "")).strip()
        if user_text:
            user_only[f"user_msg{idx}"] = user_text
    return user_only


def extract_generated_ai_text(raw_text, turn_idx):
    parsed_obj = None
    text = str(raw_text or "").strip()
    if text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                parsed_obj = parsed
        except Exception:
            fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
            candidate = fenced.group(1) if fenced else None
            if not candidate:
                raw_obj = re.search(r"(\{[\s\S]*\})", text)
                candidate = raw_obj.group(1) if raw_obj else None
            if candidate:
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        parsed_obj = parsed
                except Exception:
                    parsed_obj = None

    if isinstance(parsed_obj, dict):
        ordered_keys = [
            f"AI_msg{turn_idx}",
            f"ai_msg{turn_idx}",
            "AI_msg",
            "ai_msg",
            "response",
        ]
        for key in ordered_keys:
            value = str(parsed_obj.get(key, "")).strip()
            if value:
                return value

    return text


def generate_structured_history_for_model(model, input_history_obj, max_turns=MAX_GENERATED_TURNS):
    user_only_history = build_user_only_history(input_history_obj, max_turns=max_turns)
    generated_history = {}
    last_result = None

    for turn_idx in range(1, max_turns + 1):
        user_text = str(user_only_history.get(f"user_msg{turn_idx}", "")).strip()
        if not user_text:
            continue

        generated_history[f"user_msg{turn_idx}"] = user_text
        turn_prompt = build_sequential_turn_prompt(
            history_obj=generated_history,
            current_turn_idx=turn_idx,
            num_examples=10_000,
            max_turns=max_turns,
        )
        last_result = generate_one(model, turn_prompt)
        ai_text = extract_generated_ai_text(last_result.get("text", ""), turn_idx).strip()
        generated_history[f"AI_msg{turn_idx}"] = ai_text

    return {
        "model": model,
        "requested_model": last_result.get("requested_model", model) if last_result else model,
        "resolved_model": last_result.get("resolved_model", model) if last_result else model,
        "raw_text": last_result.get("text", "") if last_result else "",
        "structured": normalize_history_json_obj(generated_history, max_turns=max_turns),
    }


def render_chat_window(structured):
    turns = parse_history_json_obj(structured)
    render_chat_turns(turns)


def render_chat_turns(turns):
    chat_html = ['<div class="chat-window">']
    for turn in turns:
        u = str(turn.get("user", "")).strip()
        a = str(turn.get("assistant", "")).strip()
        if u:
            chat_html.append(
                f'<div class="chat-row user"><div class="bubble user">{escape(u)}</div></div>'
            )
        if a:
            chat_html.append(
                f'<div class="chat-row ai"><div class="bubble ai">{escape(a)}</div></div>'
            )
    chat_html.append("</div>")
    st.markdown("".join(chat_html), unsafe_allow_html=True)


def get_generated_assistant_preview_turns(input_history_obj, output_history_obj):
    return get_continuation_turns(input_history_obj, output_history_obj)


def get_generated_delta_turns(input_history_obj, output_history_obj):
    input_turns = parse_history_json_obj(input_history_obj)
    output_turns = parse_history_json_obj(output_history_obj)

    min_len = min(len(input_turns), len(output_turns))
    diff_idx = None
    for i in range(min_len):
        if (
            input_turns[i].get("user", "") != output_turns[i].get("user", "")
            or input_turns[i].get("assistant", "") != output_turns[i].get("assistant", "")
        ):
            diff_idx = i
            break

    if diff_idx is None:
        if len(output_turns) > len(input_turns):
            diff_idx = len(input_turns)
        elif output_turns:
            diff_idx = max(0, len(output_turns) - 1)
        else:
            diff_idx = 0

    return output_turns[diff_idx:]


def get_continuation_turns(input_history_obj, output_history_obj):
    """
    Deterministic continuation extraction:
    1) Remove longest common prefix between input and output chats.
    2) Return the remaining output turns as continuation.
    3) Hide duplicated first user prompt if it repeats input's last user turn.
    """
    input_turns = parse_history_json_obj(input_history_obj)
    output_turns = parse_history_json_obj(output_history_obj)

    if not output_turns:
        return []

    prefix = 0
    while prefix < len(input_turns) and prefix < len(output_turns):
        in_t = input_turns[prefix]
        out_t = output_turns[prefix]
        if (
            str(in_t.get("user", "")).strip() == str(out_t.get("user", "")).strip()
            and str(in_t.get("assistant", "")).strip() == str(out_t.get("assistant", "")).strip()
        ):
            prefix += 1
        else:
            break

    continuation = [
        {
            "user": str(t.get("user", "")).strip(),
            "assistant": str(t.get("assistant", "")).strip(),
        }
        for t in output_turns[prefix:]
    ]

    if continuation and input_turns:
        last_input_user = str(input_turns[-1].get("user", "")).strip()
        if continuation[0]["user"] == last_input_user:
            continuation[0]["user"] = ""

    return continuation


def get_continuation_by_generated_index(output_history_obj, generated_turn_idx, input_history_obj=None):
    output_turns = parse_history_json_obj(output_history_obj)
    if not output_turns:
        return []

    if generated_turn_idx is None:
        return []

    try:
        start_idx = max(1, int(generated_turn_idx))
    except (TypeError, ValueError):
        return []

    continuation = output_turns[start_idx - 1 :]
    if not continuation:
        return []

    # Optional de-dup of repeated user prompt against input history's last user.
    if input_history_obj:
        input_turns = parse_history_json_obj(input_history_obj)
        if input_turns:
            last_input_user = str(input_turns[-1].get("user", "")).strip()
            if str(continuation[0].get("user", "")).strip() == last_input_user:
                continuation[0]["user"] = ""

    return continuation


left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    st.markdown('<div class="section-title">Council Memory JSON (Required)</div>', unsafe_allow_html=True)
    history_file = st.file_uploader(
        "Upload conversation JSON",
        type=["json"],
        help='Format: {"user_msg1":"...","AI_msg1":"...", ...}',
    )

    uploaded_history = None
    if history_file is not None:
        try:
            uploaded_history = json.loads(history_file.getvalue().decode("utf-8"))
            turns = parse_history_json_obj(uploaded_history)
            st.success(f"Conversation JSON loaded. Detected turns: {len(turns)}")
            st.markdown("Input JSON Chat")
            render_chat_window(uploaded_history)
        except Exception:
            st.error("Invalid conversation JSON. Please upload a valid file.")

with right_col:
    st.markdown('<div class="section-title">JSON Mode Generation</div>', unsafe_allow_html=True)
    st.caption("Using all available few-shot examples from data/examples.json (max 5 turns each).")

    history_sig = json_signature(uploaded_history) if uploaded_history is not None else ""

    if st.button("Generate Model Answers", use_container_width=True):
        if uploaded_history is None:
            st.warning("Please upload conversation JSON first.")
            st.session_state.generated_outputs = []
        else:
            with st.spinner("Generating answers from all configured models..."):
                formatted_outputs = []
                user_only_history = build_user_only_history(uploaded_history, max_turns=MAX_GENERATED_TURNS)
                if not user_only_history:
                    st.warning("Could not find any user_msgN in uploaded JSON.")
                    st.session_state.generated_outputs = []
                    st.stop()

                for model in MODELS:
                    item = generate_structured_history_for_model(
                        model=model,
                        input_history_obj=uploaded_history,
                        max_turns=MAX_GENERATED_TURNS,
                    )
                    structured = item["structured"]
                    formatted_outputs.append(
                        {
                            "model": item["model"],
                            "requested_model": item.get("requested_model", item["model"]),
                            "resolved_model": item.get("resolved_model", item["model"]),
                            "raw_text": item.get("raw_text", ""),
                            "parsed_json": to_pretty_json_text(structured),
                            "structured": structured,
                        }
                    )

                st.session_state.generated_outputs = formatted_outputs
                st.session_state.generated_meta = {
                    "eval_prompt": get_latest_user_any(uploaded_history)[1] or "",
                    "history_sig": history_sig,
                }

outputs = st.session_state.generated_outputs
meta = st.session_state.generated_meta
outputs_match_context = (
    bool(outputs)
    and uploaded_history is not None
    and meta.get("history_sig") == history_sig
)

selected_model = None
selected_output = None
if outputs_match_context:
    with right_col:
        st.write("Select a generated answer to preview:")
        model_names = [o["model"] for o in outputs]
        selected_model = st.selectbox("Generated model", model_names)
        selected_output = next(o for o in outputs if o["model"] == selected_model)

        st.caption(
            f"Configured model: `{selected_output.get('requested_model', selected_model)}` | "
            f"Provider model: `{selected_output.get('resolved_model', selected_model)}`"
        )
        st.markdown("Generated JSON")
        st.code(selected_output.get("parsed_json", "{}"), language="json")

        st.markdown("Generated Model Chat")
        generated_structured = selected_output.get("structured", {})
        render_chat_window(generated_structured)
elif outputs:
    st.info("JSON file or few-shot count changed. Click Generate Model Answers to refresh outputs.")

st.markdown("---")
st.subheader("Live Council Grading")

if outputs_match_context and selected_output and uploaded_history is not None:
    st.markdown(f"**Selected Generated Model: `{selected_model}`**")

    eval_prompt = meta.get("eval_prompt", "")
    council_history = uploaded_history
    history_key = json_signature(council_history)

    results_by_model = {}
    missing_jobs = []
    for out in outputs:
        out_model = out["model"]
        participant_answer = get_latest_ai_from_history_json(out["structured"])
        cache_key = f"{eval_prompt}::{out_model}::{participant_answer}::{history_key}"
        cached = st.session_state.council_cache.get(cache_key)
        if cached is not None:
            results_by_model[out_model] = cached
        else:
            missing_jobs.append((out_model, participant_answer, cache_key))

    if missing_jobs:
        with st.spinner("Council grading in progress..."):
            max_workers = min(len(missing_jobs), 3)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        evaluate_with_council,
                        user_prompt=eval_prompt,
                        participant_answer=participant_answer,
                        conversation_history=council_history,
                    ): (out_model, cache_key)
                    for out_model, participant_answer, cache_key in missing_jobs
                }
                for future in as_completed(future_map):
                    out_model, cache_key = future_map[future]
                    result_val = future.result()
                    st.session_state.council_cache[cache_key] = result_val
                    results_by_model[out_model] = result_val

    result = results_by_model[selected_model]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median Inference", f"{result['median_inference']:.4f}")
    c2.metric("Median Memory", f"{result['median_memory']:.4f}")
    c3.metric("Median Ground Truth", f"{result['median_ground_truth']:.4f}")
    c4.metric("Median Actionability", f"{result['median_actionability']:.4f}")
    st.metric("Final Score", f"{result['final_score']:.4f}")

    model_results = list(results_by_model.values())
    overall_inf = median([r["median_inference"] for r in model_results])
    overall_mem = median([r["median_memory"] for r in model_results])
    overall_gt = median([r["median_ground_truth"] for r in model_results])
    overall_act = median([r["median_actionability"] for r in model_results])
    overall_final = (
        0.3 * overall_inf
        + 0.25 * overall_mem
        + 0.2 * overall_gt
        + 0.15 * overall_act
    )

    st.markdown("### Overall (Across Generated Models)")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Median Inference", f"{overall_inf:.4f}")
    o2.metric("Median Memory", f"{overall_mem:.4f}")
    o3.metric("Median Ground Truth", f"{overall_gt:.4f}")
    o4.metric("Median Actionability", f"{overall_act:.4f}")
    st.metric("Overall Final Score", f"{overall_final:.4f}")

    st.markdown("### Model-wise Final Scores")
    for model_name, r in results_by_model.items():
        st.write(f"- `{model_name}`: `{r['final_score']:.4f}`")
else:
    st.info("Upload JSON and generate outputs to see council grading.")
