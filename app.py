import json
import re
from html import escape
from statistics import median
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

from evaluator.council import evaluate_with_council
from llms.generate import generate_all
from utils.prompt import build_few_shot_json_prompt

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


def render_chat_window(structured):
    chat_html = ['<div class="chat-window">']
    max_idx = 0
    for k in structured.keys():
        m = re.match(r"^(user|ai)_msg(\d+)$", str(k), flags=re.IGNORECASE)
        if m:
            max_idx = max(max_idx, int(m.group(2)))
    for i in range(1, max_idx + 1):
        u = str(structured.get(f"user_msg{i}", structured.get(f"User_msg{i}", ""))).strip()
        a = str(structured.get(f"AI_msg{i}", structured.get(f"ai_msg{i}", ""))).strip()
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
    st.caption("Using all available few-shot examples from data/examples.json (max 6 turns each).")

    history_sig = json_signature(uploaded_history) if uploaded_history is not None else ""

    if st.button("Generate Model Answers", use_container_width=True):
        if uploaded_history is None:
            st.warning("Please upload conversation JSON first.")
            st.session_state.generated_outputs = []
        else:
            eval_prompt = ""
            fill_pending = False

            pending_idx, pending_user = get_latest_pending_user(uploaded_history)
            if pending_idx is not None and pending_user:
                eval_prompt = pending_user
                fill_pending = True
            else:
                latest_idx, latest_user = get_latest_user_any(uploaded_history)
                if latest_idx is not None and latest_user:
                    eval_prompt = latest_user
                    st.info(
                        f"No pending turn found. Using latest user message user_msg{latest_idx} as prompt."
                    )
                else:
                    st.warning("Could not find any user_msgN in uploaded JSON.")
                    st.session_state.generated_outputs = []
                    st.stop()

            with st.spinner("Generating answers from all configured models..."):
                runtime_prompt = build_few_shot_json_prompt(
                    history_obj=uploaded_history,
                    current_user_msg=eval_prompt,
                    num_examples=10_000,
                    max_turns=6,
                )
                raw_outputs = generate_all(runtime_prompt)

                formatted_outputs = []
                for item in raw_outputs:
                    structured = upsert_json_turn(
                        uploaded_history,
                        eval_prompt,
                        item["text"],
                        fill_pending=fill_pending,
                    )
                    formatted_outputs.append(
                        {
                            "model": item["model"],
                            "text": json.dumps(structured, ensure_ascii=False, indent=2),
                            "structured": structured,
                            "raw_text": item["text"],
                        }
                    )

                st.session_state.generated_outputs = formatted_outputs
                st.session_state.generated_meta = {
                    "eval_prompt": eval_prompt,
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

        st.markdown("Model Answer Chat")
        render_chat_window(selected_output["structured"])
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
