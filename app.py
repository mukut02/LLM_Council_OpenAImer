import json
import re
from html import escape
from statistics import median

import streamlit as st

import config
from evaluator.council import evaluate_input_history_with_council
from llms.groq_client import get_available_api_slots

DEFAULT_API_SLOT = getattr(config, "DEFAULT_API_SLOT", "api1")

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
      <p>Upload conversation JSON and score the existing AI answers with the 3-LLM council.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "council_cache" not in st.session_state:
    st.session_state.council_cache = {}

available_api_slots = get_available_api_slots()
if "selected_api_slot" not in st.session_state:
    st.session_state.selected_api_slot = (
        DEFAULT_API_SLOT if DEFAULT_API_SLOT in available_api_slots else (available_api_slots[0] if available_api_slots else DEFAULT_API_SLOT)
    )


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

    st.markdown('<div class="section-title">Council Ground Truth TXT (Optional)</div>', unsafe_allow_html=True)
    ground_truth_file = st.file_uploader(
        "Upload council ground truth text",
        type=["txt"],
        help="Optional .txt context used as council-provided ground truth during evaluation.",
    )

    uploaded_ground_truth = None
    if ground_truth_file is not None:
        try:
            uploaded_ground_truth = ground_truth_file.getvalue().decode("utf-8").strip()
            st.success("Ground-truth text loaded.")
            st.markdown("Council Ground Truth Context")
            st.text_area(
                "Ground truth preview",
                value=uploaded_ground_truth,
                height=220,
                disabled=True,
                label_visibility="collapsed",
            )
        except Exception:
            st.error("Invalid ground-truth text file. Please upload a valid `.txt` file.")

with right_col:
    st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)

    if available_api_slots:
        selected_api_slot = st.selectbox(
            "Select API Key",
            options=available_api_slots,
            index=available_api_slots.index(
                st.session_state.selected_api_slot
                if st.session_state.selected_api_slot in available_api_slots
                else available_api_slots[0]
            ),
            help="Choose which configured API key slot to use for the 3-LLM council run.",
        )
        st.session_state.selected_api_slot = selected_api_slot
        st.caption(f"Using `{selected_api_slot}` for the current council evaluation.")
    else:
        selected_api_slot = DEFAULT_API_SLOT
        st.warning("No API key slots are configured.")

    if st.button("Assess Input JSON", use_container_width=True):
        if uploaded_history is None:
            st.warning("Please upload conversation JSON first.")
        else:
            history_sig = json.dumps(uploaded_history, sort_keys=True, ensure_ascii=False)
            ground_truth_sig = (
                uploaded_ground_truth
                if uploaded_ground_truth is not None
                else ""
            )
            cache_key = f"input::{history_sig}::ground_truth::{ground_truth_sig}::api::{selected_api_slot}"
            with st.spinner("Assessing input JSON with the council..."):
                result_val = st.session_state.council_cache.get(cache_key)
                if result_val is None:
                    result_val = evaluate_input_history_with_council(
                        uploaded_history,
                        ground_truth_history=uploaded_ground_truth,
                        api_slot=selected_api_slot,
                    )
                    st.session_state.council_cache[cache_key] = result_val
                st.session_state["input_eval_result"] = result_val
                st.session_state["input_eval_has_ground_truth"] = uploaded_ground_truth is not None

st.markdown("---")
st.subheader("Live Evaluation")

result = st.session_state.get("input_eval_result")
if result and uploaded_history is not None:
    st.markdown("**Assessed Source: `Input JSON`**")
    if result.get("api_slot"):
        st.caption(f"API slot used: `{result['api_slot']}`")
    if result.get("ground_truth_source") == "provided":
        st.caption("Council ground-truth text context was used during evaluation.")
    else:
        st.caption("No separate council ground-truth text was provided. A short ground-truth summary was generated from the conversation and used for evaluation.")

    if result.get("ground_truth_context"):
        with st.expander("Ground Truth Context Used"):
            st.text(result["ground_truth_context"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median Inference", f"{result['median_inference']:.4f}")
    c2.metric("Median Memory", f"{result['median_memory']:.4f}")
    c3.metric("Median Ground Truth", f"{result['median_ground_truth']:.4f}")
    c4.metric("Median Actionability", f"{result['median_actionability']:.4f}")
    st.metric("Final Score", f"{result['final_score']:.4f}")

    st.caption("Evaluation is parsed into the requested median metrics using multi-judge scoring.")

    if result.get("turns"):
        st.markdown("### Turn-wise Scores")
        for item in result["turns"]:
            st.write(
                f"- `msg{item['turn']}`: inference=`{item['median_inference']:.4f}`, "
                f"memory=`{item['median_memory']:.4f}`, "
                f"ground_truth=`{item['median_ground_truth']:.4f}`, "
                f"actionability=`{item['median_actionability']:.4f}`, "
                f"final=`{item['final_score']:.4f}`"
            )

else:
    st.info("Upload JSON and assess the input file to see evaluation.")
