import streamlit as st

from config import WEIGHTS
from evaluator.metrics import bertscore, semantic_similarity
from llms.generate import generate_all
from utils.prompt import build_prompt

st.set_page_config(page_title="Prompt vs Model Answer Evaluator")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
      html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
      }
      .block-container {
        padding-top: 1.5rem;
        max-width: 1100px;
      }
      .hero {
        background: linear-gradient(120deg, #edf7ef 0%, #f5faf6 100%);
        border: 1px solid #d5e9d8;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 16px;
      }
      .hero h1 {
        margin: 0;
        color: #14321b;
        font-size: 1.8rem;
      }
      .hero p {
        margin: 8px 0 0;
        color: #3f5e46;
      }
      .section-title {
        color: #183a22;
        font-weight: 700;
        margin: 4px 0 10px;
      }
    </style>
    <div class="hero">
      <h1>Prompt and Model Answer Evaluator</h1>
      <p>Generate few-shot responses, select one, and evaluate quality in a cleaner workflow.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "generated_outputs" not in st.session_state:
    st.session_state.generated_outputs = []
if "model_answer_input" not in st.session_state:
    st.session_state.model_answer_input = ""

left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    st.markdown('<div class="section-title">Input</div>', unsafe_allow_html=True)
    prompt_text = st.text_area("Enter Prompt", height=140, placeholder="Type the user prompt...")
    model_answer = st.text_area(
        "Enter Model Answer",
        height=200,
        key="model_answer_input",
        placeholder="Paste a model answer or use generated output...",
    )

with right_col:
    st.markdown('<div class="section-title">Few-Shot Generation</div>', unsafe_allow_html=True)
    num_examples = st.slider("Few-shot examples to use", min_value=1, max_value=10, value=3)

    if st.button("Generate Model Answers", use_container_width=True):
        if not prompt_text.strip():
            st.warning("Please enter a prompt first.")
        else:
            with st.spinner("Generating answers from all configured models..."):
                few_shot_prompt = build_prompt(prompt_text, num_examples=num_examples)
                st.session_state.generated_outputs = generate_all(few_shot_prompt)

outputs = st.session_state.generated_outputs
if outputs:
    with right_col:
        st.write("Select a generated answer to use as Model Answer:")
        model_names = [o["model"] for o in outputs]
        selected_model = st.selectbox("Generated model", model_names)

        selected_output = next(o for o in outputs if o["model"] == selected_model)
        st.code(selected_output["text"], language="text")

        if st.button("Use Selected Output in Model Answer", use_container_width=True):
            st.session_state.model_answer_input = selected_output["text"]
            st.rerun()

if st.button("Evaluate", type="primary", use_container_width=True):
    if not prompt_text.strip() or not st.session_state.model_answer_input.strip():
        st.warning("Please enter both Prompt and Model Answer.")
    else:
        semantic = semantic_similarity(prompt_text, st.session_state.model_answer_input)
        bert_f1 = bertscore(prompt_text, st.session_state.model_answer_input)
        total_score = (WEIGHTS["semantic"] * semantic) + (WEIGHTS["bert"] * bert_f1)

        st.subheader("Evaluation Metrics")
        col1, col2, col3 = st.columns(3)

        col1.metric("Semantic Similarity", f"{semantic:.4f}")
        col2.metric("BERTScore (F1)", f"{bert_f1:.4f}")
        col3.metric("Weighted Score", f"{total_score:.4f}")