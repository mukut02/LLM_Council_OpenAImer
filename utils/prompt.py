from utils.loader import load_examples


def _extract_turns_from_json_chat(chat_obj, max_turns=6):
    turns = []
    if not isinstance(chat_obj, dict):
        return turns
    for i in range(1, max_turns + 1):
        u = str(chat_obj.get(f"user_msg{i}", "")).strip()
        a = str(chat_obj.get(f"AI_msg{i}", chat_obj.get(f"ai_msg{i}", ""))).strip()
        if not u and not a:
            continue
        turns.append((u, a))
    return turns


def build_prompt(user_input, num_examples=3):
    examples = load_examples()[: max(0, num_examples)]

    few_shot_blocks = []
    for ex in examples:
        # Backward compatible with old single-turn examples.
        u = str(ex.get("user", "")).strip()
        a = str(ex.get("ai", "")).strip()
        if u and a:
            few_shot_blocks.append(f"User: {u}\nAI: {a}")
            continue

        # New chat-style JSON examples.
        turns = _extract_turns_from_json_chat(ex, max_turns=6)
        if turns:
            lines = []
            for idx, (tu, ta) in enumerate(turns, start=1):
                lines.append(f"user_msg{idx}: {tu}")
                lines.append(f"AI_msg{idx}: {ta}")
            few_shot_blocks.append("\n".join(lines))

    few_shot_text = "\n\n".join(few_shot_blocks)

    return (
        "You are a thoughtful, emotionally intelligent friend.\n"
        "Respond naturally, with empathy and depth.\n\n"
        "Here are examples of the expected style:\n\n"
        f"{few_shot_text}\n\n"
        f"User: {user_input}\n"
        "AI:"
    )


def build_few_shot_json_prompt(history_obj, current_user_msg, num_examples=2, max_turns=5):
    examples = load_examples()[: max(0, num_examples)]

    few_shot_sections = []
    for shot_idx, ex in enumerate(examples, start=1):
        turns = _extract_turns_from_json_chat(ex, max_turns=max_turns)
        if not turns:
            continue
        lines = [f"Few-shot conversation {shot_idx}:"]
        for idx, (u, a) in enumerate(turns, start=1):
            lines.append(f"user_msg{idx}: {u}")
            lines.append(f"AI_msg{idx}: {a}")
        few_shot_sections.append("\n".join(lines))

    history_turns = _extract_turns_from_json_chat(history_obj, max_turns=max_turns)
    history_lines = []
    for idx, (u, a) in enumerate(history_turns, start=1):
        if u:
            history_lines.append(f"user_msg{idx}: {u}")
        if a:
            history_lines.append(f"AI_msg{idx}: {a}")

    few_shot_text = "\n\n".join(few_shot_sections) if few_shot_sections else "No few-shot examples."
    history_text = "\n".join(history_lines) if history_lines else "No prior conversation history."

    return (
        "You are a thoughtful, emotionally intelligent friend.\n"
        "Follow the emotional depth and tone of the few-shot examples.\n"
        # "Use conversation memory from the current session history.\n"
        "Return only the next assistant response text (not JSON).\n\n"
        "Few-shot examples (max 6 interactions each):\n"
        f"{few_shot_text}\n\n"
        "Current session history:\n"
        f"{history_text}\n\n"
        "Current user message:\n"
        f"{current_user_msg}\n\n"
        "Assistant:"
    )
