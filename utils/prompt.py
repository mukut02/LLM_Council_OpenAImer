from utils.loader import load_examples


def _extract_turns_from_json_chat(chat_obj, max_turns=5):
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
        turns = _extract_turns_from_json_chat(ex, max_turns=5)
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
        "Use conversation memory from the current session history.\n"
        "Generate the full conversation JSON for this chat.\n"
        "Output must be valid JSON only (no markdown, no explanation).\n"
        "Keep the same user_msg keys and values from the current session history.\n"
        "For every user_msgN, include a generated AI_msgN reply in sequence.\n"
        "If the uploaded JSON already has AI messages, you may rewrite them so the whole output is the selected model's version.\n"
        "Return keys in sequence like: user_msg1, AI_msg1, user_msg2, AI_msg2, ...\n"
        f"Return at most {max_turns} interactions.\n\n"
        f"Few-shot examples (max {max_turns} interactions each):\n"
        f"{few_shot_text}\n\n"
        "Current session history:\n"
        f"{history_text}\n\n"
        "Return JSON:"
    )


def build_sequential_turn_prompt(history_obj, current_turn_idx, num_examples=2, max_turns=5):
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
    previous_lines = []
    current_user_msg = ""
    for idx, (u, a) in enumerate(history_turns, start=1):
        if idx < current_turn_idx:
            if u:
                previous_lines.append(f"user_msg{idx}: {u}")
            if a:
                previous_lines.append(f"AI_msg{idx}: {a}")
        elif idx == current_turn_idx:
            current_user_msg = u
            break

    few_shot_text = "\n\n".join(few_shot_sections) if few_shot_sections else "No few-shot examples."
    memory_text = "\n".join(previous_lines) if previous_lines else "No previous generated turns yet."

    return (
        "You are a thoughtful, emotionally intelligent friend.\n"
        "Follow the emotional depth and tone of the few-shot examples.\n"
        "Use only the prior generated conversation turns as memory.\n"
        f"You are now generating AI_msg{current_turn_idx} for user_msg{current_turn_idx}.\n"
        "Output must be valid JSON only (no markdown, no explanation).\n"
        f'Return exactly this shape: {{"AI_msg{current_turn_idx}": "<your reply>"}}\n'
        "Do not return any other keys.\n\n"
        f"Few-shot examples (max {max_turns} interactions each):\n"
        f"{few_shot_text}\n\n"
        "Previous generated conversation memory:\n"
        f"{memory_text}\n\n"
        f"Current user message (user_msg{current_turn_idx}):\n"
        f"{current_user_msg}\n\n"
        "Return JSON:"
    )
