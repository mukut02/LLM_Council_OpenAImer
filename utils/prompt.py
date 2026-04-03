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


def build_sequential_turn_prompt(
    history_obj,
    current_turn_idx,
    num_examples=2,
    max_turns=5,
    window_size=2,
):
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
    previous_pairs = []
    current_user_msg = ""
    for idx, (u, a) in enumerate(history_turns, start=1):
        if idx < current_turn_idx:
            previous_pairs.append((idx, u, a))
        elif idx == current_turn_idx:
            current_user_msg = u
            break

    few_shot_text = "\n\n".join(few_shot_sections) if few_shot_sections else "No few-shot examples."
    windowed_pairs = previous_pairs[-max(0, window_size):] if window_size is not None else previous_pairs
    memory_lines = []
    for idx, u, a in windowed_pairs:
        if u:
            memory_lines.append(f"user_msg{idx}: {u}")
        if a:
            memory_lines.append(f"AI_msg{idx}: {a}")
    memory_text = "\n".join(memory_lines) if memory_lines else "No previous generated turns yet."

    return (
        "You are a thoughtful, emotionally intelligent friend.\n"
        "Follow the emotional depth and tone of the few-shot examples.\n"
        "Use only the sliding-window memory below for context.\n"
        "Do not use or recreate any uploaded AI answers.\n"
        f"You are now generating AI_msg{current_turn_idx} for user_msg{current_turn_idx}.\n"
        "Output must be valid JSON only (no markdown, no explanation).\n"
        f'Return exactly this shape: {{"AI_msg{current_turn_idx}": "<your reply>"}}\n'
        "Do not return any other keys.\n\n"
        f"Few-shot examples (max {max_turns} interactions each):\n"
        f"{few_shot_text}\n\n"
        f"Sliding-window memory from previous turns (last {max(0, window_size)} turn(s)):\n"
        f"{memory_text}\n\n"
        f"Current user message (user_msg{current_turn_idx}):\n"
        f"{current_user_msg}\n\n"
        "Return JSON:"
    )
