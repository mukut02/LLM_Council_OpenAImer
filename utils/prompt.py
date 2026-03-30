from utils.loader import load_examples


def build_prompt(user_input, num_examples=3):
    examples = load_examples()[: max(0, num_examples)]

    few_shot_blocks = []
    for ex in examples:
        u = ex.get("user", "").strip()
        a = ex.get("ai", "").strip()
        if u and a:
            few_shot_blocks.append(f"User: {u}\nAI: {a}")

    few_shot_text = "\n\n".join(few_shot_blocks)

    return (
        "You are a thoughtful, emotionally intelligent friend.\n"
        "Respond naturally, with empathy and depth.\n\n"
        "Here are examples of the expected style:\n\n"
        f"{few_shot_text}\n\n"
        f"User: {user_input}\n"
        "AI:"
    )
