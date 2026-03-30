import json
from functools import lru_cache


@lru_cache(maxsize=4)
def load_examples(path="data/examples.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("examples", [])


def load_references(path="data/examples.json"):
    return [ex["ai"] for ex in load_examples(path)]
