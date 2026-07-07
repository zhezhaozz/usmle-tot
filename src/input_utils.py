import json
from pathlib import Path


DATA_FOLDER = "./data"

def load_dataset(path):
    """
    Supports:
    - JSONL: one JSON object per line
    - JSON: a list of JSON objects
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    # Try JSON list first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL
    examples = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            examples.append(json.loads(line))
    return examples


def format_options(options):
    """
    Converts {"A": "...", "B": "..."} to:
    A. ...
    B. ...
    """
    letters = ["A", "B", "C", "D"]
    return "\n".join([f"{letter}. {options[letter]}" for letter in letters if letter in options])

