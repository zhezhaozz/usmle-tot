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

def construct_prompt(template, case, mode="generation", tot_add=None):
    user_content = template["user_message"].replace("{question}", case["question"]).replace("{options}", format_options(case["options"]))

    if mode == "tot_reasoning_generation":
        user_content = user_content.replace("{plan}", tot_add)
    if mode == "tot_termination":
        user_content = user_content.replace("{reasoning}", tot_add)
    if mode == "tot_evaluation1":
        user_content = user_content.replace("{PLAN_OR_REASONING}", "analytical plan").replace("{tot_options}", tot_add)
    if mode == "tot_evaluation2":
        user_content = user_content.replace("{PLAN_OR_REASONING}", "reasonings").replace("{tot_options}", tot_add)

    prompt =[ 
        {"role":"system",
         "content": template["system_message"]},
        {"role": "user",
         "content": user_content}
    ]
    return prompt
