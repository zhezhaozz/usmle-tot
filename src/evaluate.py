def get_gold_label(example):
    """
    Prefer answer_idx, but fall back to matching answer text.
    """
    if "answer_idx" in example and example["answer_idx"]:
        return example["answer_idx"].strip().upper()

    answer_text = example.get("answer", "").strip().lower()
    options = example.get("options", {})

    for letter, option_text in options.items():
        if option_text.strip().lower() == answer_text:
            return letter.strip().upper()

    return None

def evaluate(dataset, predictions):
    N = len(dataset)
    failure = 0
    correct = 0
    for case, pred in zip(dataset, predictions):
        gold = get_gold_label(case)
        is_correct = gold == pred if gold is not None else False

        if pred is None:
            failure += 1

        if gold is not None:
            if is_correct:
                correct += 1

    accuracy = correct / N if N > 0 else 0.0

    print()
    print(f"Total evaluated: {N}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Unparsed responses: {failure}")

