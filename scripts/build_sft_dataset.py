import json
from pathlib import Path

INPUT = "data/insta-conversations.jsonl"
OUTPUT = "data/insta-sft.jsonl"

# max messages to keep in context
WINDOW_MESSAGES = 12

# discard extremely short assistant replies
MIN_ASSISTANT_CHARS = 2


def is_good_target(text: str) -> bool:
    text = text.strip()

    if len(text) < MIN_ASSISTANT_CHARS:
        return False

    return True


def build_examples(messages):
    examples = []

    for i, msg in enumerate(messages):
        if msg["role"] != "saurya":
            continue

        target = msg["content"]

        if not is_good_target(target):
            continue

        start = max(0, i - WINDOW_MESSAGES + 1)

        sample = messages[start : i + 1]

        if len(sample) < 2:
            continue

        examples.append({"messages": sample})

    return examples


def main():
    total_conversations = 0
    total_examples = 0

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

    with (
        open(INPUT, encoding="utf-8") as infile,
        open(OUTPUT, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            conv = json.loads(line)

            examples = build_examples(conv["messages"])

            for ex in examples:
                outfile.write(json.dumps(ex, ensure_ascii=False) + "\n")

            total_examples += len(examples)
            total_conversations += 1

    print(f"Conversations: {total_conversations}")
    print(f"Training examples: {total_examples}")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
