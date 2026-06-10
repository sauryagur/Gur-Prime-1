import json
from pathlib import Path

INPUT = "data/insta-conversations.jsonl"
OUTPUT = "data/insta-sft.jsonl"

WINDOW_TURNS = 8

MIN_ASSISTANT_CHARS = 2


def is_good_target(text: str) -> bool:
    text = text.strip()
    return len(text) >= MIN_ASSISTANT_CHARS


def merge_turns(messages):
    turns = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()

        if not content:
            continue

        if not turns:
            turns.append(
                {
                    "role": role,
                    "content": content,
                }
            )
            continue

        if role == turns[-1]["role"]:
            turns[-1]["content"] += "\n" + content
        else:
            turns.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    return turns


def build_examples(messages):
    turns = merge_turns(messages)

    examples = []

    for i, turn in enumerate(turns):
        # Only train on Saurya responses
        if turn["role"] != "saurya":
            continue

        if not is_good_target(turn["content"]):
            continue

        start = max(0, i - WINDOW_TURNS + 1)

        context = turns[start : i + 1]

        # Remove leading saurya turns
        # We want contexts to start from a friend whenever possible
        while context and context[0]["role"] == "saurya":
            context = context[1:]

        if len(context) < 2:
            continue

        # Must end with saurya
        if context[-1]["role"] != "saurya":
            continue

        # Must contain at least one friend turn
        if not any(t["role"] == "friend" for t in context):
            continue

        examples.append({"messages": context})

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

            messages = conv.get("messages", [])

            examples = build_examples(messages)

            for ex in examples:
                outfile.write(json.dumps(ex, ensure_ascii=False) + "\n")

            total_examples += len(examples)
            total_conversations += 1

    print(f"Conversations: {total_conversations}")
    print(f"Training examples: {total_examples}")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
