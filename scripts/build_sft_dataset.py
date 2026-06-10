import json
from pathlib import Path

INPUT = "data/insta-conversations.jsonl"
OUTPUT = "data/insta-sft.jsonl"

WINDOW_TURNS = 8
STEP = 2

MIN_ASSISTANT_CHARS = 5

BAD_SHORT_RESPONSES = {
    "ok", "okay", "k", "lol", "lmao", "👍", "😂"
}


def normalize_role(role: str) -> str:
    role = role.lower().strip()

    if role == "saurya":
        return "assistant"
    if role == "friend":
        return "user"

    return None


def is_good_target(text: str) -> bool:
    text = (text or "").strip()
    if len(text) < MIN_ASSISTANT_CHARS:
        return False
    if text.lower() in BAD_SHORT_RESPONSES:
        return False
    return True


def merge_turns(messages):
    turns = []

    for msg in messages:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()

        if not role or not content:
            continue

        role = normalize_role(role)
        if role is None:
            continue

        if not turns or turns[-1]["role"] != role:
            turns.append({"role": role, "content": content})
        else:
            turns[-1]["content"] += "\n" + content

    return turns


def build_examples(messages):
    turns = merge_turns(messages)

    assistant_indices = [
        i for i, t in enumerate(turns)
        if t["role"] == "assistant" and is_good_target(t["content"])
    ]

    examples = []

    for i in assistant_indices[::STEP]:
        start = max(0, i - WINDOW_TURNS + 1)
        context = turns[start:i + 1]

        if len(context) < 2:
            continue

        # must end in assistant
        if context[-1]["role"] != "assistant":
            continue

        # must include at least one user message
        if not any(t["role"] == "user" for t in context[:-1]):
            continue

        # ensure alternating sanity (no broken structure)
        cleaned = []
        prev_role = None

        for t in context:
            if t["role"] == prev_role:
                # merge accidental duplicates
                cleaned[-1]["content"] += "\n" + t["content"]
            else:
                cleaned.append(t)
                prev_role = t["role"]

        if len(cleaned) < 2:
            continue

        examples.append({"messages": cleaned})

    return examples


def main():
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

    total_conversations = 0
    total_examples = 0

    with open(INPUT, encoding="utf-8") as infile, open(OUTPUT, "w", encoding="utf-8") as outfile:
        for line in infile:
            conv = json.loads(line)
            messages = conv.get("messages", [])

            if not isinstance(messages, list) or len(messages) < 2:
                continue

            examples = build_examples(messages)

            for ex in examples:
                outfile.write(json.dumps(ex, ensure_ascii=False) + "\n")

            total_examples += len(examples)
            total_conversations += 1

    print(f"Conversations processed: {total_conversations}")
    print(f"Training examples: {total_examples}")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
