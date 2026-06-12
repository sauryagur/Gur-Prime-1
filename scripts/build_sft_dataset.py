"""
Converts data/insta-conversations.jsonl → data/insta-sft.jsonl

Source schema (per line):
  {"messages": [{"role": "saurya"|"friend", "content": "...", "timestamp_ms": ...}, ...]}

Target schema expected by the training pipeline:
  {"messages": [{"role": "user"|"assistant", "content": "..."}, ...]}

Role mapping:
  "friend"  → "user"       (friend's message = the prompt / input)
  "saurya"  → "assistant"  (Saurya's reply = what we want the model to learn)

Additional transforms applied here (NOT in the training pipeline):
  - Strip timestamp_ms (pipeline's validate_message only wants role + content)
  - Drop messages with empty content after stripping
  - Drop conversations that start with "assistant" (no preceding user turn —
    apply_chat_template will error or produce garbage on these)
  - Drop conversations shorter than MIN_TURNS after all cleaning
  - Log every skip and every anomaly verbosely
"""

import json
import os
import re

INPUT_PATH = "data/insta-conversations.jsonl"
OUTPUT_PATH = "data/insta-sft.jsonl"
MIN_TURNS = 2


ROLE_MAP = {
    "saurya": "assistant",
    "friend": "user",
}


def convert_messages(raw_messages: list) -> tuple[list, list]:
    """
    Convert a list of source messages to pipeline-ready dicts.
    Returns (converted, list_of_warnings).
    """
    converted = []
    warnings = []

    for i, msg in enumerate(raw_messages):
        role_raw = (msg.get("role") or "").strip().lower()
        content = (msg.get("content") or "").strip()

        if role_raw not in ROLE_MAP:
            warnings.append(f"  [WARN] msg[{i}]: unknown role '{role_raw}' — skipping")
            continue

        if not content:
            warnings.append(f"  [WARN] msg[{i}]: empty content after strip — skipping")
            continue

        converted.append(
            {
                "role": ROLE_MAP[role_raw],
                "content": content,
            }
        )

    return converted, warnings


def validate_turn_order(messages: list) -> list[str]:
    """
    Check for structural issues the chat template will choke on.
    Returns a list of warning strings (empty = all good).
    """
    issues = []

    if not messages:
        issues.append("  [WARN] empty message list")
        return issues

    if messages[0]["role"] != "user":
        issues.append(
            f"  [WARN] first turn is '{messages[0]['role']}', not 'user' — "
            "apply_chat_template will produce malformed output"
        )

    for i in range(1, len(messages)):
        if messages[i]["role"] == messages[i - 1]["role"]:
            issues.append(
                f"  [WARN] consecutive same-role turns at index {i - 1}→{i} "
                f"(both '{messages[i]['role']}')"
            )
            break

    return issues


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Input file not found: {INPUT_PATH}")
        return

    written = 0
    skipped_too_short = 0
    skipped_bad_start = 0
    skipped_empty = 0
    total_warnings = 0
    total_msgs_in = 0
    total_msgs_out = 0
    total_msgs_dropped = 0

    print(f"[READ]  {INPUT_PATH}")
    print(f"[WRITE] {OUTPUT_PATH}\n")

    with (
        open(INPUT_PATH, "r", encoding="utf-8") as inp,
        open(OUTPUT_PATH, "w", encoding="utf-8") as out,
    ):
        for line_no, line in enumerate(inp, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[{line_no}] [ERROR] JSON parse failed: {e} — skipping")
                skipped_empty += 1
                continue

            raw_messages = obj.get("messages", [])
            total_msgs_in += len(raw_messages)

            print(f"[{line_no}] source turns: {len(raw_messages)}")

            messages, warnings = convert_messages(raw_messages)
            for w in warnings:
                print(w)
                total_warnings += 1

            dropped = len(raw_messages) - len(messages)
            if dropped:
                print(f"  [INFO] dropped {dropped} message(s) during conversion")
                total_msgs_dropped += dropped

            issues = validate_turn_order(messages)
            for issue in issues:
                print(issue)
                total_warnings += 1

            first_user_idx = None
            for i, m in enumerate(messages):
                if m["role"] == "user":
                    first_user_idx = i
                    break

            if first_user_idx is None:
                print(f"  [SKIP] No user turn found — dropping")
                skipped_bad_start += 1
                continue

            if first_user_idx > 0:
                dropped_leading = first_user_idx
                messages = messages[first_user_idx:]
                print(f"  [INFO] Trimmed {dropped_leading} leading assistant turn(s)")

            total_msgs_out += len(messages)
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            written += 1
            print(
                f"  [OK]  wrote {len(messages)} turns "
                f"(first role: {messages[0]['role']}, "
                f"last role: {messages[-1]['role']})"
            )

    w = 60
    print(f"\n{'═' * w}")
    print(f"  SFT CONVERSION COMPLETE")
    print(f"{'═' * w}")
    print(f"  Conversations written    : {written}")
    print(f"  Skipped — bad start role : {skipped_bad_start}")
    print(f"  Skipped — too short      : {skipped_too_short}")
    print(f"  Skipped — parse error    : {skipped_empty}")
    print(f"{'─' * w}")
    print(f"  Source messages total    : {total_msgs_in}")
    print(f"  Output messages total    : {total_msgs_out}")
    print(f"  Messages dropped         : {total_msgs_dropped}")
    print(f"  Warnings emitted         : {total_warnings}")
    print(f"{'═' * w}\n")


if __name__ == "__main__":
    main()
