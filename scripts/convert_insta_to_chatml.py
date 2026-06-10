import glob
import json
import os
import re

RAW_INBOX = "data/raw-insta/your_instagram_activity/messages/inbox"
OUTPUT = "data/insta-processed.jsonl"
MY_NAME = "Saurya"

SYSTEM_PATTERNS = [
    r"sent an attachment\.?\s*$",
    r"shared a story\.?\s*$",
    r"shared a post\.?\s*$",
    r"shared a reel\.?\s*$",
    r"shared a video\.?\s*$",
    r"shared a photo\.?\s*$",
    r"liked a message\.?\s*$",
    r"reacted\b.*\bto your message",
    r"is in your close friends",
    r"started a call",
    r"ended a call",
    r"missed a call",
    r"created a group",
    r"named the group",
    r"changed the group",
    r"removed\b.*\bfrom",
    r"added\b.*\bto",
    r"left the group",
    r"\bjoined",
    r"^on liked$",
]

SYSTEM_RE = re.compile("|".join(SYSTEM_PATTERNS), re.IGNORECASE)


def fix_mojibake(text: str) -> str:
    """Instagram exports encode UTF-8 bytes as individual \\uXXXX code points.
    Encode back to bytes via latin-1, then decode as proper UTF-8.
    Falls back gracefully for already-correct strings.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except Exception:
        try:
            return text.encode("latin-1").decode("utf-8", errors="replace")
        except Exception:
            return text


def fix_json_strings(obj):
    """Recursively fix all strings in parsed JSON at read level."""
    if isinstance(obj, dict):
        return {k: fix_json_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fix_json_strings(v) for v in obj]
    if isinstance(obj, str):
        return fix_mojibake(obj)
    return obj


def is_system_message(content: str) -> bool:
    return not content or not content.strip() or bool(SYSTEM_RE.search(content.strip()))


def load_thread(thread_dir):
    files = sorted(glob.glob(os.path.join(thread_dir, "message_*.json")))
    if not files:
        return None, None

    participants = None
    messages = []

    for filepath in files:
        with open(filepath, encoding="utf-8") as f:
            data = fix_json_strings(json.load(f))
        if participants is None:
            participants = data.get("participants", [])
        messages.extend(data.get("messages", []))

    return participants, messages


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    written = 0
    skipped_group = 0
    skipped_short = 0

    thread_dirs = sorted(
        d for d in glob.glob(os.path.join(RAW_INBOX, "*")) if os.path.isdir(d)
    )
    total = len(thread_dirs)

    with open(OUTPUT, "w", encoding="utf-8") as out:
        for idx, thread_dir in enumerate(thread_dirs, 1):
            participants, messages = load_thread(thread_dir)
            if not participants:
                continue

            names = [p["name"].lower() for p in participants]
            if len(names) != 2 or MY_NAME.lower() not in names:
                skipped_group += 1
                continue

            seen = set()
            unique = []
            for m in messages:
                key = (m.get("timestamp_ms"), m.get("sender_name"), m.get("content"))
                if key not in seen:
                    seen.add(key)
                    unique.append(m)

            unique.sort(key=lambda m: m.get("timestamp_ms", 0))

            conv = []
            for m in unique:
                if m.get("photos") or m.get("videos") or m.get("audio_files") or m.get("share"):
                    continue
                content = (m.get("content") or "").strip()
                if is_system_message(content):
                    continue
                role = "assistant" if (m.get("sender_name") or "").lower() == MY_NAME.lower() else "user"
                conv.append({"role": role, "content": content})

            merged = []
            for msg in conv:
                if merged and msg["role"] == merged[-1]["role"]:
                    merged[-1]["content"] += "\n" + msg["content"]
                else:
                    merged.append(msg)

            if len(merged) < 2:
                skipped_short += 1
                continue

            out.write(json.dumps({"messages": merged}, ensure_ascii=False) + "\n")
            written += 1

            if written % 100 == 0:
                print(f"  {written}/{total} conversations saved...")

    print(f"\nDone — {written} conversations → {OUTPUT}")
    print(f"Skipped: {skipped_group} group chats, {skipped_short} too short")


if __name__ == "__main__":
    main()
