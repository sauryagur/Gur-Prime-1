import glob
import json
import os
import re

RAW_INBOX = "data/raw-insta/your_instagram_activity/messages/inbox"
OUTPUT = "data/insta-conversations.jsonl"
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
    try:
        return text.encode("latin-1").decode("utf-8")
    except Exception:
        try:
            return text.encode("latin-1").decode("utf-8", errors="replace")
        except Exception:
            return text


def fix_json_strings(obj):
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


def message_to_content(msg):
    content = (msg.get("content") or "").strip()

    if content:
        return content

    if msg.get("photos"):
        return "<photo>"

    if msg.get("videos"):
        return "<video>"

    if msg.get("audio_files"):
        return "<audio>"

    if msg.get("share"):
        return "<shared_post>"

    if msg.get("gifs"):
        return "<gif>"

    if msg.get("sticker"):
        return "<sticker>"

    return None


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
                key = (
                    m.get("timestamp_ms"),
                    m.get("sender_name"),
                    m.get("content"),
                )

                if key in seen:
                    continue

                seen.add(key)
                unique.append(m)

            unique.sort(key=lambda m: m.get("timestamp_ms", 0))

            conv = []

            for m in unique:
                sender = (m.get("sender_name") or "").lower()

                role = "saurya" if sender == MY_NAME.lower() else "friend"

                content = message_to_content(m)

                if content is None:
                    continue

                if is_system_message(content):
                    continue

                conv.append(
                    {
                        "role": role,
                        "content": content,
                        "timestamp_ms": m.get("timestamp_ms"),
                    }
                )

            if len(conv) < 2:
                skipped_short += 1
                continue

            out.write(
                json.dumps(
                    {
                        "messages": conv,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            written += 1

            if written % 100 == 0:
                print(f"  {written}/{total} conversations saved...")

    print(f"\nDone — {written} conversations → {OUTPUT}")
    print(f"Skipped: {skipped_group} group chats, {skipped_short} too short")


if __name__ == "__main__":
    main()
