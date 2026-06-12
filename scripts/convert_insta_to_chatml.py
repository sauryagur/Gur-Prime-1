import glob
import json
import os
import re
import traceback
from collections import defaultdict
from datetime import datetime


RAW_INBOX = "data/raw-insta/your_instagram_activity/messages/inbox"
OUTPUT = "data/insta-conversations.jsonl"
STATS_FILE = "data/insta-dataset-stats.json"
MY_NAME = "Saurya"

MIN_MESSAGES = 2
BURST_MERGE_WINDOW_MS = 90_000


SYSTEM_EXACT = {
    "",
    "on liked",
    "you missed a video call",
    "you missed a voice call",
    "this message has been deleted",
    "this message was deleted",
    "message deleted",
}


SYSTEM_PATTERNS = [
    r"^sent an attachment\.?$",
    r"^shared a (story|post|reel|video|photo)\.?$",
    r"^liked a message\.?$",
    r"^reacted .{1,10} to (your|a) message\.?$",
    r"^(video|voice) call,?\s+\d",
    r"^(started|ended|missed|declined) (a |the )?(video |voice )?call\.?$",
    r"^(created|named|changed|removed|added|left|joined) (the |a )?group",
    r"^removed .+ from (the |a )?group",
    r"^added .+ to (the |a )?group",
    r"^.+ is now in your close friends$",
    r"^.+ started following you$",
    r"^(set messages to disappear|turned off disappearing messages)\.?$",
    r"^sent a (voice message|link|location)\.?$",
]

SYSTEM_RE = re.compile("|".join(SYSTEM_PATTERNS), re.IGNORECASE)


LONE_EMOJI_RE = re.compile(
    r"^[\U00010000-\U0010ffff"
    r"\U0001F300-\U0001F9FF"
    r"\u2600-\u27BF"
    r"\uFE00-\uFE0F"
    r"\u200d"
    r"\s]+$"
)


PII_PATTERNS = [
    (re.compile(r"https?://\S+", re.IGNORECASE), "<url>"),
    (re.compile(r"www\.\S+\.\S+", re.IGNORECASE), "<url>"),
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), "<email>"),
    (re.compile(r"(\+91[\s\-]?[6-9]\d{9}|\b[6-9]\d{4}\s\d{5}\b)"), "<phone>"),
    (
        re.compile(
            r"[a-zA-Z0-9.\-_]+@(upi|paytm|ybl|okicici|okhdfcbank|"
            r"okaxis|ibl|axl|hdfcbank|sbi|icici|kotak)\b",
            re.IGNORECASE,
        ),
        "<upi_id>",
    ),
    (re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{4}-\d{4}-\d{4}\b"), "<id_number>"),
    (re.compile(r"(?<!\d)\d{12,}(?!\d)"), "<id_number>"),
]


def scrub_pii(text: str) -> tuple[str, list[str]]:
    """
    Replace PII in text with placeholder tokens.
    Returns (scrubbed_text, list_of_what_was_replaced).
    """
    redacted = []
    for pattern, token in PII_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            redacted.append(f"{token}×{len(matches)}")
            text = pattern.sub(token, text)
    return text, redacted


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
    stripped = content.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    if lower in SYSTEM_EXACT:
        return True

    if len(stripped.split()) > 25:
        return False
    if SYSTEM_RE.search(stripped):
        return True
    if LONE_EMOJI_RE.match(stripped):
        return True
    return False


def find_thread_dirs(inbox_path: str) -> list[str]:
    """Find all 1-level-deep subdirectories containing message_N.json files."""
    thread_dirs = set()
    for root, dirs, files in os.walk(inbox_path):
        rel = os.path.relpath(root, inbox_path)
        parts = rel.split(os.sep)
        if rel != "." and len(parts) > 1:
            continue
        if any(re.match(r"message_\d+\.json$", f) for f in files):
            thread_dirs.add(root)
    return sorted(thread_dirs)


def load_thread(thread_dir: str):
    """Load and merge all message_N.json files in a thread directory."""
    files = sorted(
        glob.glob(os.path.join(thread_dir, "message_*.json")),
        key=lambda p: int(re.search(r"message_(\d+)\.json$", p).group(1)),
    )
    if not files:
        return None, [], []

    participants = None
    messages = []
    loaded_files = []

    for filepath in files:
        print(f"    [FILE] Reading {filepath}")
        try:
            with open(filepath, encoding="utf-8") as f:
                data = fix_json_strings(json.load(f))

            if participants is None:
                participants = data.get("participants", [])

            batch = data.get("messages", [])
            print(f"           → {len(batch)} raw messages")
            messages.extend(batch)
            loaded_files.append(filepath)

        except json.JSONDecodeError as e:
            print(f"    [ERROR] JSON decode error in {filepath}: {e}")
        except UnicodeDecodeError as e:
            print(f"    [ERROR] Encoding error in {filepath}: {e}")
        except Exception as e:
            print(f"    [ERROR] Unexpected error reading {filepath}: {e}")
            traceback.print_exc()

    return participants, messages, loaded_files


def message_to_content(msg: dict) -> str | None:
    content = (msg.get("content") or "").strip()

    if content and len(content) > 3:
        return content

    if msg.get("photos"):
        return "<photo>"
    if msg.get("videos"):
        return "<video>"
    if msg.get("audio_files"):
        return "<audio>"
    if msg.get("share"):
        share = msg.get("share", {})
        if isinstance(share, dict) and share.get("title"):
            return f"<shared_post: {share['title']}>"
        return "<shared_post>"
    if msg.get("gifs"):
        return "<gif>"
    if msg.get("sticker"):
        return "<sticker>"

    if content:
        return content

    return None


def merge_bursts(conv: list[dict], window_ms: int) -> tuple[list[dict], int]:
    """
    Merge consecutive same-sender messages within window_ms, BUT:
    - Never merge across a sentence boundary if total would exceed 40 words
    - Never merge if second message starts with a conjunction (preserves "but", "actually")
    """
    if not conv:
        return conv, 0

    CONJUNCTION_STARTS = (
        "but ",
        "and ",
        "actually ",
        "wait ",
        "no ",
        "yes ",
        "oh ",
        "btw ",
        "also ",
        "though ",
        "however ",
    )
    MAX_MERGED_WORDS = 40

    merged = [conv[0].copy()]
    n_merges = 0

    for msg in conv[1:]:
        prev = merged[-1]
        same_role = msg["role"] == prev["role"]
        time_delta = msg.get("timestamp_ms", 0) - prev.get("timestamp_ms", 0)
        close_in_time = time_delta <= window_ms

        starts_with_conj = (
            msg["content"].lstrip().lower().startswith(CONJUNCTION_STARTS)
        )

        combined_words = len(prev["content"].split()) + len(msg["content"].split())

        if (
            same_role
            and close_in_time
            and not starts_with_conj
            and combined_words <= MAX_MERGED_WORDS
        ):
            prev["content"] += "\n" + msg["content"]
            prev["timestamp_ms"] = msg.get("timestamp_ms", prev["timestamp_ms"])
            n_merges += 1
        else:
            merged.append(msg.copy())

    return merged, n_merges


def find_first_user_turn(conv: list[dict]) -> int:
    """Return index of first 'friend' (user) turn, or -1 if none."""
    for i, msg in enumerate(conv):
        if msg["role"] == "friend":
            return i
    return -1


def trim_to_valid_start(conv: list[dict]) -> list[dict] | None:
    """
    If conversation starts with assistant, trim leading assistant turns
    until we find a user turn. Returns None if no user turn exists.
    """
    first_user_idx = find_first_user_turn(conv)
    if first_user_idx == -1:
        return None
    if first_user_idx == 0:
        return conv

    return conv[first_user_idx:]


class Stats:
    def __init__(self):
        self.total_convos = 0
        self.total_messages = 0
        self.total_tokens_approx = 0
        self.saurya_messages = 0
        self.friend_messages = 0
        self.skipped_group = 0
        self.skipped_short = 0
        self.skipped_no_me = 0
        self.errors = 0
        self.pii_redactions = defaultdict(int)
        self.convo_lengths = []
        self.saurya_msg_lengths = []
        self.friend_msg_lengths = []
        self.system_dropped = 0
        self.empty_dropped = 0
        self.burst_merges = 0
        self.duplicate_msgs = 0
        self.skipped_bad_start = 0

    def record_convo(self, conv: list[dict]):
        self.total_convos += 1
        self.total_messages += len(conv)
        self.convo_lengths.append(len(conv))
        for msg in conv:
            words = len(msg["content"].split())
            self.total_tokens_approx += words
            if msg["role"] == "saurya":
                self.saurya_messages += 1
                self.saurya_msg_lengths.append(words)
            else:
                self.friend_messages += 1
                self.friend_msg_lengths.append(words)

    def record_pii(self, redacted: list[str]):
        for entry in redacted:
            token, _, count = entry.partition("×")
            self.pii_redactions[token] += int(count or 1)

    def _avg(self, lst):
        return round(sum(lst) / len(lst), 2) if lst else 0

    def _median(self, lst):
        if not lst:
            return 0
        s = sorted(lst)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2

    def to_dict(self) -> dict:
        return {
            "conversations": {
                "written": self.total_convos,
                "skipped_group": self.skipped_group,
                "skipped_short": self.skipped_short,
                "skipped_no_me": self.skipped_no_me,
                "errors": self.errors,
            },
            "messages": {
                "total": self.total_messages,
                "by_saurya": self.saurya_messages,
                "by_friend": self.friend_messages,
                "role_balance": round(self.saurya_messages / self.total_messages, 3)
                if self.total_messages
                else 0,
                "system_dropped": self.system_dropped,
                "empty_dropped": self.empty_dropped,
                "duplicates_removed": self.duplicate_msgs,
                "burst_merges": self.burst_merges,
            },
            "tokens_approx": {
                "total": self.total_tokens_approx,
                "avg_per_message": self._avg(
                    self.saurya_msg_lengths + self.friend_msg_lengths
                ),
                "avg_saurya_msg": self._avg(self.saurya_msg_lengths),
                "avg_friend_msg": self._avg(self.friend_msg_lengths),
            },
            "conversation_lengths": {
                "avg": self._avg(self.convo_lengths),
                "median": self._median(self.convo_lengths),
                "min": min(self.convo_lengths) if self.convo_lengths else 0,
                "max": max(self.convo_lengths) if self.convo_lengths else 0,
            },
            "pii_redactions": dict(self.pii_redactions),
        }

    def print_summary(self):
        d = self.to_dict()
        w = 60
        print(f"\n{'═' * w}")
        print(f"  DATASET BUILD COMPLETE")
        print(f"{'═' * w}")
        c = d["conversations"]
        print(f"  Conversations written : {c['written']}")
        print(f"  Skipped — group chat  : {c['skipped_group']}")
        print(f"  Skipped — too short   : {c['skipped_short']}")
        print(f"  Skipped — not me      : {c['skipped_no_me']}")
        print(f"  Errors                : {c['errors']}")
        print(f"{'─' * w}")
        m = d["messages"]
        print(f"  Total messages        : {m['total']}")
        print(
            f"  → Saurya              : {m['by_saurya']}  "
            f"({d['messages']['role_balance'] * 100:.1f}% of total)"
        )
        print(f"  → Friend              : {m['by_friend']}")
        print(f"  System msgs dropped   : {m['system_dropped']}")
        print(f"  Empty/media dropped   : {m['empty_dropped']}")
        print(f"  Duplicates removed    : {m['duplicates_removed']}")
        print(f"  Burst merges          : {m['burst_merges']}")
        print(f"{'─' * w}")
        t = d["tokens_approx"]
        print(f"  Total words (approx)  : {t['total']:,}")
        print(f"  Avg words/msg         : {t['avg_per_message']}")
        print(f"  Avg Saurya msg length : {t['avg_saurya_msg']} words")
        print(f"  Avg Friend msg length : {t['avg_friend_msg']} words")
        print(f"{'─' * w}")
        cl = d["conversation_lengths"]
        print(f"  Convo length avg/med  : {cl['avg']} / {cl['median']}")
        print(f"  Convo length min/max  : {cl['min']} / {cl['max']}")
        if d["pii_redactions"]:
            print(f"{'─' * w}")
            print(f"  PII redacted:")
            for token, count in sorted(d["pii_redactions"].items()):
                print(f"    {token:<18} {count:>6} instance(s)")
        print(f"{'═' * w}\n")


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    stats = Stats()

    print(f"[SCAN] Inbox: {RAW_INBOX}")
    thread_dirs = find_thread_dirs(RAW_INBOX)
    total = len(thread_dirs)
    print(f"[SCAN] Found {total} thread directories\n")

    with open(OUTPUT, "w", encoding="utf-8") as out:
        for idx, thread_dir in enumerate(thread_dirs, 1):
            print(f"[{idx}/{total}] {thread_dir}")

            try:
                participants, messages, loaded_files = load_thread(thread_dir)
            except Exception as e:
                print(f"  [ERROR] Failed to load thread: {e}")
                traceback.print_exc()
                stats.errors += 1
                continue

            if not participants:
                print(f"  [SKIP] No participants metadata found")
                stats.errors += 1
                continue

            names = [p["name"] for p in participants]
            names_lower = [n.lower() for n in names]
            print(f"  [INFO] Participants ({len(names)}): {', '.join(names)}")
            print(f"  [INFO] Raw messages across all files: {len(messages)}")

            if MY_NAME.lower() not in names_lower:
                print(f"  [SKIP] '{MY_NAME}' not a participant")
                stats.skipped_no_me += 1
                continue

            if len(names) != 2:
                print(f"  [SKIP] Group chat ({len(names)} participants)")
                stats.skipped_group += 1
                continue

            seen = set()
            unique = []
            for m in messages:
                key = (m.get("timestamp_ms"), m.get("sender_name"), m.get("content"))
                if key not in seen:
                    seen.add(key)
                    unique.append(m)

            dupes = len(messages) - len(unique)
            if dupes:
                print(f"  [INFO] Removed {dupes} duplicate messages")
                stats.duplicate_msgs += dupes

            unique.sort(key=lambda m: m.get("timestamp_ms", 0))

            conv = []
            skipped_system = 0
            skipped_empty = 0

            for m in unique:
                sender = (m.get("sender_name") or "").lower()
                role = "saurya" if sender == MY_NAME.lower() else "friend"
                content = message_to_content(m)

                if content is None:
                    skipped_empty += 1
                    continue

                if is_system_message(content):
                    skipped_system += 1
                    continue

                conv.append(
                    {
                        "role": role,
                        "content": content,
                        "timestamp_ms": m.get("timestamp_ms"),
                    }
                )

            stats.system_dropped += skipped_system
            stats.empty_dropped += skipped_empty
            print(
                f"  [INFO] After noise filter: {len(conv)} messages "
                f"(dropped {skipped_system} system, {skipped_empty} empty/media-only)"
            )

            conv, n_merges = merge_bursts(conv, BURST_MERGE_WINDOW_MS)
            stats.burst_merges += n_merges
            if n_merges:
                print(
                    f"  [INFO] Burst-merged {n_merges} message(s) → {len(conv)} turns"
                )

            conv = trim_to_valid_start(conv)
            if conv is None:
                print(f"  [SKIP] No 'friend' (user) turn found after filtering")
                stats.skipped_no_me += 1
                continue

            if len(conv) < MIN_MESSAGES:
                print(
                    f"  [SKIP] Too short after merging ({len(conv)} turns, need {MIN_MESSAGES})"
                )
                stats.skipped_short += 1
                continue

            out.write(json.dumps({"messages": conv}, ensure_ascii=False) + "\n")
            stats.record_convo(conv)
            print(f"  [OK]  Saved {len(conv)} turns")

    with open(STATS_FILE, "w", encoding="utf-8") as sf:
        json.dump(stats.to_dict(), sf, indent=2, ensure_ascii=False)
    print(f"[STATS] Written to {STATS_FILE}")

    stats.print_summary()


if __name__ == "__main__":
    main()
