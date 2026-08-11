"""
1. MODEL filters spam
2. LLM decides
3. ONLY the LLM can send mail to trash (and mark spam)
"""

import email
import imaplib
import os
import time
from email.header import decode_header, make_header

import joblib
from dotenv import load_dotenv
from openai import OpenAI

from machine_learning import MODEL_PATH, is_spam, spam_confidence, spam_threshold, train

load_dotenv()

POLL = int(os.getenv("POLL_SECONDS", "60"))
# how many newest inbox emails to scan each cycle (ALL if smaller)
MAIL_LIMIT = int(os.getenv("MAIL_LIMIT", "100"))
SEEN_FILE = os.path.join(os.path.dirname(__file__), "models", "processed_uids.txt")


def load_seen():
    if not os.path.exists(SEEN_FILE):
        return set()
    with open(SEEN_FILE) as f:
        return {line.strip() for line in f if line.strip()}


def save_seen(seen):
    os.makedirs(os.path.dirname(SEEN_FILE), exist_ok=True)
    with open(SEEN_FILE, "w") as f:
        f.write("\n".join(sorted(seen)))


def decode(s):
    if not s:
        return ""
    try:
        return str(make_header(decode_header(s)))
    except Exception:
        return s


def body_of(msg):
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                raw = part.get_payload(decode=True) or b""
                return raw.decode(part.get_content_charset() or "utf-8", errors="replace")
        return ""
    raw = msg.get_payload(decode=True) or b""
    return raw.decode(msg.get_content_charset() or "utf-8", errors="replace")


def connect():
    mail = imaplib.IMAP4_SSL(os.getenv("IMAP_HOST", "imap.gmail.com"))
    mail.login(os.getenv("EMAIL_ADDRESS"), os.getenv("EMAIL_PASSWORD"))
    mail.select("INBOX")
    return mail


def get_emails(mail, limit=MAIL_LIMIT):
    """Fetch newest inbox emails (read + unread), latest first."""
    _, data = mail.uid("search", None, "ALL")
    if not data or not data[0]:
        return []

    uids = data[0].split()
    # newest last in IMAP search → take the latest N, newest first
    uids = list(reversed(uids[-limit:]))

    out = []
    for uid in uids:
        _, fetched = mail.uid("fetch", uid, "(RFC822)")
        if not fetched or not fetched[0] or not isinstance(fetched[0], tuple):
            continue
        raw = fetched[0][1]
        msg = email.message_from_bytes(raw)
        subject = decode(msg.get("Subject"))
        sender = decode(msg.get("From"))
        body = body_of(msg)
        text = f"Subject: {subject}\nFrom: {sender}\n\n{body}"
        out.append((uid, subject, text))
    return out


def trash(mail, uid):
    """Label unwanted, then MOVE to Gmail Trash (not Spam — that hides it from Trash)."""
    # apply custom label while still in Inbox
    st, _ = mail.uid("store", uid, "+X-GM-LABELS", '("unwanted")')
    print(f"  unwanted label: {st}")

    # IMAP MOVE is supported by Gmail — puts it in Trash for real
    st, _ = mail.uid("move", uid, "[Gmail]/Trash")
    print(f"  move to trash: {st}")
    if st != "OK":
        # fallback: copy + delete from inbox
        mail.uid("copy", uid, "[Gmail]/Trash")
        mail.uid("store", uid, "+FLAGS", r"(\Deleted \Seen)")
        mail.expunge()
        print("  trash fallback: copy+delete")


def llm_should_trash(text):
    """LLM is the only thing allowed to trash. YES = trash, NO = leave."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # gpt-4o-mini: reliable YES/NO, still pennies/month (gpt-5-nano was returning empty)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    res = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You confirm spam. Reply with exactly YES or NO.\n"
                    "YES = spam / phishing / malware / scam / get-rich-quick / junk → trash it.\n"
                    "NO = real personal, work, receipts, bills, security alerts."
                ),
            },
            {"role": "user", "content": text[:2000]},
        ],
        temperature=0,
        max_tokens=5,
    )
    answer = (res.choices[0].message.content or "").strip().upper()
    print(f"  llm raw ({model}): {answer[:80]!r}")
    for word in answer.replace(".", " ").replace(",", " ").split():
        if word == "YES":
            return True
        if word == "NO":
            return False
    return answer.startswith("YES")


def run_once(model, seen):
    mail = connect()
    try:
        emails = get_emails(mail)
        if not emails:
            print("no mail in inbox")
            return seen

        print(f"scanning {len(emails)} latest emails...")
        for uid, subject, text in emails:
            key = uid.decode() if isinstance(uid, bytes) else str(uid)
            if key in seen:
                continue

            conf = spam_confidence(text, model)

            # 1) MODEL first — never trashes
            if not is_spam(text, model):
                print(f"model: ok     | {conf:.2f} | {subject[:55]}")
                seen.add(key)
                continue

            print(f"model: Send it to trash | {conf:.2f} | {subject[:45]}")

            # 2) LLM second — only LLM can trash
            if not llm_should_trash(text):
                print("llm:   keep")
                seen.add(key)
                continue

            trash(mail, uid)
            print("llm:   Send it trash and mark it")
            seen.add(key)
        return seen
    finally:
        mail.logout()


def main():
    if not MODEL_PATH.exists():
        print("training model...")
        train()

    model = joblib.load(MODEL_PATH)
    seen = load_seen()
    print(f"model first (>{spam_threshold():.0%}), then LLM trashes. every {POLL}s")
    print(f"reads latest {MAIL_LIMIT} inbox emails (read + unread)")
    while True:
        try:
            seen = run_once(model, seen)
            save_seen(seen)
        except KeyboardInterrupt:
            print("stopped")
            break
        except Exception as e:
            print("error:", e)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
