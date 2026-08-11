"""
1. MODEL filters spam
2. LLM decides
3. ONLY the LLM can send mail to trash

Priority: brand-new mail first, then clean older inbox mail (capped).
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

POLL = int(os.getenv("POLL_SECONDS", "30"))
# max older emails to clean per cycle (after new mail)
CLEAN_LIMIT = int(os.getenv("CLEAN_LIMIT", "20"))
# max brand-new unread to grab first each cycle
NEW_LIMIT = int(os.getenv("NEW_LIMIT", "10"))
SEEN_FILE = os.path.join(os.path.dirname(__file__), "models", "processed_uids.txt")
LAST_UID_FILE = os.path.join(os.path.dirname(__file__), "models", "last_uid.txt")


def load_seen():
    if not os.path.exists(SEEN_FILE):
        return set()
    with open(SEEN_FILE) as f:
        return {line.strip() for line in f if line.strip()}


def save_seen(seen):
    os.makedirs(os.path.dirname(SEEN_FILE), exist_ok=True)
    with open(SEEN_FILE, "w") as f:
        f.write("\n".join(sorted(seen)))


def load_last_uid():
    if not os.path.exists(LAST_UID_FILE):
        return 0
    try:
        return int(open(LAST_UID_FILE).read().strip() or "0")
    except ValueError:
        return 0


def save_last_uid(uid):
    os.makedirs(os.path.dirname(LAST_UID_FILE), exist_ok=True)
    with open(LAST_UID_FILE, "w") as f:
        f.write(str(uid))


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


def _fetch_one(mail, uid):
    _, fetched = mail.uid("fetch", uid, "(RFC822)")
    if not fetched or not fetched[0] or not isinstance(fetched[0], tuple):
        return None
    raw = fetched[0][1]
    msg = email.message_from_bytes(raw)
    subject = decode(msg.get("Subject"))
    sender = decode(msg.get("From"))
    body = body_of(msg)
    text = f"Subject: {subject}\nFrom: {sender}\n\n{body}"
    return uid, subject, text


def get_emails(mail, seen, last_uid):
    """
    1) NEW mail first: UIDs newer than last_uid (and newest unread)
    2) Then older inbox cleanup, capped by CLEAN_LIMIT
    """
    _, data = mail.uid("search", None, "ALL")
    if not data or not data[0]:
        return [], last_uid

    all_uids = data[0].split()
    newest_uid = int(all_uids[-1])

    # --- priority 1: brand new since last check ---
    new_uids = [u for u in all_uids if int(u) > last_uid]
    # also treat newest unread as new-priority
    _, undata = mail.uid("search", None, "UNSEEN")
    if undata and undata[0]:
        unread = undata[0].split()
        for u in unread[-NEW_LIMIT:]:
            if u not in new_uids:
                new_uids.append(u)

    # newest first, capped
    new_uids = sorted(set(new_uids), key=lambda u: int(u), reverse=True)[:NEW_LIMIT]

    # --- priority 2: clean older mail (not already processed), capped ---
    clean_uids = []
    for u in reversed(all_uids):  # newest → older through inbox
        key = u.decode() if isinstance(u, bytes) else str(u)
        if key in seen:
            continue
        if u in new_uids:
            continue
        clean_uids.append(u)
        if len(clean_uids) >= CLEAN_LIMIT:
            break

    print(
        f"new: {len(new_uids)} | clean: {len(clean_uids)} "
        f"(limits new={NEW_LIMIT}, clean={CLEAN_LIMIT})"
    )

    out = []
    # new first
    for uid in new_uids:
        key = uid.decode() if isinstance(uid, bytes) else str(uid)
        if key in seen:
            continue
        item = _fetch_one(mail, uid)
        if item:
            out.append(item)

    # then cleanup
    for uid in clean_uids:
        item = _fetch_one(mail, uid)
        if item:
            out.append(item)

    return out, newest_uid


def trash(mail, uid):
    """MOVE to Gmail Trash."""
    st, _ = mail.uid("move", uid, "[Gmail]/Trash")
    print(f"  move to trash: {st}")
    if st != "OK":
        mail.uid("copy", uid, "[Gmail]/Trash")
        mail.uid("store", uid, "+FLAGS", r"(\Deleted \Seen)")
        mail.expunge()
        print("  trash fallback: copy+delete")


def llm_should_trash(text):
    """LLM is the only thing allowed to trash. YES = trash, NO = leave."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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


def run_once(model, seen, last_uid):
    mail = connect()
    try:
        emails, newest_uid = get_emails(mail, seen, last_uid)
        if not emails:
            print("nothing to check")
            return seen, max(last_uid, newest_uid)

        print(f"checking {len(emails)} email(s) — new first, then clean")
        print(f"  first up: {emails[0][1][:70]}")

        for uid, subject, text in emails:
            key = uid.decode() if isinstance(uid, bytes) else str(uid)
            conf = spam_confidence(text, model)

            if not is_spam(text, model):
                print(f"model: ok     | {conf:.2f} | {subject[:55]}")
                seen.add(key)
                continue

            print(f"model: Send it to trash | {conf:.2f} | {subject[:45]}")

            if not llm_should_trash(text):
                print("llm:   keep")
                seen.add(key)
                continue

            trash(mail, uid)
            print("llm:   Send it trash and mark it")
            seen.add(key)

        return seen, max(last_uid, newest_uid)
    finally:
        mail.logout()


def main():
    if not MODEL_PATH.exists():
        print("training model...")
        train()

    model = joblib.load(MODEL_PATH)
    seen = load_seen()
    last_uid = load_last_uid()

    # first run: don't treat entire inbox as "new" — start from current newest
    if last_uid == 0:
        mail = connect()
        try:
            _, data = mail.uid("search", None, "ALL")
            if data and data[0]:
                last_uid = int(data[0].split()[-1])
                save_last_uid(last_uid)
                print(f"starting from latest uid {last_uid} (won't flood on old mail as 'new')")
        finally:
            mail.logout()

    print(f"model first (>{spam_threshold():.0%}), then LLM. every {POLL}s")
    print(f"priority: NEW mail first (up to {NEW_LIMIT}), then clean older (up to {CLEAN_LIMIT}/cycle)")

    while True:
        try:
            seen, last_uid = run_once(model, seen, last_uid)
            save_seen(seen)
            save_last_uid(last_uid)
        except KeyboardInterrupt:
            print("stopped")
            break
        except Exception as e:
            print("error:", e)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
