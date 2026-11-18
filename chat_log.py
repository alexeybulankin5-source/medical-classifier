"""Лог решений чата. Каждая пара вопрос-ответ пишется в chat_log.csv."""
import csv, os, re, datetime
from safety import assess

LOG = "chat_log.csv"
EMS = re.compile(r"скор|103|112|неотложн", re.I)

def log_turn(user_msg, assistant_msg, case_ctx=None, model_version="unknown"):
    hits = [f["id"] for f in assess(user_msg)]
    row = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_version": model_version,
        "case_urgency": (case_ctx or {}).get("urgency", ""),
        "case_top": (case_ctx or {}).get("top", ""),
        "red_flags": "|".join(hits),
        "user_msg": (user_msg or "")[:500],
        "assistant_msg": (assistant_msg or "")[:2000],
        "assistant_calls_ems": bool(EMS.search(assistant_msg or "")),
    }
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    return row
