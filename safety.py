"""Красные флаги MedBot Duo. Источник правил — red_flags.yaml."""
import yaml

FLAGS_PATH = "red_flags.yaml"

def load_flags(path=FLAGS_PATH):
    return yaml.safe_load(open(path, encoding="utf-8"))

def assess(text, flags_doc=None):
    doc = load_flags() if flags_doc is None else flags_doc
    t = (text or "").lower()
    hits = []
    for fl in doc.get("flags", []):
        if any(x in t for x in fl.get("except", [])):
            continue  # явное отрицание признака — флаг не срабатывает
        groups = fl["patterns"]
        if fl.get("match", "all") == "any":
            ok = any(any(p in t for p in g) for g in groups)
        else:
            ok = all(any(p in t for p in g) for g in groups)
        if ok:
            hits.append(fl)
    return hits

def is_emergency(text, flags_doc=None):
    return bool(assess(text, flags_doc))

def urgency_hint(text, flags_doc=None):
    hits = assess(text, flags_doc)
    if not hits:
        return None
    return "СРОЧНО: " + "; ".join(f["message"] for f in hits)
