# -*- coding: utf-8 -*-
"""Единый инференс MedBot: ансамбль + score-вектор для app.py.
Единственный источник правды для экзаменов и прода."""
import csv, hashlib, os, pickle, random, re, warnings
from collections import Counter, defaultdict
import numpy as np
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

ENCODER_NAME = "intfloat/multilingual-e5-small"
CLIN = re.compile(r"\b(гипо|гипер|ит\b|синдром|недостаточн|обострени|рецидив|хроническ|терапи|патологи)\w*", re.IGNORECASE)
PKL = "ensemble.pkl"
_CACHE = {}

MIN_SCORE = 0.55
MIN_MARGIN = 0.08

# --- гейт опасности: маркеры скорой помощи (согласованы с _REDFLAG_RULES в app.py) ---
_DANGER_ANY = (
    "перекос", "перекош", "речь нарушилась", "не могу говорить", "слова не выговариваю",
    "онемела рука", "онемела вся", "не слушается", "парализова",
    "потеря сознания", "обморок", "потерял зрение", "потеряла зрение",
    "не могу вдохнуть", "задыхаюсь внезапно", "посинели губ",
    "рвет кровью", "рвота с кровью", "кровь в рвоте", "черный стул",
    "кофейной гущей", "жгучая боль за грудиной", "звездочк", "сыпь не бледнеет",
    "не могу опустить подбородок",
)
_DANGER_PAIRS = (
    ("сжима", ("холодный пот", "отдаёт в руку", "отдает в руку", "за грудиной", "больше 20 минут")),
    ("давит в груди", ("холодный пот", "отдаёт в руку", "отдает в руку", "больше 20 минут")),
    ("боль за грудиной", ("холодный пот", "отдаёт в руку", "отдает в руку", "больше 20 минут")),
    ("ригидность", ("температур",)),
    ("светобоязн", ("температур",)),
    ("внезапная одышка", ("кровохарканье", "после перелет", "долго лежал", "долго лежала")),
    ("стало тяжело дышать", ("кровохарканье", "после перелет", "долго лежал", "долго лежала")),
    ("не хватает воздуха в покое", ("кровохарканье", "после перелет", "долго лежал", "долго лежала")),
    ("из заднего прохода", ("крови", "кровь", "слабост", "головокруж", "бледн")),
    ("задержка месячных", ("живот", "боли", "болит", "внизу", "тянет", "кровянист", "мажет")),
    ("месячных нет", ("живот", "боли", "болит", "внизу", "тянет", "кровянист", "мажет")),
    ("справа внизу живота", ("тошнит", "тошнота", "рвота", "температур", "нет аппетита")),
    ("правый низ живота", ("тошнит", "тошнота", "рвота", "температур")),
    ("отек", ("горл", "дыш", "язык", "губ", "глотк")),
)

def _norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def _ds_md5():
    return hashlib.md5(open("dataset_full.csv", "rb").read()).hexdigest()[:8]

def _final_vec(q, C):
    models, router, all_cls = C["models"], C["router"], C["all_cls"]
    rcls, colj = C["rcls"], C["colj"]
    Xtr, train_cls, idxc_by = C["Xtr"], C["train_cls"], C["idxc_by"]
    rw = router.predict_proba(q[None, :])[0]
    proba = np.zeros(len(all_cls))
    for b, clf in models.items():
        p = clf.predict_proba(q[None, :])[0]
        w = rw[rcls.index(b)]
        for c_i, c in enumerate(clf.classes_):
            proba[colj[c]] += w * p[c_i]
    cosmax = np.zeros(len(all_cls))
    for j, c in enumerate(all_cls):
        if idxc_by.get(c):
            cosmax[j] = (q @ Xtr[idxc_by[c]].T).max()
    return 0.35 * proba + 0.65 * cosmax

def _encode(text):
    return _norm(np.asarray(_CACHE["sbert"].encode(["query: " + text[:400]],
                convert_to_numpy=True), dtype=np.float32))[0]

def score_vector(text):
    """Вектор score по всем классам (порядок all_cls)."""
    warm()
    return _CACHE["all_cls"], _final_vec(_encode(text), _CACHE)

def get_scores_for_names(text, names):
    """Вектор score, выровненный по порядку names (напр. CLASS_NAMES из npz).
    Классы вне ансамбля получают 0."""
    all_cls, vec = score_vector(text)
    idx = {c: j for j, c in enumerate(all_cls)}
    out = np.zeros(len(names), dtype=np.float32)
    for j, n in enumerate(names):
        k = idx.get(str(n))
        if k is not None:
            out[j] = vec[k]
    return out

def predict(text):
    all_cls, vec = score_vector(text)
    order = np.argsort(-vec)
    top5 = [all_cls[j] for j in order[:5]]
    return top5[0], float(vec[order[0]]), top5

def score_parts(text, sims=None):
    """Гейты решения: danger (скорая) и confident (порог + margin).
    sims — вектор score по CLASS_NAMES; если передан, confident = порог + отрыв."""
    t = (text or "").lower().replace("ё", "е")
    danger = any(w in t for w in _DANGER_ANY)
    if not danger:
        for first, extras in _DANGER_PAIRS:
            if first in t and any(e in t for e in extras):
                danger = True
                break
    confident = None
    if sims is not None:
        s = np.sort(np.asarray(sims, dtype=float))[::-1]
        s1 = float(s[0]) if s.size else 0.0
        s2 = float(s[1]) if s.size > 1 else 0.0
        confident = (s1 >= MIN_SCORE) and (s1 - s2 >= MIN_MARGIN)
    return {"danger": danger, "confident": confident if confident is not None else True}

def build():
    if _CACHE:
        return
    rows = list(csv.reader(open("dataset_full.csv", encoding="utf-8")))  # заголовка в файле нет — первая строка реальная
    texts = [r[0] for r in rows]; labels = [r[1] for r in rows]
    branch = ["mixed" if CLIN.search(t) else "colloquial" for t in texts]
    random.seed(42)
    by_cls = defaultdict(list)
    for i, l in enumerate(labels): by_cls[l].append(i)
    test = set()
    for c, idxs in by_cls.items():
        test.update(random.sample(idxs, max(1, int(0.1 * len(idxs)))))
    train = [i for i in range(len(texts)) if i not in test]
    sbert = SentenceTransformer(ENCODER_NAME)
    X = np.asarray(sbert.encode(["query: " + t for t in texts], convert_to_numpy=True, batch_size=64), dtype=np.float32)
    X = _norm(X)
    counts = Counter(labels[i] for i in train)
    Xb, yb, bb = [], [], []
    for i in train:
        k = 3 if counts[labels[i]] < 40 else 1
        for _ in range(k): Xb.append(X[i]); yb.append(labels[i]); bb.append(branch[i])
    models = {}
    for b in ("colloquial", "mixed"):
        idx = [j for j, v in enumerate(bb) if v == b]
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit([Xb[j] for j in idx], [yb[j] for j in idx])
        models[b] = clf
    router = LogisticRegression(max_iter=2000); router.fit(Xb, bb)
    all_cls = sorted(by_cls)
    colj = {c: j for j, c in enumerate(all_cls)}
    rcls = list(router.classes_)
    idxc_by = defaultdict(list)
    for k, cc in enumerate([labels[i] for i in train]): idxc_by[cc].append(k)
    C = {"md5": _ds_md5(), "models": models, "router": router, "all_cls": all_cls, "colj": colj,
         "rcls": rcls, "Xtr": X[train], "train_cls": [labels[i] for i in train],
         "idxc_by": dict(idxc_by), "test_idx": sorted(test), "sbert": sbert}
    _CACHE.update(C)
    try:
        pickle.dump({k: v for k, v in C.items() if k != "sbert"}, open(PKL, "wb"))
    except Exception as e:
        print("pkl не сохранён:", e)

def warm():
    if _CACHE:
        return
    md5 = _ds_md5()
    if os.path.exists(PKL):
        try:
            obj = pickle.load(open(PKL, "rb"))
            if obj.get("md5") == md5:
                obj["sbert"] = SentenceTransformer(ENCODER_NAME)
                _CACHE.update(obj)
                print("ансамбль загружен из ensemble.pkl (md5 совпал)")
                return
            print("ensemble.pkl устарел (md5 не совпал), пересборка...")
        except Exception:
            print("ensemble.pkl повреждён, пересборка...")
    build()
