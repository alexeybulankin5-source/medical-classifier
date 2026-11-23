# -*- coding: utf-8 -*-
"""Единый инференс-пайплайн MedBot: обучает ансамбль и отдаёт predict(text).
Источник правды для всех экзаменов; сюда же переведём app.py."""
import csv, random, re, warnings
from collections import Counter, defaultdict
import numpy as np
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

ENCODER_NAME = "intfloat/multilingual-e5-small"
CLIN = re.compile(r"\b(гипо|гипер|ит\b|синдром|недостаточн|обострени|рецидив|хроническ|терапи|патологи)\w*", re.IGNORECASE)
_CACHE = {}

def _norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def build():
    if _CACHE:
        return _CACHE
    rows = list(csv.reader(open("dataset_full.csv", encoding="utf-8")))
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
    Xtr = X[train]; train_cls = [labels[i] for i in train]
    idxc_by = defaultdict(list)
    for k, cc in enumerate(train_cls): idxc_by[cc].append(k)
    rcls = list(router.classes_)

    def predict(text):
        q = _norm(np.asarray(sbert.encode(["query: " + text[:400]], convert_to_numpy=True), dtype=np.float32))[0]
        rw = router.predict_proba(q[None, :])[0]
        proba = np.zeros(len(all_cls))
        colj = {c: j for j, c in enumerate(all_cls)}
        for b, clf in models.items():
            p = clf.predict_proba(q[None, :])[0]
            w = rw[rcls.index(b)]
            for c_i, c in enumerate(clf.classes_):
                proba[colj[c]] += w * p[c_i]
        cosmax = np.zeros(len(all_cls))
        for j, c in enumerate(all_cls):
            if idxc_by.get(c):
                cosmax[j] = (q @ Xtr[idxc_by[c]].T).max()
        final = 0.35 * proba + 0.65 * cosmax
        order = np.argsort(-final)
        top5 = [all_cls[j] for j in order[:5]]
        return top5[0], float(final[order[0]]), top5

    _CACHE.update(predict=predict, texts=texts, labels=labels, test_idx=sorted(test))
    return _CACHE
