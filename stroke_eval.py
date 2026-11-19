# -*- coding: utf-8 -*-
import csv, random, re, warnings
from collections import Counter, defaultdict
import numpy as np
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

rows = list(csv.reader(open("dataset_full.csv", encoding="utf-8")))
texts = [r[0] for r in rows]; labels = [r[1] for r in rows]
CLIN = re.compile(r"\b(гипо|гипер|ит\b|синдром|недостаточн|обострени|рецидив|хроническ|терапи|патологи)\w*", re.IGNORECASE)
branch = ["mixed" if CLIN.search(t) else "colloquial" for t in texts]
random.seed(42)
by_cls = defaultdict(list)
for i, l in enumerate(labels): by_cls[l].append(i)
test = set()
for c, idxs in by_cls.items():
    test.update(random.sample(idxs, max(1, int(0.1 * len(idxs)))))
train = [i for i in range(len(texts)) if i not in test]
sbert = SentenceTransformer("intfloat/multilingual-e5-small")
X = np.asarray(sbert.encode(["query: " + t for t in texts], convert_to_numpy=True, batch_size=64), dtype=np.float32)
X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
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

# --- пробник ---
probe = list(csv.DictReader(open("stroke_probe.csv", encoding="utf-8"), delimiter="|"))
Xp = np.asarray(sbert.encode(["query: " + r["text"] for r in probe], convert_to_numpy=True), dtype=np.float32)
Xp /= (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-9)
rw = router.predict_proba(Xp); rcls = list(router.classes_)
proba = np.zeros((len(probe), len(all_cls)))
colj = {c: j for j, c in enumerate(all_cls)}
for b, clf in models.items():
    p = clf.predict_proba(Xp)
    w = rw[:, rcls.index(b)]
    for c_i, c in enumerate(clf.classes_):
        proba[:, colj[c]] += w * p[:, c_i]
Xtr = X[train]; train_cls = [labels[i] for i in train]
cosmax = np.zeros((len(probe), len(all_cls)))
for j, c in enumerate(all_cls):
    idxc = [k for k, cc in enumerate(train_cls) if cc == c]
    if idxc: cosmax[:, j] = (Xp @ Xtr[idxc].T).max(axis=1)
final = 0.35 * proba + 0.65 * cosmax
top5 = np.argsort(-final, axis=1)[:, :5]

ok1 = ok5 = fp = 0
print("=" * 70)
for i, r in enumerate(probe):
    pred = all_cls[int(top5[i][0])]
    t5 = [all_cls[j] for j in top5[i]]
    hit1 = r["label"] == "инсульт" and pred == "инсульт"
    hit5 = r["label"] == "инсульт" and "инсульт" in t5
    fp_ = r["label"] == "не инсульт" and pred == "инсульт"
    ok1 += hit1; ok5 += hit5; fp += fp_
    mark = "top1" if hit1 else ("top5" if hit5 else ("FP!" if fp_ else "ok"))
    print(f"  [{mark:4}] {r['label']:9} -> {pred[:35]:35} | {final[i].max():.3f} | {r['text'][:70]}")
print("=" * 70)
pos = sum(1 for r in probe if r["label"] == "инсульт")
neg = len(probe) - pos
print(f"СУДЬЁЙ ПРИНЯТО top-1: {ok1} из {pos} позитивов")
print(f"           top-5: {ok5} из {pos} позитивов")
print(f"ЛОЖНЫХ ТРЕВОГ: {fp} из {neg} отрицательных")
