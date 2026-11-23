# -*- coding: utf-8 -*-
"""Срез holdout по набору классов: python slice_eval.py глаз [класс ...]"""
import csv, random, re, sys, warnings
from collections import Counter, defaultdict
import numpy as np
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

TARGETS = {
    "глаз": ["конъюнктивит", "аллергический конъюнктивит", "синдром сухого глаза",
             "ячмень", "глаукома", "кератит", "увеит", "мигрень", "блефарит"],
}
args = sys.argv[1:]
name = args[0] if args and args[0] in TARGETS else "глаз"
CLASSES = set(TARGETS.get(name) if name in TARGETS else (args or TARGETS["глаз"]))

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
test_list = sorted(test)
Xt = X[test_list]
rw = router.predict_proba(Xt); rcls = list(router.classes_)
proba = np.zeros((len(test_list), len(all_cls)))
colj = {c: j for j, c in enumerate(all_cls)}
for b, clf in models.items():
    p = clf.predict_proba(Xt)
    w = rw[:, rcls.index(b)]
    for c_i, c in enumerate(clf.classes_):
        proba[:, colj[c]] += w * p[:, c_i]
Xtr = X[train]; train_cls = [labels[i] for i in train]
cosmax = np.zeros((len(test_list), len(all_cls)))
for j, c in enumerate(all_cls):
    idxc = [k for k, cc in enumerate(train_cls) if cc == c]
    if idxc: cosmax[:, j] = (Xt @ Xtr[idxc].T).max(axis=1)
final = 0.35 * proba + 0.65 * cosmax
pred = [all_cls[j] for j in final.argmax(axis=1)]
true = [labels[i] for i in test_list]

idxs = [i for i, t in enumerate(true) if t in CLASSES]
ok = sum(1 for i in idxs if pred[i] == true[i])
print(f"СРЕЗ «{name}»: {len(idxs)} строк holdout, top-1: {ok} ({ok/len(idxs)*100:.0f}%)")
per = defaultdict(lambda: [0, 0])
for i in idxs: per[true[i]][0] += pred[i] == true[i]; per[true[i]][1] += 1
print("\nпо классам (верно/всего):")
for c in sorted(per): print(f"  {c:30} {per[c][0]:2}/{per[c][1]:2}")
print("\nошибки:")
for i in idxs:
    if pred[i] != true[i]:
        print(f"  [{true[i]} -> {pred[i]} | {final[i].max():.3f}] {texts[i][:80]}")
pairs = Counter((true[i], pred[i]) for i in idxs if pred[i] != true[i])
print("\nпары путаницы:", pairs.most_common(10))
