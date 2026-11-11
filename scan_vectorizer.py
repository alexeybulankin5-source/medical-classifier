# -*- coding: utf-8 -*-
# Генератор эмбеддингов для расшифровки снимков и ЭКГ.
# Рецепт совпадает с lab_vectorizer.py: вектор термина = e5("passage: " + desc).
import json
import numpy as np
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("intfloat/multilingual-e5-small")

for kb_file, emb_file in (("xray_knowledge.json", "xray_embeddings.npz"),
                          ("ecg_knowledge.json", "ecg_embeddings.npz")):
    kb = json.load(open(kb_file, encoding="utf-8-sig"))
    names = list(kb.keys())
    texts = ["passage: " + kb[n].get("desc", "") for n in names]
    emb = encoder.encode(texts, normalize_embeddings=True).astype(np.float32)
    np.savez(emb_file, embeddings=emb, names=np.array(names, dtype=object))
    print(emb_file, "создан. Термины:", len(names))
