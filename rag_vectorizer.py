# -*- coding: utf-8 -*-
# Генератор эмбеддингов RAG-базы. Рецепт прежний: вектор темы = e5("passage: " + title + ". " + desc).
import json
import numpy as np
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("intfloat/multilingual-e5-small")
kb = json.load(open("rag_knowledge.json", encoding="utf-8-sig"))
names = list(kb.keys())
texts = ["passage: " + kb[n]["title"] + ". " + kb[n]["desc"] for n in names]
emb = encoder.encode(texts, normalize_embeddings=True).astype(np.float32)
np.savez("rag_embeddings.npz", embeddings=emb, names=np.array(names, dtype=object))
print("rag_embeddings.npz создан. Тем:", len(names))
