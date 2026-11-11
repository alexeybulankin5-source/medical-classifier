# -*- coding: utf-8 -*-
# Генератор lab_embeddings.npz: вектор паттерна = e5 encode("passage: " + desc)
# Рецепт восстановлен по сверке с оригинальным файлом (косинус 1.000 на 70/70).
import json
import numpy as np
from sentence_transformers import SentenceTransformer

lab = json.load(open("lab_knowledge.json", encoding="utf-8-sig"))
patterns = lab["patterns"]
names = list(patterns.keys())
texts = ["passage: " + patterns[n].get("desc", "") for n in names]

encoder = SentenceTransformer("intfloat/multilingual-e5-small")
embeddings = encoder.encode(texts, normalize_embeddings=True).astype(np.float32)

np.savez("lab_embeddings.npz", embeddings=embeddings, names=np.array(names, dtype=object))
print("Файл lab_embeddings.npz успешно создан! Паттернов:", len(names))
