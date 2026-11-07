# vectorizer.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer

print("Загрузка knowledge.json...")
with open('knowledge.json', 'r', encoding='utf-8-sig') as f:
    kb = json.load(f)

class_names = list(kb.keys())
print(f"Найдено классов: {len(class_names)}")

# Загружаем модель
print("Загрузка модели e5...")
encoder = SentenceTransformer('intfloat/multilingual-e5-small')

# Генерируем векторы для всех 213 диагнозов
print("Генерация векторов для базы (это долго, 1 раз)...")
embeddings = encoder.encode(class_names, show_progress_bar=True, normalize_embeddings=True)

# Сохраняем
np.savez_compressed('class_embeddings.npz', embeddings=embeddings, names=np.array(class_names))
print("✅ Векторы сохранены в class_embeddings.npz")
