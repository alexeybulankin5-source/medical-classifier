import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os

def load_knowledge_base():
    # Загружаем базу знаний
    with open('knowledge.json', 'r', encoding='utf-8-sig') as f:
        kb = json.load(f)
    return kb

def create_embeddings(kb):
    # Инициализируем модель
    encoder = SentenceTransformer('intfloat/multilingual-e5-small')
    
    class_names = []
    descriptions = []
    
    # Собираем описания и названия классов
    for class_id, data in kb.items():
        if not data.get('draft', True) and data.get('desc'):
            class_names.append(class_id)
            descriptions.append(data['desc'])
    
    # Генерируем эмбеддинги
    embeddings = encoder.encode(descriptions)
    return embeddings, class_names

def save_embeddings(embeddings, class_names):
    # Сохраняем в npz формат
    np.savez(
        'class_embeddings.npz',
        embeddings=embeddings,
        names=class_names
    )
    print("Файл class_embeddings.npz успешно создан!")

if __name__ == "__main__":
    # Проверяем наличие knowledge.json
    if not os.path.exists('knowledge.json'):
        print("Ошибка: файл knowledge.json не найден!")
        exit(1)
    
    # Загружаем и обрабатываем данные
    kb = load_knowledge_base()
    embeddings, class_names = create_embeddings(kb)
    
    # Сохраняем результат
    save_embeddings(embeddings, class_names)
