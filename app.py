
import streamlit as st
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Проверяем наличие файлов
if not os.path.exists('knowledge.json'):
    st.error("Файл knowledge.json не найден!")
    raise FileNotFoundError

if not os.path.exists('class_embeddings.npz'):
    st.error("Файл class_embeddings.npz не найден!")
    raise FileNotFoundError

# Основная логика приложения
@st.cache_resource
def load_resources():
    global kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS
    
    # Загружаем базу знаний
    with open('knowledge.json', 'r', encoding='utf-8-sig') as f:
        kb = json.load(f)
        
    # Загружаем векторные представления
    data = np.load('class_embeddings.npz')
    CLASS_EMBEDDINGS = data['embeddings']
    CLASS_NAMES = data['names'].tolist()
    
    # Инициализируем энкодер
    encoder = SentenceTransformer('intfloat/multilingual-e5-small')
    
    return kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS

kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS = load_resources()

# Интерфейс приложения
st.title("🩺 MedBot Duo")
st.markdown("Интеллектуальная система первичной диагностики")

# Остальной код интерфейса...
