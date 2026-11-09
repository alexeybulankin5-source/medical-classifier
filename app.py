import streamlit as st
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity  # Добавлен импорт

# --- ПУТИ ---
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(PROJECT_PATH, 'knowledge.json')
LOG_PATH = os.path.join(PROJECT_PATH, 'unknown_queries.txt')
EMB_PATH = os.path.join(PROJECT_PATH, 'class_embeddings.npz')

# --- ЗАГРУЗКА РЕСУРСОВ ---
@st.cache_resource
def load_resources():
    global kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS
    
    try:
        # 1. Загрузка базы знаний
        with open(KB_PATH, 'r', encoding='utf-8-sig') as f:
            kb = json.load(f)
            
        # 2. Загрузка векторов
        if os.path.exists(EMB_PATH):
            data = np.load(EMB_PATH)
            CLASS_EMBEDDINGS = data['embeddings']
            CLASS_NAMES = data['names'].tolist()
        else:
            st.error("❌ Файл class_embeddings.npz не найден. Запустите vectorizer.py")
            st.stop()
            
        # 3. Энкодер для текста пользователя
        encoder = SentenceTransformer('intfloat/multilingual-e5-small')
        
    except Exception as e:
        st.error(f"Ошибка при загрузке ресурсов: {str(e)}")
        raise
        
    return kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS

kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS = load_resources()

# --- ЛОГИРОВАНИЕ (Wave 8) ---
def log_unknown_query(query: str):
    if not query.strip(): 
        return False
    
    try:
        # Кодируем запрос пользователя
        user_emb = encoder.encode([query])[0]
        
        # Сравниваем с базой
        sims = cosine_similarity([user_emb], CLASS_EMBEDDINGS)[0]
        
        # Если лучшая уверенность ниже 65% - считаем запрос "неизвестным"
        if np.max(sims) < 0.65:
            try:
                os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
                with open(LOG_PATH, 'a', encoding='utf-8') as f:
                    f.write(f"{query}\n")
                return True
            except Exception as e:
                print(f"Ошибка при записи в лог: {e}")
                return False
    except Exception as e:
        st.error(f"Ошибка при обработке запроса: {str(e)}")
        return False
    return False

# --- ИНТЕРФЕЙС ---
st.title("🩺 MedBot Duo")
st.markdown("Интеллектуальная система первичной диагностики")

tab1, tab2, tab3 = st.tabs(["📝 Жалобы", "🧪 Анализы", "📊 Аналитика"])

with tab1:
    st.header("📝 Анализ жалоб")
    user_input = st.text_area("Опишите жалобы:", height=150)
    
    if st.button("Определить диагноз"):
        if user_input.strip():
            with st.spinner("Анализ..."):
                # Кодируем запрос пользователя
                user_emb = encoder.encode([user_input])[0]
                
                # Сравниваем с базой векторов (ТОЧНЫЙ ПОИСК ПО СМЫСЛУ)
                sims = cosine_similarity([user_emb], CLASS_EMBEDDINGS)[0]
                
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]
                found_class_id = CLASS_NAMES[best_idx]

                # ЛОГИРУЕМ (Wave 8)
                log_unknown_query(user_input)

                # --- ВЫВОД (Исправлено: выводим только если нашли по смыслу) ---
