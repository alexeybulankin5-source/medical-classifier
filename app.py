import streamlit as st
import json
import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer  # <-- ЭТОЙ СТРОКИ НЕ ХВАТАЛО
from sklearn.metrics.pairwise import cosine_similarity

# --- ПУТИ ---
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(PROJECT_PATH, 'knowledge.json')
EMB_PATH = os.path.join(PROJECT_PATH, 'emb_all.npz')
PIPELINE_PATH = os.path.join(PROJECT_PATH, 'pipeline.joblib')
LOG_PATH = os.path.join(PROJECT_PATH, 'unknown_queries.txt')

# --- ЗАГРУЗКА МОДЕЛИ И БАЗЫ ---
@st.cache_resource
def load_resources():
    try:
        kb = json.load(open(KB_PATH, 'r', encoding='utf-8-sig'))
        emb_matrix = np.load(EMB_PATH)['arr_0']
        pipeline = joblib.load(PIPELINE_PATH)
        encoder = SentenceTransformer('intfloat/multilingual-e5-small')
        return kb, emb_matrix, pipeline, encoder
    except Exception as e:
        st.error(f"Ошибка загрузки ресурсов: {e}")
        return None, None, None, None

kb, emb_matrix, pipeline, encoder = load_resources()

if not kb:
    st.stop()

# --- ФУНКЦИЯ ЛОГИРОВАНИЯ (Wave 8) ---
def log_unknown_query(query: str):
    if not query.strip():
        return False
    
    user_emb = encoder.encode([query])[0]
    sims = cosine_similarity([user_emb], emb_matrix)[0]
    
    if np.max(sims) < 0.65:
        try:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(f"{query}\n")
            return True
        except Exception as e:
            print(f"Log error: {e}")
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
                
                # УМНЫЙ ПОИСК (Wave 7b)
                user_emb = encoder.encode([user_input])[0]
                sims = cosine_similarity([user_emb], emb_matrix)[0]
                
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]
                found_class_id = list(kb.keys())[best_idx]

                # ЛОГИРОВАНИЕ (Wave 8)
                log_unknown_query(user_input)

                # ОТОБРАЖЕНИЕ
                if best_score > 0.65:
                    data = kb[found_class_id]
                    if not data.get('draft', True) and data.get('desc'):
                        st.markdown("### ⚕️ Результаты анализа")
                        st.markdown(f"**Диагноз:** {data['desc']}")
                        st.write(f"**Уверенность:** {best_score:.2%}")
                        
                        if data.get('confirm'):
                            st.markdown("**Что нужно сделать:**")
                            for act in data['confirm']:
                                st.write(f"• {act}")
                        
                        if data.get('urgent'):
                            st.error("⛔ **ТРЕБУЕТСЯ СРОЧНАЯ ПОМОЩЬ!**")
                    else:
                        st.info("Описание временно недоступно.")
                else:
                    st.info("ℹ️ Уверенного ответа нет (порог уверенности не пройден).")
        else:
            st.warning("Введите текст.")

with tab2:
    st.header("🧪 Анализы")
    st.info("Модуль в разработке.")

with tab3:
    st.header("📊 Аналитика (Wave 8)")
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, 'r', encoding='utf-8') as f:
                logs = f.readlines()
            if logs:
                st.write(f"📊 Собрано неизвестных запросов: {len(logs)}")
                st.text_area("Логи:", value="".join(logs[-100:]), height=300)
                if st.button("🗑 Очистить лог"):
                    open(LOG_PATH, 'w').close()
                    st.rerun()
            else:
                st.info("Лог пуст.")
        except Exception as e:
            st.error(f"Ошибка: {e}")
    else:
        st.info("Лог будет создан после первого запроса.")
