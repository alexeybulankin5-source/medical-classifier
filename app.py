import streamlit as st
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- ПУТИ ---
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(PROJECT_PATH, 'knowledge.json')
LOG_PATH = os.path.join(PROJECT_PATH, 'unknown_queries.txt')

# --- ЗАГРУЗКА ---
@st.cache_resource
def load_resources():
    global kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS
    try:
        with open(KB_PATH, 'r', encoding='utf-8-sig') as f:
            kb = json.load(f)
        
        encoder = SentenceTransformer('intfloat/multilingual-e5-small')
        CLASS_NAMES = list(kb.keys())
        CLASS_EMBEDDINGS = encoder.encode(CLASS_NAMES, show_progress_bar=False, normalize_embeddings=True)
        
        return kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        st.stop()

kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS = load_resources()

# --- ЛОГИРОВАНИЕ (Wave 8) ---
def log_unknown_query(query: str):
    if not query.strip(): return
    user_emb = encoder.encode([query])[0]
    sims = cosine_similarity([user_emb], CLASS_EMBEDDINGS)[0]
    if np.max(sims) < 0.65:
        try:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(f"{query}\n")
        except Exception as e:
            print(f"Log error: {e}")

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
                user_emb = encoder.encode([user_input])[0]
                sims = cosine_similarity([user_emb], CLASS_EMBEDDINGS)[0]
                
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]
                found_class_id = CLASS_NAMES[best_idx]

                # Логируем (Wave 8)
                log_unknown_query(user_input)

                # --- ИСПРАВЛЕННЫЙ БЛОК ВЫВОДА ---
                if best_score > 0.65:
                    data = kb[found_class_id]
                    
                    # 1. Выводим диагноз (даже если это черновик, чтобы видеть, что найдено)
                    diagnosis_text = data.get('desc', f"Диагноз: {found_class_id.replace('_', ' ').title()}")
                    
                    st.markdown("### ⚕️ Результаты анализа")
                    st.markdown(f"**Диагноз:** {diagnosis_text}")
                    st.write(f"**Уверенность:** {best_score:.2%}")

                    # 2. КРАСНЫЙ ФЛАГ (Urgent)
                    if data.get('urgent', False):
                        st.error("⛔ **ОСТОРОЖНО: ЭКСТРЕННЫЙ СЛУЧАЙ!**")
                        st.markdown("**Немедленно вызовите скорую помощь (103 или 112).**")

                    # 3. СОВЕТЫ (исправлено на data.get('confirm'))
                    if data.get('confirm'):
                        st.markdown("**Что нужно сделать:**")
                        for act in data['confirm']:
                            st.write(f"• {act}")
                
                else:
                    st.info("ℹ️ Уверенности недостаточно для диагноза.")
                    if log_unknown_query(user_input):
                        st.success("✅ Запрос сохранен в аналитику (Wave 8).")

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
