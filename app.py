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

# --- ЗАГРУЗКА (Надежная) ---
@st.cache_resource
def load_resources():
    global kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS
    
    # 1. База
    try:
        with open(KB_PATH, 'r', encoding='utf-8-sig') as f:
            kb = json.load(f)
    except Exception as e:
        st.error(f"❌ Ошибка загрузки knowledge.json: {e}")
        st.stop()

    # 2. Энкодер (модель для понимания смысла)
    try:
        encoder = SentenceTransformer('intfloat/multilingual-e5-small')
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели. Проверьте requirements.txt. Ошибка: {e}")
        st.stop()

    # 3. Генерация векторов для ВСЕХ 213 диагнозов (делается один раз при старте)
    CLASS_NAMES = list(kb.keys())
    CLASS_EMBEDDINGS = encoder.encode(CLASS_NAMES, show_progress_bar=False, normalize_embeddings=True)
    
    return kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS

kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS = load_resources()

# --- ФУНКЦИЯ ЛОГИРОВАНИЯ (Wave 8) ---
def log_unknown_query(query: str):
    if not query.strip(): return
    
    user_emb = encoder.encode([query])[0]
    sims = cosine_similarity([user_emb], CLASS_EMBEDDINGS)[0]
    
    # Если модель вообще не уверена (ни один класс не похож)
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
                
                # Кодируем запрос пользователя
                user_emb = encoder.encode([user_input])[0]
                
                # Сравниваем с базой
                sims = cosine_similarity([user_emb], CLASS_EMBEDDINGS)[0]
                
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]
                found_class_id = CLASS_NAMES[best_idx]

                # ЛОГИРУЕМ (Wave 8)
                log_unknown_query(user_input)

                # ОТОБРАЖЕНИЕ
                if best_score > 0.65: # Порог уверенности
                    data = kb[found_class_id]
                    
                    # Проверка на черновик (Wave 7b fix)
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
                            st.markdown("**Немедленно вызовите скорую помощь (103 или 112).**")
                    else:
                        st.info("Описание временно недоступно.")
                else:
                    st.info("ℹ️ Уверенности модели недостаточно для вывода диагноза.")
        else:
            st.warning("Введите текст жалобы.")

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
