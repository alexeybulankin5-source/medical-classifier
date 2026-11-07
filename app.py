import streamlit as st
import json
import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. НАСТРОЙКИ ПУТЕЙ ---
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(PROJECT_PATH, 'knowledge.json')
PIPELINE_PATH = os.path.join(PROJECT_PATH, 'pipeline.joblib')
LOG_PATH = os.path.join(PROJECT_PATH, 'unknown_queries.txt')

# --- 2. ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ---
kb = {}
encoder = None
CLASS_NAMES = []
CLASS_EMBEDDINGS = None

# --- 3. ЗАГРУЗКА (С ЗАЩИТОЙ ОТ ЗАВИСАНИЯ) ---
@st.cache_resource(show_spinner="Загрузка нейросети (это может занять 1-2 минуты)...")
def load_resources():
    global kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS
    
    # Проверка наличия файлов
    if not os.path.exists(KB_PATH):
        st.error(f"Файл knowledge.json не найден в {KB_PATH}")
        st.stop()
    if not os.path.exists(PIPELINE_PATH):
        st.error(f"Файл pipeline.joblib не найден в {PIPELINE_PATH}")
        st.stop()

    try:
        # Загрузка базы
        with open(KB_PATH, 'r', encoding='utf-8-sig') as f:
            kb = json.load(f)
        
        CLASS_NAMES = list(kb.keys())
        
        # Загрузка энкодера (самое тяжелое место)
        # Таймаут 180 секунд (3 минуты) на скачивание модели
        encoder = SentenceTransformer('intfloat/multilingual-e5-small') 
        
        # Генерация векторов для 213 классов (делается ОДИН РАЗ при старте)
        if CLASS_EMBEDDINGS is None:
            st.info("Генерирую векторы для 213 диагнозов (один раз)...")
            CLASS_EMBEDDINGS = encoder.encode(CLASS_NAMES, show_progress_bar=True)
            
        return kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS
        
    except Exception as e:
        st.error(f"❌ Критическая ошибка загрузки. Попробуйте перезагрузить страницу. \nОшибка: {e}")
        st.stop()

# --- 4. ЛОГИКА ---
kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS = load_resources()

def log_unknown_query(query: str):
    if not query.strip(): return False
    
    user_emb = encoder.encode([query])[0]
    sims = cosine_similarity([user_emb], CLASS_EMBEDDINGS)[0]
    
    if np.max(sims) < 0.65:
        try:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(f"{query}\n")
            return True
        except Exception as e:
            print(f"Log error: {e}")
    return False

# --- 5. ИНТЕРФЕЙС ---
st.title("🩺 MedBot Duo")
st.markdown("Интеллектуальная система первичной диагностики")

tab1, tab2, tab3 = st.tabs(["📝 Жалобы", "🧪 Анализы", "📊 Аналитика"])

with tab1:
    st.header("📝 Анализ жалоб")
    user_input = st.text_area("Опишите жалобы:", height=150)
    
    if st.button("Определить диагноз"):
        if user_input.strip():
            with st.spinner("Идет магия..."):
                user_emb = encoder.encode([user_input])[0]
                sims = cosine_similarity([user_emb], CLASS_EMBEDDINGS)[0]
                
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]
                found_class_id = CLASS_NAMES[best_idx]

                # Логируем для Wave 8
                log_unknown_query(user_input)

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
                    st.info("ℹ️ Уверенности модели недостаточно для вывода.")
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
