import streamlit as st
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# --- ПУТИ ---
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(PROJECT_PATH, 'knowledge.json')
LOG_PATH = os.path.join(PROJECT_PATH, 'unknown_queries.txt')
EMB_PATH = os.path.join(PROJECT_PATH, 'class_embeddings.npz')

# --- ЗАГРУЗКА ---
@st.cache_resource
def load_resources():
    global kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS
    
    # 1. База
    with open(KB_PATH, 'r', encoding='utf-8-sig') as f:
        kb = json.load(f)
        
    # 2. Вектора (самое важное - грузим готовый файл)
    if os.path.exists(EMB_PATH):
        data = np.load(EMB_PATH)
        CLASS_EMBEDDINGS = data['embeddings']
        CLASS_NAMES = data['names'].tolist()
    else:
        st.error("❌ Файл class_embeddings.npz не найден. Запустите vectorizer.py")
        st.stop()
        
    # 3. Энкодер для текста пользователя
    encoder = SentenceTransformer('intfloat/multilingual-e5-small')
    
    return kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS

kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS = load_resources()

# --- ЛОГИРОВАНИЕ (Wave 8) ---
def log_unknown_query(query: str):
    if not query.strip(): return False
    
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
                
                # Сравниваем с базой векторов (ТОЧНЫЙ ПОИСК ПО СМЫСЛУ)
                sims = cosine_similarity([user_emb], CLASS_EMBEDDINGS)[0]
                
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]
                found_class_id = CLASS_NAMES[best_idx]

                # ЛОГИРУЕМ (Wave 8)
                log_unknown_query(user_input)

                # --- ВЫВОД (Исправлено: выводим только если нашли по смыслу) ---
                if best_score > 0.65: # Порог уверенности
                    data = kb[found_class_id]
                    
                    # Если это не черновик
                    if not data.get('draft', True) and data.get('desc'):
                        diagnosis_text = data['desc']
                        is_urgent = data.get('urgent', False)
                        
                        st.markdown("### ⚕️ Результаты анализа")
                        st.markdown(f"**Диагноз:** {diagnosis_text}")
                        st.write(f"**Уверенность:** {best_score:.2%}")
                        
                        if data.get('confirm'):
                            st.markdown("**Что нужно сделать:**")
                            for act in data['confirm']:
                                st.write(f"• {act}")
                        
                        if is_urgent:
                            st.error("⛔ **ВНИМАНИЕ: ЭКСТРЕННЫЙ СЛУЧАЙ!**")
                            st.markdown("**Немедленно вызовите скорую помощь (103 или 112).**")
                    else:
                        st.info("Описание временно недоступно.")
                else:
                    st.info("ℹ️ Уверенности недостаточно для постановки диагноза.")
                    # Сообщаем, что записали в аналитику
                    if log_unknown_query(user_input):
                        st.success("✅ Запрос сохранен в аналитику (Wave 8).")
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
                st.text_area("Логи неизвестных запросов:", value="".join(logs[-100:]), height=300)
                if st.button("🗑 Очистить лог"):
                    open(LOG_PATH, 'w').close()
                    st.rerun()
            else:
                st.info("Лог пуст.")
        except Exception as e:
            st.error(f"Ошибка: {e}")
    else:
        st.info("Лог будет создан после первого запроса.")
