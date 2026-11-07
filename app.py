import streamlit as st
import json
import os
from datetime import datetime

# --- 1. ПУТИ ---
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(PROJECT_PATH, 'knowledge.json')
LOG_PATH = os.path.join(PROJECT_PATH, 'unknown_queries.txt')

# --- 2. ЗАГРУЗКА БАЗЫ ---
@st.cache_data
def load_kb():
    try:
        with open(KB_PATH, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Ошибка: {e}")
        return {}

kb = load_kb()

# --- 3. ФУНКЦИЯ ЛОГИРОВАНИЯ (Wave 8) ---
def log_unknown_query(query: str):
    if not query.strip():
        return False
    
    # Проверяем по словам (на случай если эмбеддинги не загрузились)
    query_lower = query.lower()
    found = False
    for key in kb.keys():
        if query_lower.replace(' ', '_') in key.lower() or key.lower() in query_lower:
            found = True
            break
            
    if not found:
        try:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {query}\n")
            return True
        except Exception as e:
            print(f"Log error: {e}")
    return False

# --- 4. ИНТЕРФЕЙС ---
st.title("🩺 MedBot Duo")
st.markdown("Интеллектуальная система первичной диагностики")

tab1, tab2, tab3 = st.tabs(["📝 Жалобы", "🧪 Анализы", "📊 Аналитика"])

with tab1:
    st.header("📝 Анализ жалоб")
    user_input = st.text_area("Опишите жалобы:", height=150)

    if st.button("Определить диагноз"):
        if user_input.strip():
            with st.spinner("Анализ..."):
                
                # --- УМНЫЙ ПОИСК (Wave 7b) ---
                found_class_id = None
                max_sim = 0.0
                
                # ВАЖНО: Загрузите ваш энкодер (e5_onnx) здесь
                # from sentence_transformers import SentenceTransformer
                # encoder = SentenceTransformer('intfloat/multilingual-e5-small')
                
                # user_emb = encoder.encode([user_input])[0]
                # sims = cosine_similarity([user_emb], CLASS_EMBEDDINGS)[0]
                # best_idx = np.argmax(sims)
                
                # if sims[best_idx] > 0.65: # Порог
                #     found_class_id = CLASS_NAMES[best_idx]
                
                # --- ЗАГЛУШКА (пока нет векторов) ---
                # Ищем просто по словам (надежно, но менее точно)
                query_words = set(user_input.lower().split())
                for key in kb.keys():
                    key_words = set(key.lower().replace('_', ' ').split())
                    common = len(query_words & key_words)
                    total = len(query_words | key_words)
                    sim = common / total if total > 0 else 0
                    if sim > max_sim and sim > 0.4: # Порог 40%
                        max_sim = sim
                        found_class_id = key
                # --- КОНЕЦ ЗАГЛУШКИ ---

                # ЛОГИРУЕМ (Wave 8)
                log_unknown_query(user_input)

                # ОТОБРАЖЕНИЕ
                if found_class_id:
                    data = kb[found_class_id]
                    if not data.get('draft', True) and data.get('desc'):
                        st.markdown("### ⚕️ Результаты")
                        st.markdown(f"**Диагноз:** {data['desc']}")
                        if data.get('confirm'):
                            st.markdown("**Что нужно сделать:**")
                            for act in data['confirm']:
                                st.write(f"• {act}")
                        if data.get('urgent'):
                            st.warning("⚠️ **Внимание:** Требуется врач!")
                    else:
                        st.info("Описание временно недоступно.")
                else:
                    st.info("ℹ️ Уверенного ответа нет.")
                    if log_unknown_query(user_input):
                        st.success("✅ Запрос записан в аналитику (Wave 8).")
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
