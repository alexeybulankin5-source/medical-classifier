import streamlit as st
import json
import os
from datetime import datetime

# --- Пути (работает и в GitHub, и в Colab) ---
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(PROJECT_PATH, 'knowledge.json')
LOG_PATH = os.path.join(PROJECT_PATH, 'unknown_queries.txt')

# --- Загрузка базы ---
@st.cache_data
def load_kb():
    try:
        with open(KB_PATH, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Ошибка: {e}")
        return {}

kb = load_kb()

# --- ФУНКЦИЯ ЛОГИРОВАНИЯ (Wave 8) ---
def log_unknown_query(query: str):
    if not query.strip():
        return False
    
    # Проверяем, есть ли похожий класс в базе
    query_lower = query.lower()
    found = False
    for key in kb.keys():
        # Простая проверка: если слова из запроса есть в названии класса
        if any(word in key.lower() for word in query_lower.split()):
            found = True
            break

    # Если в базе такого нет - записываем
    if not found:
        try:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {query}\n")
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
                
                # ВАША ЛОГИКА МОДЕЛИ (заглушка)
                # Замените этот список на вывод вашей модели
                TOP_3_IDS = ["артроз_коленного_сустава", "остеохондроз_поясничный", "плечелопаточный_периартрит"] 
                
                found_class_id = None
                for pid in TOP_3_IDS:
                    if pid in kb:
                        found_class_id = pid
                        break

                # ЛОГИРУЕМ НЕИЗВЕСТНЫЕ ЗАПРОСЫ (Wave 8)
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
                    # Уведомление пользователю
                    if log_unknown_query(user_input):
                        st.success("✅ Запрос записан в аналитику (Wave 8).")

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
