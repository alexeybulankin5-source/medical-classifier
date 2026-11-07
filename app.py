import streamlit as st
import json
import os
import re

# --- ПУТИ ---
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(PROJECT_PATH, 'knowledge.json')
LOG_PATH = os.path.join(PROJECT_PATH, 'unknown_queries.txt')

# --- ЗАГРУЗКА ---
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
def log_unknown(query: str):
    if not query.strip(): return False
    
    # Проверяем, есть ли в базе похожие слова
    query_words = query.lower().split()
    found = False
    for key in kb.keys():
        key_words = key.lower().replace('_', ' ').split()
        # Если есть хотя бы одно общее слово
        if any(word in key_words for word in query_words):
            found = True
            break
            
    if not found:
        try:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, 'a', encoding='utf-8') as log_file:
                log_file.write(f"{query}\n")
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
                
                # --- ПРОСТОЙ ПОИСК (ТОП-3) ---
                query_lower = user_input.lower()
                results = []
                
                for class_id, data in kb.items():
                    # Собираем текст для поиска: ID + Описание
                    search_text = (class_id + " " + data.get('desc', '')).lower()
                    
                    # Считаем количество общих слов
                    score = sum(1 for word in query_lower.split() if word in search_text)
                    
                    if score > 0:
                        results.append({
                            "id": class_id,
                            "score": score,
                            "data": data
                        })
                
                # Сортируем по количеству совпадений
                results.sort(key=lambda x: x['score'], reverse=True)
                TOP_3 = results[:3]

                # ЛОГИРУЕМ (Wave 8)
                log_unknown_query(user_input)

                # --- ВЫВОД ТОП-3 ---
                if TOP_3:
                    st.markdown("### ⚕️ Возможные диагнозы")
                    
                    for item in TOP_3:
                        data = item['data']
                        class_id = item['id']
                        
                        # Готовим текст
                        diagnosis_text = data.get('desc', class_id.replace('_', ' ').title())
                        is_urgent = data.get('urgent', False)
                        
                        # Выводим блок
                        st.markdown(f"**{diagnosis_text}** (Совпадений: {item['score']})")
                        
                        if data.get('confirm'):
                            st.write("**Что нужно сделать:**")
                            for act in data['confirm']:
                                st.write(f"• {act}")
                        
                        if is_urgent:
                            st.error("⛔ **ТРЕБУЕТСЯ СРОЧНАЯ ПОМОЩЬ!**")
                        
                        st.write("---")
                else:
                    st.info("ℹ️ Уверенного ответа нет.")
                    if log_unknown_query(user_input):
                        st.success("✅ Запрос сохранен в аналитику.")

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
                st.write(f"📊 Неизвестных запросов: {len(logs)}")
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
