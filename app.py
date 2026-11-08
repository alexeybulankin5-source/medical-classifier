import streamlit as st
import json
import os
import numpy as np
from datetime import datetime

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
        st.error(f"Ошибка загрузки базы: {e}")
        return {}

kb = load_kb()

# --- ФУНКЦИЯ ЛОГИРОВАНИЯ (Wave 8) ---
def log_unknown_query(query: str):
    if not query.strip(): return False
    query_lower = query.lower().replace('ё', 'е')
    found = any(query_lower.replace(' ', '_') in key.lower() for key in kb.keys())
    if not found:
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
                query_lower = user_input.lower().replace('ё', 'е')
                results = []
                
                for class_id, data in kb.items():
                    search_text = (class_id + " " + data.get('desc', '')).lower()
                    score = sum(1 for word in query_lower.split() if word in search_text)
                    if score > 0:
                        results.append({"id": class_id, "score": score, "data": data})
                
                results.sort(key=lambda x: x['score'], reverse=True)
                TOP_3 = results[:3]

                log_unknown_query(user_input)

                if TOP_3:
                    st.markdown("### ⚕️ Возможные диагнозы")
                    
                    for item in TOP_3:
                        data = item['data']
                        class_id = item['id']
                        
                        # --- ГЛАВНОЕ ИСПРАВЛЕНИЕ ---
                        # Если desc пустой (черновик), мы НЕ выводим его как диагноз
                        # Мы выводим заглушку, чтобы вы видели, что класс найден
                        diagnosis_text = data.get('desc')
                        if not diagnosis_text:
                            diagnosis_text = f"[{class_id.replace('_', ' ').title()}] — Требует описания (Wave 7b)"
                        
                        is_urgent = data.get('urgent', False)
                        
                        st.markdown(f"**{diagnosis_text}** (Совпадений: {item['score']})")
                        
                        if data.get('confirm'):
                            st.markdown("**Что нужно сделать:**")
                            for act in data['confirm']:
                                st.write(f"• {act}")
                        
                        if is_urgent:
                            st.error("⛔ **ВНИМАНИЕ: ЭКСТРЕННЫЙ СЛУЧАЙ!**")
                            st.markdown("**Немедленно вызовите скорую помощь (103 или 112).**")
                        
                        st.write("---")
                else:
                    st.info("ℹ️ Уверенности недостаточно.")
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
