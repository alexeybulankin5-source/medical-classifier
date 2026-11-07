import streamlit as st
import json
import os

# --- ПУТИ ---
# (Работает и в Colab, и в GitHub репозитории)
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(PROJECT_PATH, 'knowledge.json')
LOG_PATH = os.path.join(PROJECT_PATH, 'unknown_queries.txt')

# --- ЗАГРУЗКА БАЗЫ (Мгновенно) ---
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
    
    query_lower = query.lower().replace('ё', 'е')
    found = False
    
    # Простой поиск по словам
    for key in kb.keys():
        key_words = key.lower().split('_')
        query_words = query_lower.split()
        
        # Если хотя бы одно слово из запроса есть в названии класса
        if any(word in key_words for word in query_words):
            found = True
            break
            
    # Если вообще ничего не нашли в базе - пишем в лог
    if not found:
        try:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, 'a', encoding='utf-8') as log_file:
                log_file.write(f"{query}\n")
            return True
        except Exception as e:
            print(f"Ошибка записи лога: {e}")
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
                
                # --- ПОИСК (Простой, по словам) ---
                query_lower = user_input.lower().replace('ё', 'е')
                found_class_id = None
                
                for key in kb.keys():
                    key_words = key.lower().split('_')
                    query_words = query_lower.split()
                    
                    # Если есть совпадение слов
                    if any(word in query_words for word in key_words):
                        found_class_id = key
                        break
                
                # ЛОГИРУЕМ (Wave 8)
                log_unknown_query(user_input)

                # --- ВЫВОД ---
                if found_class_id:
                    data = kb[found_class_id]
                    
                    # Если это не черновик и есть описание
                    if not data.get('draft', True) and data.get('desc'):
                        st.markdown("### ⚕️ Результаты анализа")
                        st.markdown(f"**Диагноз:** {data['desc']}")
                        
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
                    st.info("ℹ️ Уверенного ответа нет.")
        else:
            st.warning("Введите текст жалобы.")

# --- ВКЛАДКА АНАЛИЗЫ ---
with tab2:
    st.header("🧪 Анализы")
    st.info("Модуль в разработке.")

# --- ВКЛАДКА АНАЛИТИКА ---
with tab3:
    st.header("📊 Аналитика (Wave 8)")
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, 'r', encoding='utf-8') as f:
                logs = f.readlines()
            if logs:
                st.write(f"📊 Собрано неизвестных запросов: {len(logs)}")
                st.text_area("Логи для расширения базы:", value="".join(logs[-100:]), height=300)
                
                if st.button("🗑 Очистить лог"):
                    open(LOG_PATH, 'w').close()
                    st.rerun()
            else:
                st.info("Лог пуст.")
        except Exception as e:
            st.error(f"Ошибка чтения лога: {e}")
    else:
        st.info("Лог будет создан после первого неизвестного запроса.")
