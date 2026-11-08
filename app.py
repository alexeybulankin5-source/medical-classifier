# -*- coding: utf-8 -*-
import os
import streamlit as st

import medbot

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(PROJECT_PATH, "unknown_queries.txt")

st.set_page_config(page_title="MedBot", page_icon="🩺", layout="wide")

def log_unknown_query(query: str):
    """Wave 8: запрос записан как 'неизвестный', если модель дала tier=low."""
    if not query.strip():
        return False
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(query.strip() + "\n")
        return True
    except Exception as e:
        st.caption(f"Не удалось записать в лог: {e}")
    return False

st.title("🩺 MedBot")
st.markdown("Первичная классификация жалоб: диагноз, уверенность, шаги подтверждения.")

tab1, tab2, tab3 = st.tabs(["📝 Жалобы", "🧪 Анализы", "📊 Аналитика"])

with tab1:
    st.header("📝 Анализ жалоб")
    user_input = st.text_area("Опишите жалобы:", height=150)
    if st.button("Определить диагноз"):
        if user_input.strip():
            with st.spinner("Анализ..."):
                res = medbot.predict(user_input)
                tier = res.get("confidence")
                urgent_msgs = medbot.emergency_check(user_input)

                if tier == "low":
                    log_unknown_query(user_input)
                    st.info("ℹ️ Уверенности недостаточно. Опишите симптомы подробнее: где болит, как давно, что усиливает.")
                else:
                    st.markdown("### ⚕️ Результат")
                    st.markdown(f"**Диагноз:** {res['diagnosis']}")
                    st.caption(f"Уровень уверенности: **{tier}**")

                    if res.get("description"):
                        st.write(res["description"])
                    elif res.get("draft_advice"):
                        st.info("Описание временно недоступно.")

                    if res.get("confirm"):
                        st.markdown("**Что нужно сделать:**")
                        for act in res["confirm"]:
                            st.write(f"• {act}")

                    if res.get("urgent"):
                        st.error("⛔ **ВНИМАНИЕ: ЭКСТРЕННЫЙ СЛУЧАЙ! Немедленно вызовите скорую помощь (103 или 112).**")

                    for msg in urgent_msgs:
                        st.error(f"🚨 {msg}")

                    if res.get("guard"):
                        st.caption(f"⚙️ {res['guard']}")

                    alts = res.get("top", [])[1:]
                    if alts:
                        st.caption("Альтернативы: " + ", ".join(f"{c} ({v:.2f})" for c, v in alts))
        else:
            st.warning("Введите текст жалобы.")

with tab2:
    st.header("🧪 Анализы")
    st.info("Модуль в разработке.")

with tab3:
    st.header("📊 Аналитика (Wave 8)")
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                logs = f.readlines()
            if logs:
                st.write(f"📊 Собрано неизвестных запросов: {len(logs)}")
                st.text_area("Последние запросы:", value="".join(logs[-100:]), height=300)
                if st.button("🗑 Очистить лог"):
                    open(LOG_PATH, "w").close()
                    st.rerun()
            else:
                st.info("Лог пуст.")
        except Exception as e:
            st.error(f"Ошибка: {e}")
    else:
        st.info("Лог будет создан после первого запроса.")
