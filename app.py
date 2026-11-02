# -*- coding: utf-8 -*-
import streamlit as st
import medbot

st.set_page_config(page_title="МедБот", page_icon="🩺")
st.title("🩺 МедБот — первичная оценка жалоб")
st.caption("Модель не ставит диагноз. При тревожных симптомах обратитесь к врачу.")

st.sidebar.markdown("**build: grid-v3**")
st.sidebar.caption("76+ диагнозов · SBERT · LinearSVC")

text = st.text_area("Опишите жалобы своими словами:", height=100)

if st.button("🩺 Поставить предварительный диагноз", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Опишите жалобы — хотя бы пару слов.")
    else:
        flags = medbot.emergency_check(text)
        if flags:
            st.error("⚠️ Признаки, требующие срочной помощи:\n\n" +
                     "\n".join(f"• {f}" for f in flags) +
                     "\n\nЭто вне компетенции модели — звоните в скорую: **103** или **112**.")
            st.caption("Предсказания ниже — только справка, не основание откладывать вызов.")
        r = medbot.predict(text)
        st.markdown(f"Ветка модели: **{r['branch']}** · маржа уверенности: **{r['margin']:.2f}**")
        if r["confident"]:
            st.success(f"Предварительный диагноз: **{r['top'][0][0]}** ({r['top'][0][1]*100:.0f}%)")
            st.info(medbot.advice_for(r['top'][0][0]))
        else:
            st.warning("Уверенного ответа нет — жалобы похожи на несколько заболеваний.")
        st.subheader("Топ-3 варианта")
        for name, p in r["top"]:
            st.markdown(f"- **{name}** — {p*100:.0f}%")
        st.subheader("Похожие случаи из базы")
        for name, sim in medbot.get_state()["similar"](text, k=3):
            st.markdown(f"- {name} — сходство {sim*100:.0f}%")
