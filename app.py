# -*- coding: utf-8 -*-
import io
import traceback
import streamlit as st
import medbot

st.set_page_config(page_title="МедБот", page_icon="🩺", layout="centered")
st.title("🩺 МедБот — первичная оценка жалоб")
st.caption("Модель не ставит диагноз. При тревожных симптомах обратитесь к врачу.")

text = st.text_area("Опишите жалобы своими словами:", height=100)
audio = st.audio_input("…или запишите голосом 🎙")

if audio is not None and not text.strip():
    import speech_recognition as sr
    try:
        rec = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio.getvalue())) as src:
            rec.adjust_for_ambient_noise(src, duration=0.5)
            data = rec.record(src)
        text = rec.recognize_google(data, language="ru-RU")
        st.success(f"Распознано: «{text}»")
    except Exception:
        st.warning("Не удалось распознать речь. Введите текст вручную.")

if text.strip():
    with st.spinner("Анализирую..."):
        try:
            r = medbot.predict(text.strip())
        except Exception:
            st.error("**Отладка — полный трейсбек:**\n\n```\n"
                     + traceback.format_exc() + "\n```")
            st.stop()

    st.write(f"Ветка модели: **{r['branch']}** · маржа уверенности: **{r['margin']:.2f}**")

    if r["confident"]:
        top_name = r["top"][0][0]
        st.success(f"Наиболее вероятный диагноз: **{top_name}** ({r['top'][0][1]*100:.0f}%)")
        st.subheader("Что делать дальше")
        st.write(medbot.advice_for(top_name))
        st.markdown("**Красные флаги** — при любом из них сразу к врачу:")
        st.write(" • " + "\n • ".join(medbot.RED_FLAGS))
    else:
        st.warning("Уверенного ответа нет — жалобы похожи на несколько заболеваний.")
        st.subheader("Топ-3 варианта")
        for name, p in r["top"]:
            st.write(f"**{name}** — {p*100:.0f}%")
        st.info("Уточните: что из этого больше похоже на ваше состояние?")
        choice = st.radio("Выберите вероятный диагноз:", [n for n, _ in r["top"]], index=None)
        if choice:
            st.markdown(f"**Если это {choice}, то что делать:**")
            st.write(medbot.advice_for(choice))
            st.markdown("**Красные флаги:**")
            st.write(" • " + "\n • ".join(medbot.RED_FLAGS))
