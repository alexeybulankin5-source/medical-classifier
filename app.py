# -*- coding: utf-8 -*-
import streamlit as st
import medbot

st.set_page_config(page_title="МедБот", page_icon="🩺")
st.title("🩺 МедБот — первичная оценка жалоб")
st.caption("Модель не ставит диагноз. При тревожных симптомах обратитесь к врачу.")

key = medbot.cache_key() if hasattr(medbot, "cache_key") else "fallback"

_diag = {}
try:
    import joblib, numpy as np
    pd_ = joblib.load("pipeline.joblib")
    _diag["model"] = type(pd_["colloquial"]["model"]).__name__
    _diag["cardio"] = "да" if "стенокардия" in set(map(str, pd_["colloquial"]["classes"])) else "НЕТ"
    _diag["rows"] = np.load("emb_all.npz")["X"].shape[0]
except Exception as e:
    _diag["model"] = "ошибка: " + str(e)[:60]
_msrc = open("medbot.py", encoding="utf-8").read()
_diag["dedup"] = "есть" if "seen.add" in _msrc else "НЕТ"
_diag["predict"] = "union веток" if "объединяем" in _msrc else "старый — одна ветка"

st.sidebar.markdown("**build: grid-v8**")
st.sidebar.caption(f"модель colloquial: {_diag.get('model', '?')}")
st.sidebar.caption(f"стенокардия в классах: {_diag.get('cardio', '?')}")
st.sidebar.caption(f"эталон: {_diag.get('rows', '?')} строк · дедуп: {_diag.get('dedup', '?')}")
st.sidebar.caption(f"predict: {_diag.get('predict', '?')}")

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
        for name, sim in medbot.get_state(key)["similar"](text, k=3):
            st.markdown(f"- {name} — сходство {sim*100:.0f}%")
