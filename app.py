# -*- coding: utf-8 -*-
import importlib.util, os, sys
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("medbot_live", os.path.join(_HERE, "medbot.py"))
medbot = importlib.util.module_from_spec(_spec)
sys.modules["medbot_live"] = medbot
_spec.loader.exec_module(medbot)

st.set_page_config(page_title="МедБот", page_icon="🩺")
st.title("🩺 МедБот — первичная оценка жалоб")
st.caption("Модель не ставит диагноз. При тревожных симптомах обратитесь к врачу.")

key = medbot.cache_key() if hasattr(medbot, "cache_key") else "fallback"

st.sidebar.markdown(f"**build: {getattr(medbot, 'VERSION', 'СТАРЫЙ — версии нет!')}**")
st.sidebar.caption("гибрид в medbot: " + ("да" if hasattr(medbot, "W_MODEL") else "НЕТ — старый код!"))
st.sidebar.caption(f"модуль загружен из: {getattr(medbot, '__file__', '?')}")
_diag = {}
try:
    import joblib, numpy as np
    pd_ = joblib.load("pipeline.joblib")
    _diag["model"] = type(pd_["colloquial"]["model"]).__name__
    _diag["cardio"] = "да" if "стенокардия" in set(map(str, pd_["colloquial"]["classes"])) else "НЕТ"
    _diag["rows"] = np.load("emb_all.npz")["X"].shape[0]
except Exception as e:
    _diag["model"] = "ошибка: " + str(e)[:60]
st.sidebar.caption(f"модель colloquial: {_diag.get('model', '?')}")
st.sidebar.caption(f"стенокардия в классах: {_diag.get('cardio', '?')}")
st.sidebar.caption(f"эталон: {_diag.get('rows', '?')} строк")

text = st.text_area("Опишите жалобы своими словами:", height=100)

if st.button("🩺 Поставить предварительный диагноз", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Опишите жалобы — хотя бы пару слов.")
        st.stop()
    flags = medbot.emergency_check(text)
    if flags:
        st.error("⚠️ Признаки, требующие срочной помощи:\n\n" +
                 "\n".join(f"• {f}" for f in flags) +
                 "\n\nЭто вне компетенции модели — звоните в скорую: **103** или **112**.")
        st.warning("При экстренных признаках автоматический диагноз не выдаётся: "
                   "оценка требует врача. До приезда скорой не оставляйте человека одного.")
        st.stop()
    r = medbot.predict(text)
    st.markdown(f"Ветка модели: **{r['branch']}** · маржа уверенности: **{r['margin']:.2f}**")
    conf = r.get("confidence") or ("high" if r["confident"] else "low")
    if conf == "high":
        st.success(f"Предварительный диагноз: **{r['top'][0][0]}** ({r['top'][0][1]*100:.0f}%)")
        from advice_ui import render_advice  # advice UI (wave5)
        render_advice(r)
    elif conf == "medium":
        alts = ", ".join(c for c, _ in r["top"][1:3])
        st.info(f"Предположительно **{r['top'][0][0]}** ({r['top'][0][1]*100:.0f}%). Похожие варианты: {alts} — если что-то из этого ближе к вашим жалобам, уточните формулировку.")
    else:
        st.warning("Уверенного ответа нет — опишите главный симптом подробнее (где болит, когда началось, что усиливает), и поиск сузится.")
    st.subheader("Топ-3 варианта")
    for name, p in r["top"]:
        st.markdown(f"- **{name}** — {p*100:.0f}%")
    st.subheader("Похожие случаи из базы")
    for name, sim in medbot.get_state(key)["similar"](text, k=3):
        st.markdown(f"- {name} — сходство {sim*100:.0f}%")
