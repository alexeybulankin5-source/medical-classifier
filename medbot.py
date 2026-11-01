# -*- coding: utf-8 -*-
import csv
import glob
import numpy as np
import joblib
import streamlit as st

SBERT_MODEL = "ai-forever/sbert_large_nlu_ru"
MARGIN_THRESHOLD = 0.12

@st.cache_resource
def get_state():
    from sentence_transformers import SentenceTransformer
    pipe = joblib.load("pipeline.joblib")
    sbert = SentenceTransformer(SBERT_MODEL)
    X = np.load(glob.glob("emb_*.npz")[0])["X"].astype(np.float32)

    with open("dataset.csv", encoding="utf-8") as f:
        text0 = next(csv.reader(f))[0].lstrip("\ufeff")

    def cos(a, b):
        a = np.atleast_2d(a); b = np.atleast_2d(b)
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return float(a @ b.T)

    e_raw = sbert.encode([text0], convert_to_numpy=True).astype(np.float32)
    e_norm = e_raw / (np.linalg.norm(e_raw) + 1e-9)
    use_norm = cos(e_norm, X[0]) >= cos(e_raw, X[0])

    def embed(texts):
        e = sbert.encode(texts, convert_to_numpy=True).astype(np.float32)
        if use_norm:
            e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        return e

    return {"pipe": pipe, "embed": embed, "normalized": use_norm}

def _branch_name(raw, pipe):
    """Роутер может возвращать и строку, и индекс — сводим оба случая к имени ветки."""
    if isinstance(raw, (str, np.str_)):
        return str(raw)
    classes = list(pipe["router"].classes_)
    if all(isinstance(c, (str, np.str_)) for c in classes):
        return str(classes[int(raw)])
    # числовые метки: порядок алфавитный — 0 -> colloquial, 1 -> mixed
    return ["colloquial", "mixed"][int(raw)]

def predict(text):
    s = get_state()
    emb = s["embed"]([text])
    branch = _branch_name(s["pipe"]["router"].predict(emb)[0], s["pipe"])
    sub = s["pipe"][branch]
    proba = sub["model"].predict_proba(emb)[0]
    order = np.argsort(proba)[::-1][:3]
    top = [(sub["classes"][i], float(proba[i])) for i in order]
    margin = top[0][1] - top[1][1]
    return {"branch": branch, "top": top, "margin": margin,
            "confident": margin >= MARGIN_THRESHOLD}

ADVICE = {
    "грипп": "Постельный режим 3–5 дней, обильное тёплое питьё, жаропонижающее при температуре выше 38.5.",
    "ОРВИ": "Покой, питьё, симптоматическое лечение. Обычно проходит за 5–7 дней.",
    "ангина": "Осмотр врача обязателен: при гнойной ангине нужны антибиотики. Полоскания тёплым солевым раствором — до визита.",
    "бронхит": "Тёплое питьё, покой. Если кашель с гнойной мокротой или держится дольше 2 недель — к терапевту.",
    "гастрит": "Дробное питание, исключить острое, кофе и алкоголь. Если боль повторяется неделю — к гастроэнтерологу.",
    "эзофагит": "Не ложиться 2 часа после еды, ужин за 3 часа до сна, избегать жирного и кислого.",
    "цистит": "Обильное питьё, тепло на низ живота. Если появилась кровь в моче или температура — срочно к урологу.",
    "мигрень": "Тишина и темнота в приступе, зафиксировать триггеры в дневнике. Частые приступы — к неврологу.",
    "астма": "Ингалятор должен быть всегда под рукой. Одышка в покое или ночью — к пульмонологу.",
    "чесотка": "Лечение только по рецепту врача. Одновременно обрабатывают всех членов семьи, бельё — горячая стирка.",
    "пневмония": "Срочно к врачу: нужна рентгенография и назначение антибиотиков. Самолечение недопустимо.",
    "пищевое отравление": "Обильное дробное питьё, щадящая диета. Температура, кровь в стуле или признаки обезвоживания — к врачу.",
}

RED_FLAGS = [
    "температура выше 39 и не сбивается",
    "сильная одышка или боль в груди",
    "спутанность сознания, обморок",
    "кровь в стуле, моче или рвоте",
    "резкая слабость, сыпь с температурой",
]

def advice_for(diagnosis):
    if diagnosis in ADVICE:
        return ADVICE[diagnosis]
    return ("Стандартная тактика: наблюдать за динамикой 2–3 дня, пить достаточно жидкости, "
            "не переносить болезнь на ногах. Если симптомы нарастают — к терапевту.")
