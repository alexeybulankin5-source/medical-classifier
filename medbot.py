# -*- coding: utf-8 -*-
import os, re, numpy as np, joblib, streamlit as st

SBERT_MODEL = "ai-forever/sbert_large_nlu_ru"
MARGIN_THRESHOLD = 0.12
BRANCHES = ["colloquial", "mixed"]

EMERGENCY = [
    (r"болит\s+серд[це]|серд[це]\s+(болит|колет|жмёт|жет|давит|тупит)|боль\s+в\s+сердце", "боль в области сердца"),
    (r"болит\s+за\s+грудиной|за\s+грудиной[^.;]{0,10}болит", "боль за грудиной"),
    (r"липк\w*\s+пот|холодн\w*\s+пот", "липкий или холодный пот — признак сердечного приступа"),
    (r"(жмёт|давит|сжимает|сдавливает|спазм)[^.;]{0,25}(груд|за\s+грудин)", "давление или сжатие в груди"),
    (r"(отда[её]т|прострел|простреливает)[^.;]{0,20}(лев[ууюю]\s+(руку|плечо)|лопатк)", "боль, отдающая в левую руку или лопатку"),
    (r"задыхаю|не\s+хватает\s+воздуха|удуш|не\s+могу\s+дышать", "внезапная нехватка воздуха"),
    (r"перекосило\s+лицо|онемел\w*\s+(лев|прав)\w*\s+(сторона|половина)|нарушилась\s+речь|не\s+могу\s+говорить", "возможные признаки инсульта"),
    (r"(рвот\w*\s+с\s+кровью|кровь\s+в\s+стуле|кровь\s+в\s+моче|кровотечение\s+не\s+останавливается)", "сильное кровотечение"),
    (r"потерял\w*\s+сознание|обморок\w*", "потеря сознания"),
    (r"температур\w+\s+(выше|под)\s+(39|40)|не\s+сбивается", "высокая температура, не сбиваемая жаропонижающими"),
]

def emergency_check(text):
    t = text.lower()
    return [msg for pat, msg in EMERGENCY if re.search(pat, t)]

def cache_key():
    parts = []
    for f in ("pipeline.joblib", "emb_all.npz", "labels.npy", "medbot.py"):
        try:
            parts.append(str(os.path.getmtime(f)))
        except OSError:
            parts.append("0")
    return "|".join(parts)

@st.cache_resource
def get_state(cache_key):
    from sentence_transformers import SentenceTransformer
    pipe = joblib.load("pipeline.joblib")
    sbert = SentenceTransformer(SBERT_MODEL)
    X = np.load("emb_all.npz")["X"].astype(np.float32)
    labels = [str(x) for x in np.load("labels.npy", allow_pickle=True)]
    n = min(len(labels), len(X))
    labels, X = labels[:n], X[:n]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    def embed(texts):
        e = np.asarray(sbert.encode(list(texts), convert_to_numpy=True), dtype=np.float32)
        if e.ndim == 1:
            e = e.reshape(1, -1)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    def similar_cases(text, k=5):
        e = embed([text])[0]
        e = e / (np.linalg.norm(e) + 1e-9)
        sims = Xn @ e
        idx = np.argsort(sims)[::-1]
        out, seen = [], set()
        for i in idx:
            if labels[i] in seen:
                continue
            seen.add(labels[i])
            out.append((labels[i], float(sims[i])))
            if len(out) >= k:
                break
        return out

    return {"pipe": pipe, "embed": embed, "similar": similar_cases}

def _branch_name(raw, pipe):
    return BRANCHES[int(np.asarray(raw).item())]

def predict(text):
    s = get_state(cache_key())
    emb = s["embed"]([text])
    # объединяем вероятности обеих веток: ошибка роутера не влияет на топ диагнозов
    scores = {}
    for name in BRANCHES:
        sub = s["pipe"][name]
        proba = sub["model"].predict_proba(emb)[0]
        for i, c in enumerate(sub["classes"]):
            c = str(c)
            p = float(proba[i])
            if p > scores.get(c, 0.0):
                scores[c] = p
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
    margin = top[0][1] - top[1][1]
    branch = _branch_name(s["pipe"]["router"].predict(emb)[0], s["pipe"])
    return {"branch": branch, "top": top, "margin": margin,
            "confident": margin >= MARGIN_THRESHOLD}

ADVICE = {
    "грипп": "Постельный режим 3–5 дней, обильное тёплое питьё, жаропонижающее при температуре выше 38.5.",
    "ОРВИ": "Покой, питьё, симптоматическое лечение. Обычно проходит за 5–7 дней.",
    "ангина": "Осмотр врача обязателен: при гнойной ангине нужны антибиотики. Полоскания тёплым солевым раствором — до визита.",
    "бронхит": "Тёплое питьё, покой. Если кашель с гнойной мокротой или дольше 2 недель — к терапевту.",
    "гастрит": "Дробное питание, исключить острое, кофе и алкоголь. Повторяется неделю — к гастроэнтерологу.",
    "цистит": "Обильное питьё, тепло на низ живота. Кровь в моче или температура — срочно к урологу.",
    "мигрень": "Тишина и темнота в приступе, вести дневник триггеров. Частые приступы — к неврологу.",
    "астма": "Ингалятор всегда под рукой. Одышка в покое или ночью — к пульмонологу.",
    "стенокардия": "Сесть, расстегнуть воротник. Боль за грудиной дольше 5 минут — вызывать скорую.",
    "аритмия": "Зафиксировать пульс и время приступа. Перебои с головокружением или обмороком — к кардиологу.",
    "пневмония": "Срочно к врачу: нужна рентгенография и антибиотики по назначению.",
}

def advice_for(diagnosis):
    return ADVICE.get(diagnosis, "Наблюдать 2–3 дня, пить достаточно жидкости, не переносить болезнь на ногах. Если симптомы нарастают — к терапевту.")
