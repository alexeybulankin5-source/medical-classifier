# -*- coding: utf-8 -*-
import os, re, numpy as np, joblib, streamlit as st

VERSION = "v13n-relsaniti"
SBERT_MODEL = "cointegrated/rubert-tiny2"
MARGIN_THRESHOLD = 0.10
SIM_EXACT = 0.92
BRANCHES = ["colloquial", "mixed"]
W_MODEL, W_SIM = 0.35, 0.65
SIM_MIN_HIGH = 0.60  # минимум похожести на базу для high/medium
GAP_MIN = 0.05     # отрыв лучшего класса от медианы по всем классам
MIN_WORDS = 2       # минимум содержательных слов для уверенного ответа

EMERGENCY = [
    (r"болит\s+серд[це]|серд[це]\s+(болит|колет|жмёт|жет|давит|тупит)|боль\s+в\s+сердце", "боль в области сердца"),
    (r"болит\s+за\s+грудиной|за\s+грудиной[^.;]{0,10}болит", "боль за грудиной"),
    (r"липк\w*\s+пот|холодн\w*\s+пот", "липкий или холодный пот — признак сердечного приступа"),
    (r"(жмёт|давит|сжимает|сдавливает|спазм)[^.;]{0,25}(груд|за\s+грудин)", "давление или сжатие в груди"),
    (r"(отда[её]т|прострел|простреливает)[^.;]{0,20}(лев[ууюю]\s+(руку|плечо)|лопатк)", "боль, отдающая в левую руку или лопатку"),
    (r"задыхаю|не\s+хватает\s+воздуха|удуш|не\s+могу\s+дышать", "внезапная нехватка воздуха"),
    (r"перекосило\s+лицо|онемел\w*\s+(лев|прав)\w*\s+(сторона|половина)|нарушилась\s+речь|не\s+могу\s+говорить", "возможные признаки инсульта"),
    (r"(рв\w*\s+(с\s+)?кровью|кровь\s+в\s+(стуле|моче)|кровотечение\s+не\s+останавливается|сильное\s+кровотечение)", "сильное кровотечение"),
    (r"потерял\w*\s+сознание|обморок\w*", "потеря сознания"),
    (r"температур\w+\s+(выше|под)\s+(39|40)|не\s+сбивается", "высокая температура, не сбиваемая жаропонижающими"),
    (r"(груд|сердц|грудин)[^.;]{0,40}(не\s+отпускает|не\s+проходит|полчаса|больше\s+часа)", "длительная боль или давление в груди — возможен инфаркт"),
    (r"нитроглицерин[^.;]{0,20}не\s+помог", "боль не снимается нитроглицерином — возможен инфаркт"),
    (r"страх\s+смерти[^.;]{0,40}(груд|сердц|тиск|давлени)|(груд|сердц|тиск|челюст)[^.;]{0,40}страх\s+смерти", "страх смерти с давящей болью — возможен инфаркт"),
]

CARDIO_RULES = [
    ("инфаркт", [r"(жм[ёе]т|давит|болит)[^.;]{0,30}не\s+отпускает",
                 r"(груд|сердц)[^.;]{0,30}(полчаса|не\s+отпускает)",
                 r"страх\s+смерти",
                 r"как\s+в\s+тисках",
                 r"нитроглицерин\s+не\s+помог",
                 r"(груд|сердц)[^.;]{0,20}началось\s+в\s+покое"], 0.30),
    ("стенокардия", [r"(болит|жм[ёе]т|давит|сжимает|сдавил|жж[ёе]т|спазм)[^.;]{0,25}(серд[це]|груд|грудин)",
                     r"за\s+грудиной",
                     r"боль\s+в\s+груди"], 0.15),
    ("аритмия", [r"(серд|пульс|груд)[^.;]{0,20}(замирает|спотыкается|колотится|перебо[ия])",
                 r"перебо[ия][^.;]{0,20}серд",
                 r"пульс\s+(то|скачет|пропускает)"], 0.20),
]

NEG_PATTERNS = [r"поясниц", r"\bспин", r"\bзуб", r"\bколен", r"\bстоп[аеу]", r"\bсустав", r"\bживот"]

def emergency_check(text):
    t = text.lower()
    return [msg for pat, msg in EMERGENCY if re.search(pat, t)]

def rule_bonus(text):
    t = text.lower()
    out = {}
    for key, pats, bonus in CARDIO_RULES:
        hits = sum(1 for p in pats if re.search(p, t))
        if hits:
            out[key] = bonus * min(hits, 2)
    return out

def negative_guard(text, rules):
    t = text.lower()
    if rules:
        return 0.0
    if any(re.search(p, t) for p in NEG_PATTERNS):
        return 0.25
    return 0.0

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

    def class_sims(text):
        e = embed([text])[0]
        sims = Xn @ e
        per = {}
        for lbl, s in zip(labels, sims):
            per.setdefault(lbl, []).append(float(s))
        out = {}
        for lbl, arr in per.items():
            arr.sort(reverse=True)
            out[lbl] = float(np.mean(arr[:3]))
        return out

    def exact_match(text):
        e = embed([text])[0]
        sims_all = Xn @ e
        i = int(np.argmax(sims_all))
        return labels[i], float(sims_all[i])

    return {"pipe": pipe, "embed": embed, "similar": similar_cases,
            "class_sims": class_sims, "exact_match": exact_match}

def _branch_name(raw, pipe):
    if isinstance(raw, str):
        return raw
    return BRANCHES[int(np.asarray(raw).item())]

def predict(text):
    s = get_state(cache_key())
    emb = s["embed"]([text])
    sims = s["class_sims"](text)
    rb = s["pipe"]["router"].predict_proba(emb)[0]
    rnames = [str(n) for n in s["pipe"]["router"].classes_]
    rw = {n: float(p) for n, p in zip(rnames, rb)}
    model_scores = {}
    for name in BRANCHES:
        sub = s["pipe"][name]
        proba = sub["model"].predict_proba(emb)[0]
        # вес ветки: основная ветка не глушится, побочная масштабируется роутером
        w = 1.0 if name == "colloquial" else rw.get(name, 0.0)
        for i, c in enumerate(sub["classes"]):
            c = str(c)
            p = float(proba[i]) * w
            if p > model_scores.get(c, 0.0):
                model_scores[c] = p
    classes = set(model_scores) | set(sims)
    scores = {c: W_MODEL * model_scores.get(c, 0.0) + W_SIM * sims.get(c, 0.0)
              for c in classes}
    rules = rule_bonus(text)
    for c in scores:
        for key, bonus in rules.items():
            if key in c.lower():
                scores[c] += bonus
    neg = negative_guard(text, rules)
    if neg > 0.0:
        scores = {c: v for c, v in scores.items()
                  if not any(k in c.lower() for k in ("стенокардия", "аритмия", "инфаркт"))}
    if "инфаркт" in rules:
        for c in scores:
            if "инфаркт" in c.lower():
                scores[c] = max(scores.values()) + 0.02
                break
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
    top = [(c, min(v, 0.99)) for c, v in top]
    margin = top[0][1] - top[1][1]
    top_rule = max((b for key, b in rules.items() if key in top[0][0].lower()), default=0.0)
    conf_thresh = 0.05 if neg > 0.0 else MARGIN_THRESHOLD
    confident = (margin >= conf_thresh or sims.get(top[0][0], 0.0) >= SIM_EXACT
                 or top_rule > 0.0)
    # NN-GUARD v2: точное совпадение с одной эталонной фразой на 0.97+
    nn_lbl, nn_sim = s["exact_match"](text)
    if nn_sim >= 0.97 and "инфаркт" not in rules:
        top = [(nn_lbl, max(top[0][1], min(nn_sim, 0.99)))] + \
              [(c, v) for c, v in top if c != nn_lbl]
        confident = True
    branch = _branch_name(s["pipe"]["router"].predict(emb)[0], s["pipe"])
    # шкала уверенности: high — отвечаем как есть, medium — ответ + альтернативы,
    # low — уточняющий вопрос
    sim_top = sims.get(top[0][0], 0.0)
    med_sim = float(np.median(list(sims.values()))) if sims else 0.0
    sim_gap = sim_top - med_sim
    n_words = len(re.findall(r"[а-яёa-z]{2,}", text.lower()))
    ok_sanity = n_words >= MIN_WORDS and sim_gap >= GAP_MIN
    if nn_sim >= 0.97:
        tier = "high"
    elif ok_sanity and (sim_top >= 0.85 or margin >= 0.08):
        tier = "high"
    elif ok_sanity and (margin >= 0.03 or sim_top >= 0.68):
        tier = "medium"
    else:
        tier = "low"
    return {"branch": branch, "top": top, "margin": margin,
            "confident": confident, "confidence": tier}

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
