
import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

kb = json.load(open("knowledge.json", encoding="utf-8-sig"))
data = np.load("class_embeddings.npz", allow_pickle=True)
E, names = data["embeddings"], list(data["names"])
enc = SentenceTransformer("intfloat/multilingual-e5-small")

RED_FLAGS = {"инфаркт миокарда", "стенокардия", "пневмоторакс", "тромбофлебит",
             "тромбоз глубоких вен", "аппендицит", "менингит", "тромбоэмболия лёгочной артерии"}

def classify(text, top=4):
    q = enc.encode([f"query: {text}"], normalize_embeddings=True)[0].astype(np.float32)
    order = np.argsort(-(E @ q))[:top]
    return [(names[i], float(E[i] @ q)) for i in order]

def get_recs(info):
    recs = info.get("recommendations") or info.get("recommendation") or {}
    if isinstance(recs, dict):
        return recs.get("specialist", ""), recs.get("advice", recs.get("text", ""))
    return "", str(recs)

URG = {
    "норма":      ("✅", "Не требует действий"),
    "наблюдение": ("👀", "Наблюдение — при жалобах к врачу"),
    "планово":    ("🗓", "Плановое обследование"),
    "срочно":     ("⚠️", "Обратиться в ближайшие дни"),
    "экстренно":  ("🚨", "Неотложная помощь"),
}

def load(fname):
    return json.load(open(fname, encoding="utf-8"))["items"]

def render_findings(items, prefix):
    c1, c2 = st.columns([2, 3])
    urg_filter = c1.multiselect(
        "Показать по срочности:", list(URG), default=["срочно", "экстренно"],
        format_func=lambda u: f"{URG[u][0]} {u}", key=f"{prefix}_urg",
    )
    search = c2.text_input("Поиск:", placeholder="например: ST, пневмо, ритм",
                           key=f"{prefix}_search").strip().lower()

    visible = items
    if urg_filter:
        visible = [i for i in visible if i["urgency"] in urg_filter]
    if search:
        hay = lambda i: (i["finding"] + " " + i["picture"] + " " + i["means"]).lower()
        visible = [i for i in visible if search in hay(i)]

    from collections import Counter
    cnt = Counter(i["urgency"] for i in visible)
    cols = st.columns(len(URG))
    for col, (u, (emoji, label)) in zip(cols, URG.items()):
        col.metric(f"{emoji} {u}", cnt.get(u, 0))

    if not visible:
        st.info("Ничего не найдено — ослабьте фильтры.")
        return

    st.divider()
    pick = st.selectbox(
        "Заключение:", [i["finding"] for i in visible],
        format_func=lambda f: next(f"{URG[i['urgency']][0]} {f}" for i in visible if i["finding"] == f),
        key=f"{prefix}_pick",
    )
    it = next(i for i in visible if i["finding"] == pick)
    emoji, label = URG[it["urgency"]]

    top = st.columns([1, 4])
    top[0].markdown(f"<div style='font-size:52px;text-align:center'>{emoji}</div>", unsafe_allow_html=True)
    with top[1]:
        st.subheader(it["finding"])
        st.markdown(f"**Срочность:** {label}")

    st.markdown("### 🔬 Что видно на исследовании")
    st.write(it["picture"])
    st.markdown("### 🧠 Что это значит")
    st.write(it["means"])

    if it["related"]:
        st.markdown("### 🔗 Связанные диагнозы из базы")
        for rel in it["related"]:
            info = kb.get(rel)
            if not info:
                continue
            with st.expander(rel):
                st.write(info.get("description", "описание не заполнено"))
                spec, advice = get_recs(info)
                if advice:
                    st.markdown("🩺 **Рекомендации:**")
                    st.write(advice)

    st.markdown("### 🩺 Что делать")
    st.success(it["advice"])
    if it["urgency"] == "экстренно":
        st.error("🚨 Эти признаки требуют неотложной помощи. Не откладывайте — вызывайте скорую.")
    elif it["urgency"] == "срочно":
        st.warning("⚠️ Обратитесь к врачу в ближайшие дни — состояние требует контроля.")

st.title("🩺 MedBot Duo")
tab_complaints, tab_analyses, tab_xray, tab_ecg = st.tabs(
    ["📝 Жалобы", "🧪 Анализы", "🦴 Рентген", "📈 ЭКГ"]
)

with tab_complaints:
    st.caption("Опишите жалобы своими словами — система сопоставит их с базой знаний")
    text = st.text_area("Жалобы:", placeholder="например: болит сердце и пот", key="complaint_text")
    if st.button("Анализировать", key="complaint_btn") and text.strip():
        results = classify(text)
        best, sim = results[0]
        info = kb.get(best, {})
        st.markdown(f"### Возможный диагноз: **{best}**")
        st.caption(f"Уверенность модели: {sim:.2f}")
        st.write(info.get("description", "описание не заполнено"))
        spec, advice = get_recs(info)
        if spec:
            st.markdown(f"🩺 **К какому врачу:** {spec}")
        if advice:
            st.markdown("🩺 **Рекомендации:**")
            st.write(advice)
        if len(results) > 1:
            st.markdown("🔍 **Возможные альтернативные диагнозы:**")
            for alt_name, alt_sim in results[1:]:
                st.markdown(f"- {alt_name} — сходство {alt_sim:.2f}")
        if best in RED_FLAGS:
            st.error("🚨 Ваша жалоба может указывать на экстренное состояние. "
                     "При выраженных симптомах не откладывайте — вызовите скорую помощь (103, 112).")
        else:
            st.info("ℹ️ Рекомендуется плановая консультация врача для подтверждения диагноза.")

with tab_analyses:
    st.caption("Раздел в разработке — здесь появится расшифровка анализов")

with tab_xray:
    st.caption("16 типовых находок: от нормы до экстренных состояний")
    render_findings(load("xray_findings.json"), prefix="xray")

with tab_ecg:
    st.caption("20 типовых заключений: что они значат и насколько срочно")
    render_findings(load("ecg_findings.json"), prefix="ecg")
