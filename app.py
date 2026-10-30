import streamlit as st
from medbot import MedBot

st.set_page_config(page_title="Medical Classifier", page_icon="🩺")
st.title("🩺 Medical Classifier")
st.caption("Классификатор жалоб: разговорная и клиническая ветки, уверенность по марже")

@st.cache_resource(show_spinner="Загружаю модель (несколько минут при первом запуске)...")
def get_bot():
    return MedBot()

bot = get_bot()

text = st.text_area("Опишите жалобы:", height=120,
                    placeholder="Например: три дня болит горло и температура под 38")

if st.button("Определить", type="primary") and text.strip():
    r = bot.predict_top3(text)
    st.markdown(f"**{r['verdict']}**")
    st.caption(f"маржа уверенности: {r['margin']:.2f} · ветка: {r['branch']}")
    st.subheader("Топ-3 гипотезы")
    for name, prob in r["top3"]:
        st.write(f"- **{name}** — {prob:.0%}")
