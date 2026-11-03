import streamlit as st

DISCLAIMER = "Информация не заменяет консультацию врача."

def render_advice(r):
    if not r.get("top"):
        return
    conf = r.get("confidence")
    if conf not in ("high", "medium"):
        return  # low: уточняющий вопрос уже отработал в чате
    label = r.get("diagnosis") or r["top"][0][0]
    if r.get("urgent"):
        st.error(f"🚨 КРИТИЧЕСКОЕ СОСТОЯНИЕ: {label}. Вызывайте скорую (103/112) немедленно.")
    desc = r.get("description")
    if desc:
        st.markdown(f"**Что это:** {desc}")
    confirm = r.get("confirm") or []
    if confirm:
        st.markdown("**Как подтвердить:**")
        for c in confirm:
            st.markdown(f"- {c}")
    if r.get("draft_advice"):
        st.caption("Черновик описания — будет уточнён.")
    if r.get("guard"):
        st.caption(f"⚖ {r['guard']}")
    st.caption(DISCLAIMER)
