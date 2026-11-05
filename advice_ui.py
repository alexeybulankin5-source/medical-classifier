import streamlit as st

DISCLAIMER = "Информация не заменяет консультацию врача."

def render_advice(r):
    if not r.get("top"):
        return
    conf = r.get("confidence")
    if conf not in ("high", "medium"):
        return  # low: уточняющий вопрос уже отработал в чате
    label = r.get("diagnosis") or r["top"][0][0]
    # --- волна 7б (D-008): три уровня триажа ---
    urgency = "emergency" if r.get("urgent") else r.get("urgency", "doctor_visit")
    if urgency == "emergency":
        st.error(f"🚨 КРИТИЧЕСКОЕ СОСТОЯНИЕ: {label}. Вызывайте скорую (103/112) немедленно.")
    elif urgency == "self_care":
        st.info(f"💡 {label}: с этим обычно справляются домашними мерами. "
                f"Если станет хуже или не пройдёт за 2–3 дня — обратитесь к врачу.")
    else:
        st.warning(f"📋 {label}: нужен осмотр врача. Запишитесь на приём, "
                   f"не откладывая, если симптомы усиливаются.")
    desc = r.get("description")
    if desc:
        st.markdown(f"**Что это:** {desc}")
    else:
        st.caption("Краткое описание состояния ещё готовится — ориентируйтесь на шаги подтверждения ниже.")
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
