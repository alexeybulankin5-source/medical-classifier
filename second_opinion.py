# -*- coding: utf-8 -*-
"""Вкладка «Второе мнение»: чат с YandexGPT по контексту случая из «Жалоб».
Второе мнение опирается на вердикт классификатора, а не отвечает с нуля.
ГРАНИЦЫ: ты не можешь записать пациента на приём, узнать расписание, телефоны поликлиник и оформить вызов врача на дом. На такие вопросы отвечай коротко и честно: «записаться и узнать телефоны я не могу», и сразу подсказывай путь: запись к врачу — через Госуслуги (gosuslugi.ru, раздел «Моё здоровье»), вызов врача на дом — по телефону своей поликлиники или через Госуслуги, неотложная помощь — 103 или 112. Медицинские вопросы по случаю — твоя основная работа, отвечай на них полно."""
import os, json, urllib.request
import streamlit as st

from safety import assess
from chat_log import log_turn

_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

def _api_key():
    return (os.environ.get("YC_API_KEY") or os.environ.get("YC_IAM_TOKEN") or "").strip()

def _model_name():
    return os.environ.get("YC_MODEL", "yandexgpt-lite")

SYSTEM_PROMPT = """Ты — дружелюбный медицинский ассистент в приложении MedBot Duo. Ты помогаешь человеку разобраться в его симптомах и уже полученных результатах классификатора. Ты НЕ ставишь диагнозов и НЕ заменяешь врача.

ПРАВИЛА ТОНУ:
- Говори спокойно и по-человечески, как заботливый знакомый медик. Без паники, без восклицательных знаков, без запугивания.
- Отвечай кратко: 3-6 предложений или короткий список. Не пиши лекций.
- Если пользователю тревожно — сначала успокой одним предложением, потом по делу.

ПРАВИЛА СРОЧНОСТИ (главное!):
- Слово «скорая» и экстренные рекомендации используй ТОЛЬКО если в сообщении пользователя, в контексте случая или в перечисленных красных флагах есть явные признаки: боль в груди с одышкой, кровотечение, потеря сознания или обморок, признаки инсульта (асимметрия лица, слабость руки/ноги, нарушение речи), затруднённое дыхание, температура выше 39.5 с сыпью или ригидностью затылка, судороги, неукротимая рвота с признаками обезвоживания, внезапная острая «разрывающая» боль.
- Во всех остальных случаях НИКОГДА не призывай вызывать скорую. Давай градуированный ответ: что это похоже на, что сделать сейчас, за какими симптомами следить, когда записаться к врачу (например, «если не пройдёт за 2-3 дня» или «если станет хуже»).
- Не дублируй экстренные предупреждения, если контекст случая уже не экстренный.

ПРАВИЛА ДИАЛОГА:
- Если в вопросе не хватает важных деталей (давность, интенсивность, что уже пробовали, сопутствующие симптомы) — задай 1-2 коротких уточняющих вопроса ПРЕЖДЕ чем давать рекомендации. Не задавай больше двух вопросов за раз.
- Опирайся на контекст случая, если он есть: учитывай диагноз и уровень срочности из классификатора, можешь вежливо обсудить вердикт («согласна с вероятным гастроэнтеритом, но при тёмной моче стоит проверить обезвоживание»).
- Если в контексте перечислены красные флаги — тогда и только тогда упомяни скорую, одной фразой, и объясни какой признак на это указывает.

Запрещено: ставить окончательный диагноз, назначать рецептурные препараты с дозировками, обесценивать жалобы («это ерунда»), игнорировать вопрос ради общей инструкции."""

def _case_text(case):
    parts = []
    if case:
        parts.append("КОНТЕКСТ СЛУЧАЯ ПОЛЬЗОВАТЕЛЯ (результат классификатора MedBot Duo):")
        if (case or {}).get("text"):
            parts.append("Жалобы: " + str(case["text"]))
        if (case or {}).get("top"):
            parts.append("Вероятные диагнозы: " + str(case["top"]))
        if (case or {}).get("urgency"):
            parts.append("Уровень срочности: " + str(case["urgency"]))
        hits = assess((case or {}).get("text", "") or "")
        if hits:
            parts.append("Красные флаги, обнаруженные правилами: " +
                         "; ".join(f["message"] for f in hits))
    _lc = st.session_state.get("lab_context")
    if _lc:
        _lv = _lc.get("values") or {}
        _lp = _lc.get("patterns") or []
        if _lv or _lp:
            _lt = "Анализы крови (распознанные показатели): " + "; ".join(str(k) + " = " + str(v) for k, v in _lv.items())
            if _lp:
                _lt += ". Обнаруженные паттерны: " + ", ".join(map(str, _lp))
            _lt += ". Интерпретируй показатели в пределах своей роли: объясняй значения и отклонения от нормы, не ставь диагноз по анализам, при сомнении направляй к врачу."
            if len(_lt) > 2000:
                _lt = _lt[:2000]
            parts.append(_lt)
    _xt = st.session_state.get("xray_in") or ""
    if _xt.strip():
        parts.append("Заключение рентгена: " + _xt.strip())
    _et = st.session_state.get("ecg_in") or ""
    if _et.strip():
        parts.append("Заключение ЭКГ: " + _et.strip())
    return "\n".join(parts) if parts else ""
def ask_yandex_gpt(system_text, user_text, history=None,
                   temperature=0.3, max_tokens=600):
    key = _api_key()
    folder = os.environ.get("YC_FOLDER_ID", "").strip()
    if not key or not folder:
        return "Ключи не настроены: задайте YC_API_KEY (или YC_IAM_TOKEN) и YC_FOLDER_ID и перезапустите приложение."
    msgs = [{"role": "system", "text": system_text}]
    for h in (history or []):
        msgs.append({"role": h["role"], "text": h["text"]})
    msgs.append({"role": "user", "text": user_text})
    body = {"modelUri": "gpt://" + folder + "/" + _model_name(),
            "completionOptions": {"temperature": temperature,
                                  "maxTokens": max_tokens},
            "messages": msgs}
    req = urllib.request.Request(_URL, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Api-Key " + key,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        return data["result"]["alternatives"][0]["message"]["text"]
    except urllib.error.HTTPError as e:
        return "Ошибка API: HTTP " + str(e.code) + " " + e.read().decode()[:300]
    except Exception as e:
        return "Ошибка соединения: " + str(e)

def render_second_opinion():
    st.subheader("💬 Второе мнение")
    with st.expander("☎️ Полезное: запись и вызов врача (всегда под рукой)"):
        st.markdown(
            "- **Неотложная помощь:** 103 или 112\n"
            "- **Запись к врачу и вызов на дом:** через [Госуслуги](https://www.gosuslugi.ru), "
            "раздел «Моё здоровье»\n"
            "- **Телефон вашей поликлиники:** на стойке регистрации, на информационном стенде "
            "или на сайте вашей поликлиники — я его не знаю и не могу передать вызов")
    case = st.session_state.get("last_case")
    _lc_any = st.session_state.get("lab_context") or st.session_state.get("xray_in") or st.session_state.get("ecg_in")
    if not case and not _lc_any:
        st.info("Сначала опишите жалобы во вкладке «Жалобы» или разберите анализ, снимок или ЭКГ. Второе мнение даётся по вашему случаю, а не в вакууме.")
        return

    _hdr = []
    if case:
        _hdr.append(f"**Жалобы:** {str((case or {}).get('text', ''))[:200]}")
        _hdr.append(f"**Вердикт классификатора:** {(case or {}).get('top', '—')}")
        _hdr.append(f"**Срочность:** {(case or {}).get('urgency', '—')}")
    else:
        _hdr.append("**Контекст случая:** разобранные данные без жалоб (анализы / снимок / ЭКГ)")
    st.markdown("  \n".join(_hdr))
    st.divider()
    _render_brief(case)

    model_version = "unknown"
    try:
        model_version = json.load(open("model_version.json"))["model_version"]
    except Exception:
        pass

    if "so_messages" not in st.session_state:
        st.session_state.so_messages = []

    system_text = SYSTEM_PROMPT
    ctx = _case_text(case)
    if ctx:
        system_text += "\n\n" + ctx


    for m in st.session_state.so_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["text"])

    _chips()
    user_msg = st.chat_input("Спросите о своём случае")
    if not user_msg:
        user_msg = st.session_state.pop("so_pending", None)
    if user_msg:
        st.session_state.so_messages.append({"role": "user", "text": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)
        answer = ask_yandex_gpt(system_text, user_msg,
                                history=st.session_state.so_messages[:-1])
        st.session_state.so_messages.append({"role": "assistant", "text": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
        log_turn(user_msg, answer,
                 case_ctx={"urgency": (case or {}).get("urgency", ""),
                           "top": (case or {}).get("top", "")},
                 model_version=model_version)


# --- расширение: разбор случая, чипсы, экспорт ---
def _norm(s):
    return str(s).strip().lower().replace("ё", "е")

def _ada_answers():
    v = st.session_state.get("adaptive_answers")
    if isinstance(v, dict) and v:
        return v
    for _k, _val in st.session_state.items():
        if isinstance(_val, dict) and _val and any(
                str(_x).startswith("head") for _x in _val):
            return _val
    return {}

def _canon_of(ad, key, a):
    m = getattr(ad, "_ANSWER_CANON", None)
    if isinstance(m, dict):
        return m.get((key, a), a)
    if isinstance(m, (list, tuple)):
        for it in m:
            if it[0] == key and _norm(it[1]) == a:
                return it[2]
    return a

def _delta_args(answers, names):
    import adaptive as ad
    pro, con = {}, {}
    src_adj = getattr(ad, "HEAD_ADJUST", None) or {}
    entries = []  # (ключ_вопроса, фраза, [(класс, дельта)])
    if isinstance(src_adj, dict):
        for k, val in src_adj.items():
            if isinstance(k, (list, tuple)) and len(k) >= 2:
                entries.append((k[0], k[1], val or []))
            else:
                if isinstance(val, dict):
                    for ph, dd in val.items():
                        entries.append((k, ph, dd or []))
                elif isinstance(val, (list, tuple)) and val and isinstance(val[0], (list, tuple)) and len(val[0]) >= 2 and isinstance(val[0][1], (list, tuple)):
                    entries.append((k, val[0][0], val[0][1]))
    elif isinstance(src_adj, (list, tuple)):
        for item in src_adj:
            if item and len(item) >= 2:
                entries.append((item[0], item[1], item[2] if len(item) > 2 else []))
    for key, phrase, deltas in entries:
        if key not in answers:
            continue
        raw = answers[key]
        parts = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        cands = set()
        can = getattr(ad, "_ANSWER_CANON", None) or {}
        for p in parts:
            pn = _norm(p)
            if pn:
                cands.add(pn)
                cv = can.get((key, p)) or can.get((key, pn))
                if cv:
                    cands.add(_norm(cv))
        if not any(c == _norm(phrase) for c in cands):
            continue
        for entry in deltas:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                cls, d = entry[0], entry[1]
            elif isinstance(src_adj, dict) and isinstance(k, (list, tuple)):
                continue
            else:
                continue
            if cls in names:
                tgt = pro if float(d) > 0 else con
                tgt[cls] = tgt.get(cls, 0.0) + float(d)
    return pro, con

def _doctor_questions(case):
    qs = ["Как давно появились симптомы и что меняло их?",
          "Что уже принимали и помогло ли?",
          "Есть ли хронические болезни и аллергии?"]
    try:
        for fn in ("knowledge.json", "kb.json", "medical_kb.json"):
            try:
                kb = json.load(open(fn, encoding="utf-8"))
                break
            except Exception:
                kb = {}
        kb = kb.get("classes", kb) if isinstance(kb, dict) else {}
        top = (case or {}).get("top") or []
        name = top[0][0] if top and isinstance(top[0], (list, tuple)) else None
        if name and name in kb:
            spec, adv = (kb[name].get("recommendations") or {}).get(
                "specialist", "")
            if spec:
                qs.insert(0, "К какому специалисту ведёт этот профиль: "
                          + str(spec) + "?")
    except Exception:
        pass
    return qs

def _render_brief(case):
    answers = _ada_answers()
    top = (case or {}).get("top") or []
    names = [t[0] if isinstance(t, (list, tuple)) else str(t) for t in top]
    pro, con = _delta_args(answers, set(names))
    with st.expander("🧾 Разбор случая: аргументы за и против", expanded=True):
        if not names:
            st.info("Классификация ещё не запускалась — запустите её во вкладке «Жалобы».")
        for t in top:
            if isinstance(t, (list, tuple)) and len(t) > 1:
                st.markdown("- **" + str(t[0]) + "** — сходство "
                            + str(round(t[1], 2)))
            else:
                st.markdown("- " + str(t))
        if answers:
            st.markdown("**Ответы на уточняющие вопросы:**")
            for k, v in answers.items():
                st.markdown("- " + str(k) + ": " + str(v))
        else:
            st.info("Уточняющие вопросы не отвечены — аргументов формы нет.")
        if pro or con:
            st.markdown("**Аргументы правил (за / против):**")
            for n in names:
                a, b = pro.get(n), con.get(n)
                if a or b:
                    pa = "+" + str(round(a, 2)) if a else "—"
                    pb = str(round(b, 2)) if b else "—"
                    st.markdown("- " + n + ": за " + pa + " / против " + pb)
        st.markdown("**Стоит спросить у врача:**")
        for q in _doctor_questions(case):
            st.markdown("❓ " + q)
        st.caption("Это структурированная сводка ваших данных, "
                   "а не заключение специалиста.")
        md = ["# Случай для врача (MedBot Duo)", ""]
        if case:
            md += ["Жалобы: " + str(case.get("text", "")),
                   "Вердикт: " + str(case.get("top", "")),
                   "Срочность: " + str(case.get("urgency", "")), ""]
        for k, v in answers.items():
            md.append("- " + str(k) + ": " + str(v))
        for n in names:
            a, b = pro.get(n), con.get(n)
            if a or b:
                md.append("- " + n + ": за " + str(a) + " / против " + str(b))
        md += ["", "Вопросы врачу:", ""]
        md += ["- " + q for q in _doctor_questions(case)]
        st.code(chr(10).join(md), language="markdown")

CHIP_QS = ["Объясните вердикт простыми словами",
           "Какие симптомы тревожные?",
           "Что мне делать сейчас?",
           "К какому врачу записаться?"]

def _chips():
    cols = st.columns(len(CHIP_QS))
    hit = None
    for col, q in zip(cols, CHIP_QS):
        if col.button(q, key="so_chip_" + str(CHIP_QS.index(q))):
            hit = q
    if hit:
        st.session_state["so_pending"] = hit
        st.rerun()
