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
    if not case:
        return ""
    parts = ["КОНТЕКСТ СЛУЧАЯ ПОЛЬЗОВАТЕЛЯ (результат классификатора MedBot Duo):"]
    if case.get("text"):
        parts.append("Жалобы: " + str(case["text"]))
    if case.get("top"):
        parts.append("Вероятные диагнозы: " + str(case["top"]))
    if case.get("urgency"):
        parts.append("Уровень срочности: " + str(case["urgency"]))
    hits = assess(case.get("text", "") or "")
    if hits:
        parts.append("Красные флаги, обнаруженные правилами: " +
                     "; ".join(f["message"] for f in hits))
    return "\n".join(parts)

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
    if not case:
        st.info("Сначала опишите жалобы во вкладке «Жалобы». Второе мнение даётся по вашему случаю, а не в вакууме.")
        return

    st.markdown(
        f"**Жалобы:** {str(case.get('text', ''))[:200]}  \n"
        f"**Вердикт классификатора:** {case.get('top', '—')}  \n"
        f"**Срочность:** {case.get('urgency', '—')}")
    st.divider()

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

    user_msg = st.chat_input("Спросите о своём случае")
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
                 case_ctx={"urgency": case.get("urgency", ""),
                           "top": case.get("top", "")},
                 model_version=model_version)
