import json
import os

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- ПУТИ К ФАЙЛАМ ---
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(PROJECT_PATH, 'knowledge.json')
LOG_PATH = os.path.join(PROJECT_PATH, 'unknown_queries.txt')
EMB_PATH = os.path.join(PROJECT_PATH, 'class_embeddings.npz')

# --- ЗАГРУЗКА РЕСУРСОВ ---
@st.cache_resource
def load_resources(_kb_md5=''):
    global kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS
    
    try:
        # Загрузка базы знаний
        with open(KB_PATH, 'r', encoding='utf-8-sig') as f:
            kb = json.load(f)
            
        # Загрузка векторов
        if os.path.exists(EMB_PATH):
            data = np.load(EMB_PATH, allow_pickle=True)
            CLASS_EMBEDDINGS = data['embeddings']
            CLASS_NAMES = data['names'].tolist()
        else:
            st.error("❌ Файл class_embeddings.npz не найден")
            st.stop()
            
        # Инициализация энкодера
        encoder = SentenceTransformer('intfloat/multilingual-e5-small')
        
    except Exception as e:
        st.error(f"Ошибка: {e!s}")
        raise
        
    return kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS

import hashlib as _h_cache
_kb_md5 = _h_cache.md5(open(KB_PATH, "rb").read()).hexdigest()[:8]
kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS = load_resources(_kb_md5)

# --- ВЕРСИЯ ДАННЫХ (диагностика расхождений) ---
import hashlib as _hl
_kb_md5 = _hl.md5(open(KB_PATH, "rb").read()).hexdigest()[:8]
try:
    _n_drafts = sum(1 for v in kb.get("classes", kb).values()
                    if "черновик" in (v.get("desc") or "").lower())
except Exception:
    _n_drafts = -1
try:
    _n_recs = sum(1 for v in kb.get("classes", kb).values() if v.get("recommendations"))
    _n_cls = len(kb.get("classes", kb))
except Exception as _e:
    _n_recs = f'ERR {_e!r}'
    _n_cls = -1
st.sidebar.caption(f"KB-версия: {_kb_md5} • классов: {_n_cls} • с рекомендациями: {_n_recs}")

# --- ДИСАМБИГУАЦИЯ ДАВЛЕНИЯ (правило сильнее сходств) ---
import re as _re
_CONS = "бвгджзклмнпрстфхцчшщbcdfghjklmnpqrstvwxz"
_RUN4 = _re.compile("[" + _CONS + "]{4,}")
_KEYBOARD_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm", "йцукенгшщзхъ", "фывапролдж", "ячсмитьбю")
_RUN3 = _re.compile("[" + _CONS + "]{3}")

def _gib_word(w):
    if _RUN4.search(w):
        return True
    m = _RUN3.search(w)
    if m:
        cons = sum(1 for c in w if c in _CONS)
        if cons / len(w) >= 0.75:
            return True
    wl = w.lower()
    for row in _KEYBOARD_ROWS:
        if len(wl) >= 5 and all(c in row for c in wl):
            return True
    return False

def is_gibberish_query(t):
    words = _re.findall(r"[а-яёa-z]{3,}", t.lower())
    if not words:
        return True
    return all(_gib_word(w) for w in words)
def _bp_override(text, top_names):
    t = text.lower()
    m = _re.search(r"(\d{2,3})\s*(?:на|/|–|-)\s*(\d{2,3})", t)
    if m:
        hi, lo = int(m.group(1)), int(m.group(2))
        if hi >= 140 or lo >= 90:
            crisis = ["разрывается", "распирает", "тошнит", "тошнота", "рвот", "мушки",
                      "пелена", "шумит в ушах", "звон в ушах", "не сбивается", "не сбил",
                      "давит в груди", "боль в груди", "перекос", "онемел",
                      "речь нарушилась", "смазан", "двоится", "мутит"]
            if any(w in t for w in crisis):
                return "гипертонический криз" if "гипертонический криз" in top_names else None
            return "гипертония" if "гипертония" in top_names else None
        if hi <= 110:
            return "гипотония" if "гипотония" in top_names else None
    if any(w in t for w in ["повышенное давление", "высокое давление",
                            "подскочило давление", "давление подскочило"]):
        return "гипертония" if "гипертония" in top_names else None
    if any(w in t for w in ["низкое давление", "пониженное давление",
                            "давление упало", "упало давление"]):
        return "гипотония" if "гипотония" in top_names else None
    return None

# --- КРАСНОФЛАЖКОВЫЙ СЛОЙ (правила важнее сходств) ---
_REDFLAG_RULES = [
    (["отняла", "отнялась", "онемела", "не слушается", "парализова"],
     ["речь", "говорить", "говорю", "лицо", "перекос", "век", "рука", "нога", "половин"],
     "инсульт"),
    (["речь нарушилась", "не могу говорить", "не могу сказать", "речь нарушил",
      "понимать перестал", "слова не выговариваю"],
     [], "инсульт"),
    (["свет не выношу", "светобоязнь", "боли в глазах от света", "не могу опустить подбородок",
      "ригидность", "сыпь не бледнеет", "звездочк"],
     ["температура", "температур", "39", "38.5", "38,5", "рвота", "тошнит"],
     "менингит"),
    (["давит в груди", "жжёт за грудиной", "жгучая боль за грудиной", "боль за грудиной"],
     ["отдаёт в руку", "отдает в руку", "в левую руку", "в челюсть", "в лопатку",
      "холодный пот", "прошло больше 20 минут", "нитроглицерин не помог"],
     "инфаркт миокарда"),
    (["внезапная одышка", "стало тяжело дышать", "не хватает воздуха в покое",
      "задыхаюсь внезапно"],
     ["кровохарканье", "боль при вдохе", "посинели губ", "синие губы", "обморок",
      "после операции", "после перелёта", "после перелета", "долго лежал", "долго лежала"],
     "тромбоэмболия лёгочной артерии"),
    (["задержка", "задержки", "нет месячных", "месячных нет",
      "не пришли месячные", "задержка месячных"],
     ["живот", "боли", "боль", "болит", "внизу", "тянет", "кровянист", "мажет"],
     "внематочная беременность"),
    (["справа внизу живота", "правая нижняя часть живота", "правый низ живота",
      "боль смещается в правую", "у пупка и смещается"],
     ["тошнит", "тошнота", "рвота", "температур", "нет аппетита"],
     "аппендицит"),
    (["отёк губ", "отек губ", "опухли губы", "опухли губ", "отёк языка", "опух язык",
      "отёк горла", "отек горла", "опухло горло", "сдавило горло", "не могу вдохнуть",
      "хрипы при дыхании", "свистит в горле"],
     ["после еды", "после орех", "после укуса", "после укола", "после лекарства",
      "после препарата", "принял таблетку", "приняла таблетку", "осы", "пчел"],
     "анафилаксия"),
    (["рвота с кровью", "рвет кровью", "рвёт кровью", "кровь в рвоте", "рвотные массы с кровью",
      "чёрный стул", "черный стул", "стул чёрного цвета", "стул дёгтем", "дегтеобразный стул",
      "кофейной гущей", "мелена"],
     [],
     "кровотечение ЖКТ"),
]
def _redflag_check(text, top_names):
    t = text.lower()
    warn = None
    for req, extra, target in _REDFLAG_RULES:
        if any(w in t for w in req) and (not extra or any(w in t for w in extra)):
            if target in top_names:
                return target, None
            if warn is None:
                warn = target
    return None, warn


# --- ЛОГИРОВАНИЕ НЕИЗВЕСТНЫХ ЗАПРОСОВ ---
def log_unknown_query(query: str):
    if not query.strip(): 
        return False
    if is_gibberish_query(query):
        try:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, 'a', encoding='utf-8') as f:
                f.write("[gibberish] " + query + chr(10))
            st.session_state.setdefault("unknown_log", []).append("[gibberish] " + query)
            return True
        except Exception as e:
            print(f"Ошибка записи: {e}")
            return False
    
    try:
        user_emb = encoder.encode(["query: " + query])[0]
        sims = cosine_similarity([user_emb], CLASS_EMBEDDINGS)[0]
        
        if np.max(sims) < 0.65:
            try:
                os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
                with open(LOG_PATH, 'a', encoding='utf-8') as f:
                    f.write(f"{query}\n")
                st.session_state.setdefault("unknown_log", []).append(query)
                return True
            except Exception as e:
                print(f"Ошибка записи: {e}")
                return False
    except Exception as e:
        st.error(f"Ошибка обработки: {e!s}")
        return False
    return False

import re as _are

def _allergy_override(text, top5):
    """Отёк губ/лица/языка/глотки — приоритет отёку Квинке, а не "заедам" по слову губы."""
    t = text.lower().replace("ё", "е")
    sw = _are.search(r"(опух|припух|распух|отек)", t)
    zone = _are.search(r"(губ|язык|лица|лицо|глот|горл|век)", t)
    if not (sw and zone):
        return None
    cand = [c for c in CLASS_NAMES if "квинке" in c.lower().replace("ё", "е")]
    return cand[0] if cand else None

# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---
st.title("🩺 MedBot Duo")
st.markdown("Интеллектуальная система первичной диагностики")

# ==== RAG-слой источников (волна 10) ====
import json as _rj, numpy as _rnp
try:
    _rag_kb = _rj.load(open("rag_knowledge.json", encoding="utf-8"))
    _rag_e = _rnp.load("rag_embeddings.npz", allow_pickle=True)
    _rag_X = _rag_e["embeddings"].astype(_rnp.float32)
    _rag_X /= (_rnp.linalg.norm(_rag_X, axis=1, keepdims=True) + 1e-9)
    _rag_names = [str(x) for x in _rag_e["names"]]
    _RAG_READY = True
except Exception:
    _RAG_READY = False

@st.cache_resource
def _load_rag_e5():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("intfloat/multilingual-e5-small")

def _rag_hits(text, dx=None, k=2):
    """Подбор источников по диагнозу и жалобам: детерминированные правила, а не косинус.
    e5-small на этой базе давал нерелевантные совпадения (замеры: 'болит горло' -> пиелонефрит)."""
    t = (str(dx or "") + " " + str(text or "")).lower().replace("ё", "е")
    rules = [
        (("квинке", "анафилакс", "крапивниц", "отек губ", "отек лица", "отек языка"), "Крапивница и анафилаксия"),
        (("тиреоидит", "хашимото"), "Аутоиммунный тиреоидит"),
        (("гипотиреоз",), "Гипотиреоз"),
        (("тиреотоксикоз", "базедов", "грейвс"), "Тиреотоксикоз"),
        (("криз",), "Гипертонический криз"),
        (("гипертенз", "гипертони"), "Артериальная гипертензия"),
        (("инфаркт",), "Инфаркт миокарда"),
        (("ибс", "стенокард", "коронарн"), "ИБС и стенокардия"),
        (("фибрилляц", "мерцательн", "аритм"), "Фибрилляция предсердий"),
        (("инсульт",), "Инсульт"),
        (("тромбоэмбол", "тэла", "легочной артерии"), "Тромбоэмболия лёгочной артерии"),
        (("пневмони", "воспаление легких"), "Пневмония"),
        (("орви", "грипп", "фарингит", "тонзиллит", "ринит", "ларингит", "трахеит", "ложный круп", "ангина", "сопли", "насморк", "простуд"), "ОРВИ и грипп"),
        (("диабет",), "Сахарный диабет 2 типа"),
        (("анемия", "ферритин"), "Железодефицитная анемия"),
        (("гэрб", "рефлюкс", "изжог"), "ГЭРБ"),
        (("язвен", "хеликобактер", "гастрит", "дуоденит"), "Язвенная болезнь и хеликобактер"),
        (("раздраженнного кишечника", "срк"), "Синдром раздражённого кишечника"),
        (("мигрень",), "Мигрень"),
        (("головной боли напряжения",), "Головная боль напряжения"),
        (("артроз", "гонартроз", "коксартроз", "остеоартр"), "Остеоартроз"),
        (("цистит",), "Цистит"),
        (("пиелонефрит", "воспаление почек"), "Пиелонефрит"),
        (("мочекаменн", "уролитиаз", "почечной колик"), "Мочекаменная болезнь"),
        (("дерматит", "экзем"), "Атопический дерматит"),
        (("астма",), "Бронхиальная астма"),
        (("хобл", "эмфизем", "хронический бронхит"), "ХОБЛ"),
        (("паническ", "тревож"), "Тревожные расстройства и панические атаки"),
        (("депресс",), "Депрессия"),
        (("бессонниц", "инсомни", "плохой сон", "не могу уснуть", "нарушение сна", "нарушился сон"), "Бессонница"),
        (("лактаз", "непереносимость молочн"), "Лактазная недостаточность"),
        (("целиаки", "глютен"), "Целиакия"),
        (("поликистоз", "спкя"), "СПКЯ"),
        (("атеросклероз", "дислипидем", "холестерин"), "Атеросклероз и дислипидемия"),
        (("остеопор",), "Остеопороз"),
    ]
    hits = []
    for keys, topic in rules:
        if any(w in t for w in keys):
            hits.append(topic)
        if len(hits) >= k:
            break
    return [(h, 1.0) for h in hits]

def _tiebreak_pick(sims, text):
    """При ничьей argmax (разница < 0.01) предпочитаем класс, чьё имя дословно встречается в запросе."""
    import re as _re2
    order = np.argsort(np.asarray(sims))[::-1]
    i0 = int(order[0])
    if len(order) < 2 or float(sims[i0]) - float(sims[order[1]]) > 0.01:
        return i0
    toks = set(_re2.findall(r"[а-яёa-z]{4,}", (text or "").lower()))
    best, best_hits = i0, -1
    for i in order[:10]:
        if float(sims[i0]) - float(sims[int(i)]) > 0.01:
            break
        nm = CLASS_NAMES[int(i)].lower()
        hits = sum(1 for w in toks if w in nm)
        if hits > best_hits:
            best, best_hits = int(i), hits
    return best


def _tonsil_override(text, top5):
    """Маркеры ангины сильнее сходства: миндалины и белый налёт -> тонзиллит/ангина."""
    t = (text or "").lower().replace("ё", "е")
    if not any(w in t for w in ("миндалин", "белый налет", "гнойные пробки", "пробки в горле", "налет на гландах", "гной в горле")):
        return None
    for c in top5:
        cl = (c or "").lower()
        if "тонзилл" in cl or "ангин" in cl:
            return c
    for c in CLASS_NAMES:
        cl = (c or "").lower()
        if "тонзилл" in cl or "ангин" in cl:
            return c
    return None


def _symptom_override(text, top5):
    """Симптомные маркеры сильнее сходства: мигрень, диабет, отёк Квинке."""
    t = (text or "").lower().replace("ё", "е")

    def pick(sub):
        for c in top5:
            if sub in (c or "").lower():
                return c
        for c in CLASS_NAMES:
            if sub in (c or "").lower():
                return c
        return None

    mig = sum(1 for w in ("половин", "с одной стороны", "односторонн", "светобоязн", "фотофобия", "пульсирующ") if w in t)
    if mig >= 2:
        c = pick("мигрень")
        if c:
            return c
    diab = sum(1 for w in ("жажда", "полиури", "частое мочеиспускание", "много пью") if w in t)
    if diab >= 2:
        c = pick("диабет")
        if c:
            return c
    swell = any(w in t for w in ("отек", "опухли", "распухли", "отекает"))
    face = any(w in t for w in ("губ", "лиц", "язык", "гортан", "глотк"))
    if swell and face:
        c = pick("квинке")
        if c:
            return c
    if any(w in t for w in ("понос", "диарея", "жидкий стул", "расстройство кишечника", "частый стул")):
        c = pick("диарея") or pick("кишечная инфекция") or pick("гастроэнтерит")
        if c:
            return c
    return None


tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Жалобы", "🧪 Анализы", "🦴 Снимки (рентген)", "📈 ЭКГ", "📊 Аналитика"])

with tab1:
    st.header("📝 Анализ жалоб")
    user_input = st.text_area("Опишите жалобы:", height=150)
    
    if st.button("Определить диагноз"):
        if user_input.strip():
            with st.spinner("Анализ..."):
                try:
                    # Кодируем запрос
                    user_emb = encoder.encode(["query: " + user_input])[0]
                    sims = cosine_similarity([user_emb], CLASS_EMBEDDINGS)[0]
                    
                    best_idx = _tiebreak_pick(sims, user_input)
                    best_score = sims[best_idx]
                    if is_gibberish_query(user_input):
                        best_score = 0.0
                    found_class_id = CLASS_NAMES[best_idx]

                    top5 = [CLASS_NAMES[i] for i in np.argsort(sims)[::-1][:5]]
                    _override, _redflag_missing = _redflag_check(user_input, top5)
                    if not _override:
                        _override = _bp_override(user_input, top5)
                        if not _override:
                            _override = _allergy_override(user_input, top5)
                            if not _override:
                                _override = _tonsil_override(user_input, top5)
                                if not _override:
                                    _override = _symptom_override(user_input, top5)
                    if _override and _override != found_class_id:
                        best_idx = CLASS_NAMES.index(_override)
                        best_score = sims[best_idx]
                        found_class_id = _override
                    if _redflag_missing:
                        st.error("⚠️ Похожие симптомы встречаются при опасных состояниях "
                                 f"(например, {_redflag_missing}). Если они выражены или нарастают — "
                                 "немедленно обратитесь к врачу или вызовите скорую помощь.")
                    
                    # Логируем неизвестный запрос
                    log_unknown_query(user_input)
                    
                    # --- ВЫВОД РЕЗУЛЬТАТОВ ---
                    if best_score > 0.65:
                        data = kb[found_class_id]

                        is_urgent = data.get('urgent', False)
                        diagnosis_text = data.get('desc', 'Описание диагноза не найдено в базе знаний.')
                        
                        st.subheader(f"Возможный диагноз: {found_class_id}")
                        st.write(diagnosis_text)
                        _lctx = st.session_state.get("lab_context")
                        if _lctx:
                            _pats = _lctx.get("patterns", [])
                            _vals = _lctx.get("values", {})
                            if _pats:
                                st.info("🧪 В разобранных анализах найдено: " + ", ".join(_pats) +
                                        ". Это дополняет картину по жалобам — при оценке диагноза учитывайте оба источника.")
                            elif _vals:
                                st.info("🧪 Ранее разобран бланк анализов (явных паттернов не найдено). Показатели: " +
                                        ", ".join(sorted(_vals))[:300] + ".")

                        # --- Вывод рекомендаций (фолбэк на шаги подтверждения из KB) ---
                        recommendations = data.get('recommendations') or data.get('confirm') or []
                        if recommendations:
                            st.markdown("**🩺 Рекомендации:**")
                            if isinstance(recommendations, list):
                                for rec in recommendations:
                                    st.write(f"• {rec}")
                            else:
                                st.write(recommendations)

                        # --- Вывод красных флагов ---
                        red_flags = data.get('red_flags')
                        if red_flags:
                            st.markdown("**🚩 Красные флаги (тревожные симптомы):**")
                            if isinstance(red_flags, list):
                                for flag in red_flags:
                                    st.write(f"• {flag}")
                            else:
                                st.write(red_flags)

                        # --- Возможные альтернативные диагнозы (топ-3 по сходству) ---
                        try:
                            sims_np = np.asarray(sims)
                            order = np.argsort(sims_np)[::-1]
                            alts = [(CLASS_NAMES[k], float(sims_np[k])) for k in order if CLASS_NAMES[k] != found_class_id][:3]
                            if alts:
                                st.markdown("**🔍 Возможные альтернативные диагнозы:**")
                                for alt_name, alt_score in alts:
                                    st.write(f"• {alt_name} — сходство {alt_score:.2f}")
                        except Exception:
                            pass

                        # --- Уровень срочности (триаж из KB) ---
                        urgency = data.get('urgency')
                        if is_urgent or urgency in ('emergency', 'call_ambulance'):
                            st.warning("⚠️ Данное состояние может требовать срочной медицинской помощи. Обратитесь к врачу немедленно или вызовите скорую.")
                        elif urgency == 'doctor_visit':
                            st.info("ℹ️ Рекомендуется плановая консультация врача для подтверждения диагноза.")

                        # --- Почему такой вывод (объяснимость, волна 10) ---
                        with st.expander("🧭 Почему такой вывод"):
                            st.markdown(f"- Запрос распознан как **{found_class_id}**, сходство **{best_score:.2f}**; порог показа диагноза — 0.65")
                            _pairs = [(c, float(sims[CLASS_NAMES.index(c)])) for c in top5]
                            st.markdown("- Топ-5 кандидатов: " + "; ".join(f"{c} ({s:.2f})" for c, s in _pairs))
                            if _override:
                                st.markdown(f"- Сработало правило красных флагов: гипотеза скорректирована на **{_override}**")
                            if _redflag_missing:
                                st.markdown(f"- Сработал краснофлаг-предупреждение: {_redflag_missing}")
                            if is_gibberish_query(user_input):
                                st.markdown("- Сработал гейт тарабарщины: уверенность обнулена")
                        
                        # --- Что говорят источники (RAG, волна 10) ---
                        _src_hits = _rag_hits(user_input, found_class_id)
                        if _src_hits:
                            st.markdown("**📚 Что говорят источники:**")
                            for _sn, _sc in _src_hits:
                                _ru = _rag_kb[_sn]["url"]
                                _rd = _rag_kb[_sn]["desc"][:180]
                                st.markdown(f"- [{_sn}]({_ru}) — {_rd}")
                            st.caption("Ссылки ведут на клинические рекомендации и справочники; окончательную оценку делает врач.")
                        
                        st.session_state["last_dx"] = {"text": user_input, "dx": found_class_id, "score": round(float(best_score), 2), "top5": list(top5)}
                        
                        st.markdown("---")
                        st.caption(f"Уверенность модели: {best_score:.2f}")

                        # --- Обратная связь пользователя ---
                        _fb_key = f"fb1_{abs(hash(found_class_id)) % 100000}"
                        _fc1, _fc2 = st.columns(2)
                        if _fc1.button("👍 Диагноз полезен", key=_fb_key + "_up"):
                            _fb_record("жалобы", found_class_id, "👍")
                            st.toast("Спасибо! Отзыв записан в аналитику.")
                        if _fc2.button("👎 Диагноз не подошёл", key=_fb_key + "_down"):
                            _fb_record("жалобы", found_class_id, "👎")
                            st.toast("Записал — посмотрим в аналитику.")
                    else:
                        st.info("Система не смогла определить диагноз с достаточной уверенностью. Попробуйте переформулировать жалобы или обратитесь к врачу.")
                        log_unknown_query(user_input)

                except Exception as e:
                    st.error(f"Произошла ошибка во время анализа: {e!s}")

# --- ОБРАТНАЯ СВЯЗЬ: запись отзывов ---
FEEDBACK_PATH = "feedback.csv"
def _fb_record(source, item, verdict):
    import csv as _csv, datetime as _dt, os as _os
    exists = _os.path.exists(FEEDBACK_PATH)
    with open(FEEDBACK_PATH, "a", encoding="utf-8", newline="") as fh:
        w = _csv.writer(fh, delimiter=";")
        if not exists:
            w.writerow(["время", "источник", "объект", "оценка"])
        w.writerow([_dt.datetime.now().strftime("%Y-%m-%d %H:%M"), source, item, verdict])

# --- ЛАБОРАТОРНЫЙ МОДУЛЬ: база, знаковый движок, парсер ---
@st.cache_resource
def _load_lab_e5():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("intfloat/multilingual-e5-small")

import json as _lj, re as _lre, numpy as _lnp
try:
    _lab_kb = _lj.load(open("lab_knowledge.json", encoding="utf-8"))
    _lab_emb = _lnp.load("lab_embeddings.npz", allow_pickle=True)
    _PARAMS, _PATTERNS = _lab_kb["parameters"], _lab_kb["patterns"]
    _LAB_X = _lab_emb["embeddings"].astype(_lnp.float32)
    _LAB_NAMES = [str(x) for x in _lab_emb["names"]]
    _LAB_XN = _LAB_X / _lnp.linalg.norm(_LAB_X, axis=1, keepdims=True)
    _LAB_READY = True
except Exception as _le:
    _LAB_READY, _LAB_ERR = False, str(_le)

def _norm_range(p, sex):
    n = p.get("norms", {}).get(sex)
    return (n[0], n[1]) if n else (p["low"], p["high"])

def _egfr(creat_mmol, sex, age):
    scr = creat_mmol / 88.4
    k, alpha, mult = (0.7, -0.241, 1.012) if sex == "ж" else (0.9, -0.302, 1.0)
    return 142 * (min(scr / k, 1.0) ** alpha) * (max(scr / k, 1.0) ** (-1.200)) * (0.9938 ** age) * mult

def _lab_facts(values, sex="ж", age=None):
    if values.get("креатинин") and age:
        values["СКФ"] = round(_egfr(values["креатинин"], sex, age))
    f = {}
    for name, p in _PARAMS.items():
        if values.get(name) is None: continue
        v = values[name]
        lo, hi = _norm_range(p, sex)
        f[name] = "low" if v < lo else ("high" if v > hi else "normal")
    return f

def _lab_match_signs(f, values):
    hits = []
    for name, pat in _PATTERNS.items():
        signs = pat.get("signs")
        if not signs: continue
        ok = True
        for param, want in signs:
            if param in ("глюкоза", "гликированный гемоглобин"):
                v = values.get(param)
                if v is None: ok = False
                elif want == "high_moderate":
                    lo, hi = ((5.6, 6.9) if param == "глюкоза" else (5.7, 6.4))
                    if not (lo <= v <= hi): ok = False
                elif want == "high_severe":
                    if not (v >= (7.0 if param == "глюкоза" else 6.5)): ok = False
                elif want == "low":
                    if not (v < 3.3): ok = False
            elif param not in values: ok = False
            elif param == "ферритин" and want == "normal_or_high":
                if f.get(param) not in ("normal", "high"): ok = False
            elif f.get(param) != want: ok = False
            if not ok: break
        if ok: hits.append(name)
    return hits

_FOLLOW_EXCLUDE = {
    "гемоглобин": ["гликированный", "a1c"],
    "билирубин": ["прямой", "непрямой", "связанный", "несвязанный"],
    "холестерин": ["лпнп", "лпвп"],
    "белок": ["реактивный"],
}

_URINE_KEYWORDS = ["общий анализ мочи", "анализ мочи", "оам", "моча:"]
_PCT_CELLS = ("нейтрофилы", "лимфоциты", "эозинофилы")

def _parse_region(t, names):
    found = {}
    _pct = {}
    for name in names:
        p = _PARAMS[name]
        for alias in p["aliases"]:
            a = alias.lower().replace("ё", "е")
            for m in _lre.finditer(_lre.escape(a) + r"[^\d\-]{0,25}(\d+(?:\.\d+)?)", t):
                ns_, ne_ = m.start(1), m.end(1)
                if ns_ > 0 and t[ns_ - 1] == "(":
                    continue
                gap = t[m.start() + len(a): ns_]
                prev_words = t[:m.start()].split()
                prev = prev_words[-1].strip(",;:()") if prev_words else ""
                if any(w in gap or w in prev for w in _FOLLOW_EXCLUDE.get(a, [])):
                    continue
                try: v = float(m.group(1))
                except ValueError: continue
                if p["low"] > 0:
                    if not (0.01 * p["low"] < v < 100 * (p.get("high") or 1)):
                        continue
                else:
                    if not (0 <= v <= 100 * (p.get("high") or 1)):
                        continue
                if t[ne_:ne_ + 3].lstrip().startswith("%") and name in _PCT_CELLS:
                    _pct[name] = v
                    break
                found[name] = v
                break
            if name in found or name in _pct:
                break
    for cell, pc in _pct.items():
        if found.get("лейкоциты"):
            found[cell] = round(pc / 100.0 * found["лейкоциты"], 1)
    return found

def _parse_lab_text(text):
    t = text.lower().replace("ё", "е").replace(",", ".")
    urine_names = [n for n, p in _PARAMS.items() if p.get("urine")]
    blood_names = [n for n, p in _PARAMS.items() if not p.get("urine")]
    idx = None
    for kw in _URINE_KEYWORDS:
        j = t.find(kw)
        if j >= 0:
            idx = j
            break
    if idx is None:
        return _parse_region(t, blood_names)
    res = _parse_region(t[:idx], blood_names)
    u = _parse_region(t[idx:], urine_names)
    if "нитриты" not in u:
        k = t[idx:].find("нитрит")
        if k >= 0:
            tail = t[idx + k: idx + k + 60]
            if any(w in tail for w in ("положител", "positive", "пол.")):
                u["нитриты"] = 1.0
    res.update(u)
    return res

with tab2:
    st.header("🧪 Анализы")
    st.markdown("### Вставьте текст анализа или загрузите файл — расшифровка, паттерны и красные флаги")

    if not _LAB_READY:
        st.error(f"Лабораторная база недоступна: {_LAB_ERR}")
    else:
        sex = st.radio("Пол для нормативов", ["ж", "м"], horizontal=True,
                       format_func=lambda s: "женский" if s == "ж" else "мужской")
        age = st.number_input("Возраст (нужен для расчёта СКФ)", min_value=14, max_value=120, value=40)
        user_text = st.text_area("Текст анализа (скопируйте из бланка):", height=140)
        uploaded_file = st.file_uploader("Или загрузите файл (PDF, JPG, PNG, TXT)",
                                         type=["pdf", "jpg", "jpeg", "png", "txt"])
        file_text = ""
        if uploaded_file is not None:
            st.success(f"Файл {uploaded_file.name} успешно загружен.")
            file_bytes = uploaded_file.getvalue()
            try:
                if uploaded_file.type == "application/pdf":
                    try:
                        from pypdf import PdfReader
                    except ImportError:
                        from PyPDF2 import PdfReader
                    import io as _io
                    file_text = "".join((page.extract_text() or "") for page in PdfReader(_io.BytesIO(file_bytes)).pages)
                elif uploaded_file.type.startswith("image/"):
                    try:
                        import pytesseract
                        from PIL import Image
                        import io as _io
                        file_text = pytesseract.image_to_string(Image.open(_io.BytesIO(file_bytes)))
                    except ImportError:
                        st.warning("Для изображений нужен OCR: pip install pytesseract pillow. Или вставьте текст вручную.")
                else:
                    file_text = file_bytes.decode("utf-8", errors="ignore")
                if file_text:
                    st.text_area("Распознанный текст файла:", value=file_text, height=150)
            except Exception as e:
                st.error(f"Не удалось обработать файл: {e!s}")

        if st.button("🔍 Интерпретировать"):
            full = ((user_text or "") + "\n" + (file_text or "")).strip()
            if not full:
                st.warning("Вставьте текст или загрузите файл.")
            else:
                with st.spinner("Разбор показателей..."):
                    values = _parse_lab_text(full)
                if not values:
                    st.info("Показателей не распознано. Формат: «гемоглобин 90», «ТТГ 8.5», «сахар крови 6.1».")
                else:
                    st.markdown("### 📋 Расшифровка показателей")
                    f = _lab_facts(values, sex, age)
                    vd = {"low": "ниже нормы", "high": "выше нормы", "normal": "в норме"}
                    for name, v in values.items():
                        p = _PARAMS[name]
                        mark = {"low": "🔻", "high": "🔺", "normal": "✅"}[f[name]]
                        lo, hi = _norm_range(p, sex)
                        st.markdown(f"- {mark} **{name}: {v} {p['unit']}** — {vd[f[name]]} (норма {lo}–{hi})")

                    crit = []
                    for name, v in values.items():
                        p = _PARAMS[name]
                        cl, ch = p.get("crit_low"), p.get("crit_high")
                        if cl is not None and v < cl:
                            crit.append(f"**{name} = {v}** — критически низкий (порог {cl})")
                        if ch is not None and v > ch:
                            crit.append(f"**{name} = {v}** — критически высокий (порог {ch})")
                    if crit:
                        st.error("🚨 **Критические значения — нужен врач сейчас:** " + "; ".join(crit))

                    hits = _lab_match_signs(f, values)
                    if not hits:
                        q = _load_lab_e5().encode([f"query: {full[:400]}"], normalize_embeddings=True)[0].astype(_lnp.float32)
                        sims = _LAB_XN @ q
                        best = int(_lnp.argmax(sims))
                        if sims[best] >= 0.85:
                            hits = [_LAB_NAMES[best]]

                    st.session_state["lab_context"] = {
                        "patterns": list(hits),
                        "values": dict(values),
                    }

                    if hits:
                        st.markdown("### 🔎 Возможные паттерны")
                        for h in hits:
                            pat = _PATTERNS[h]
                            st.subheader(h)
                            st.write(pat["desc"])
                            st.warning(pat["danger"])
                            st.markdown("**🩺 Рекомендации:**")
                            for r in pat["recommendations"]:
                                st.markdown(f"- {r}")
                    else:
                        st.info("Явных паттернов не найдено — серьёзных сочетаний отклонений в распознанных показателях нет.")
                    # --- Обратная связь пользователя ---
                    _fb_item = ", ".join(hits) if hits else "без паттернов"
                    _fb_key2 = f"fb2_{abs(hash(_fb_item)) % 100000}"
                    _fc3, _fc4 = st.columns(2)
                    if _fc3.button("👍 Расшифровка полезна", key=_fb_key2 + "_up"):
                        _fb_record("анализы", _fb_item, "👍")
                        st.toast("Спасибо! Отзыв записан в аналитику.")
                    if _fc4.button("👎 Расшифровка не помогла", key=_fb_key2 + "_down"):
                        _fb_record("анализы", _fb_item, "👎")
                        st.toast("Записал — посмотрим в аналитику.")
                    # --- Экспорт отчёта для врача ---
                    try:
                        _marks = {"normal": "✅ в норме", "low": "▼ ниже нормы", "high": "▲ выше нормы"}
                        _rows = ""
                        for _n in values:
                            _p = _PARAMS.get(_n, {})
                            _st_ = f.get(_n)
                            _rows += ("<tr><td>" + _n + "</td><td>" + str(values[_n]) + " " + _p.get("unit", "") + "</td><td>" + _marks.get(_st_, "—") + "</td></tr>")
                        _pats = "".join("<li>" + h + "</li>" for h in hits) if hits else "<li>не выявлено</li>"
                        _sex_h = "женский" if sex == "ж" else "мужской"
                        _age_h = (" | Возраст: " + str(int(age))) if age else ""
                        _rep = ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Отчёт по анализам</title>"
                                "<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse;width:100%}"
                                "td,th{border:1px solid #ccc;padding:6px;text-align:left}h1{font-size:20px}"
                                ".disclaimer{color:#666;font-size:12px;margin-top:16px}</style></head><body>"
                                "<h1>Расшифровка лабораторных показателей</h1>"
                                "<p>Дата: " + __import__("datetime").datetime.now().strftime("%d.%m.%Y %H:%M") +
                                " | Пол: " + _sex_h + _age_h + "</p>"
                                "<table><tr><th>Показатель</th><th>Значение</th><th>Оценка</th></tr>" + _rows + "</table>"
                                "<h3>Выявленные паттерны</h3><ul>" + _pats + "</ul>"
                                "<p class='disclaimer'>Автоматическая расшифровка, не является диагнозом. "
                                "Окончательную интерпретацию выполняет врач с учётом клинической картины.</p>"
                                "</body></html>")
                        st.download_button("📄 Скачать отчёт для врача", data=_rep.encode("utf-8"),
                                           file_name="otchet_analizy.html", mime="text/html",
                                           key="dl_report_lab")
                    except Exception:
                        st.caption("Отчёт для врача сейчас недоступен")

                    st.caption("Расшифровка носит информационный характер и не заменяет консультацию врача: анализы интерпретируются вместе с жалобами и осмотром.")
# ==== Слой расшифровки снимков и ЭКГ (волна 9) ====
try:
    _xray_kb = _lj.load(open("xray_knowledge.json", encoding="utf-8"))
    _ecg_kb = _lj.load(open("ecg_knowledge.json", encoding="utf-8"))
    _xr_e = _lnp.load("xray_embeddings.npz", allow_pickle=True)
    _ecg_e = _lnp.load("ecg_embeddings.npz", allow_pickle=True)
    _xr_X = _xr_e["embeddings"].astype(_lnp.float32)
    _xr_X /= (_lnp.linalg.norm(_xr_X, axis=1, keepdims=True) + 1e-9)
    _xr_names = [str(x) for x in _xr_e["names"]]
    _ecg_X = _ecg_e["embeddings"].astype(_lnp.float32)
    _ecg_X /= (_lnp.linalg.norm(_ecg_X, axis=1, keepdims=True) + 1e-9)
    _ecg_names = [str(x) for x in _ecg_e["names"]]
    _scan_ready = True
except Exception:
    _scan_ready = False

_SCAN_ICON = {"critical": "🚨", "urgent": "⚠️", "watch": "ℹ️", "normal": "✅"}
_SCAN_ORDER = {"critical": 0, "urgent": 1, "watch": 2, "normal": 3}

def _scan_match(t, kb, X, names):
    t2 = t.lower().replace("ё", "е")
    det = []
    for term, entry in kb.items():
        if any(n in t2 for n in entry.get("negative", [])):
            continue
        for a in entry.get("aliases", []):
            if a.lower() in t2:
                det.append(term)
                break
    sem = []
    if not det and X is not None:
        try:
            q = _load_lab_e5().encode(["query: " + t[:400]], normalize_embeddings=True)[0].astype(_lnp.float32)
            sims = X @ q
            order = _lnp.argsort(sims)[::-1][:3]
            sem = [names[k] for k in order if float(sims[k]) >= 0.80]
        except Exception:
            sem = []
    det.sort(key=lambda x: _SCAN_ORDER.get(kb[x].get("urgency"), 4))
    return det, sem

def _scan_hr(t):
    m = _lre.search(r"ч\s*с\s*с[^\d]{0,10}(\d{2,3})", t.lower())
    if not m:
        m = _lre.search(r"пульс[^\d]{0,10}(\d{2,3})", t.lower())
    return int(m.group(1)) if m else None

with tab3:
    st.markdown("### 🦴 Расшифровка заключения рентгена")
    st.caption("Вставьте текст заключения рентгенолога — поясню каждое понятие простыми словами. Сам снимок модель не читает: интерпретирует только врач.")
    xtxt = st.text_area("Заключение рентгена", height=140, key="xray_in")
    if st.button("Расшифровать", key="xray_btn"):
        if not xtxt.strip():
            st.info("Вставьте текст заключения.")
        elif not _scan_ready:
            st.error("Файлы базы снимков не найдены: xray_knowledge.json / xray_embeddings.npz")
        else:
            det, sem = _scan_match(xtxt, _xray_kb, _xr_X, _xr_names)
            crit = [d for d in det if _xray_kb[d].get("urgency") == "critical"]
            urg = [d for d in det if _xray_kb[d].get("urgency") == "urgent"]
            if crit:
                st.error("🚨 **Требуется медицинская помощь сейчас:** " + "; ".join(crit))
            if urg:
                st.warning("⚠️ **Обратитесь к врачу в ближайшие дни, не откладывая:** " + "; ".join(urg))
            for term in det:
                if term in crit or term in urg:
                    continue
                u = _xray_kb[term].get("urgency")
                st.markdown(f"{_SCAN_ICON.get(u, 'ℹ️')} **{term}** — {_xray_kb[term]['desc']}")
            if sem:
                st.caption("Возможно, речь о: " + ", ".join(sem))
            if not det and not sem:
                st.info("Знакомых формулировок не нашёл. Пришлите заключение как есть — постараюсь распознать термины.")

with tab4:
    st.markdown("### 📈 Расшифровка ЭКГ")
    st.caption("Вставьте заключение ЭКГ — поясню термины простыми словами. ЭКГ фиксирует один момент записи; оценивает её лечащий врач.")
    etxt = st.text_area("Заключение ЭКГ", height=140, key="ecg_in")
    if st.button("Расшифровать", key="ecg_btn"):
        if not etxt.strip():
            st.info("Вставьте текст заключения.")
        elif not _scan_ready:
            st.error("Файлы базы не найдены: ecg_knowledge.json / ecg_embeddings.npz")
        else:
            hr = _scan_hr(etxt)
            if hr is not None:
                if hr < 50:
                    st.warning(f"⚠️ **ЧСС {hr}** — редкий пульс. Часто норма у тренированных; при слабости и головокружениях — к кардиологу.")
                elif hr > 90:
                    st.warning(f"⚠️ **ЧСС {hr}** — учащённый пульс. Причины: нагрузка, температура, анемия, щитовидная железа. Постоянная тахикардия — повод для обследования.")
                else:
                    st.success(f"✅ **ЧСС {hr}** — в обычном диапазоне (50–90).")
            det, sem = _scan_match(etxt, _ecg_kb, _ecg_X, _ecg_names)
            crit = [d for d in det if _ecg_kb[d].get("urgency") == "critical"]
            urg = [d for d in det if _ecg_kb[d].get("urgency") == "urgent"]
            if crit:
                st.error("🚨 **Требуется медицинская помощь сейчас:** " + "; ".join(crit))
            if urg:
                st.warning("⚠️ **Обратитесь к врачу в ближайшие дни:** " + "; ".join(urg))
            for term in det:
                if term in crit or term in urg:
                    continue
                u = _ecg_kb[term].get("urgency")
                st.markdown(f"{_SCAN_ICON.get(u, 'ℹ️')} **{term}** — {_ecg_kb[term]['desc']}")
            if sem:
                st.caption("Возможно, речь о: " + ", ".join(sem))
            if not det and not sem:
                st.info("Знакомых формулировок не нашёл. Пришлите заключение как есть — постараюсь распознать термины.")

with tab5:
    st.header("📊 Аналитика")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Неизвестные запросы")
        try:
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, "r", encoding="utf-8") as f:
                    logs = f.readlines()
                if logs:
                    st.text_area("Запросы, которые система не смогла распознать:",
                                 value="".join(logs), height=300)
                    _dl_all = list(dict.fromkeys([l.rstrip() for l in logs] + st.session_state.get("unknown_log", [])))
                    if _dl_all:
                        st.download_button("⬇️ Скачать лог запросов (файл + текущая сессия)", data=chr(10).join(_dl_all) + chr(10), file_name="unknown_queries_backup.txt", mime="text/plain")
                    if st.button("Очистить журнал"):
                        open(LOG_PATH, "w", encoding="utf-8").close()
                        st.rerun()
                else:
                    st.success("Журнал неизвестных запросов пуст.")
                    # --- Отчёт для врача (волна 10) ---
                    with st.expander("📄 Отчёт для врача"):
                        _parts = []
                        _dx = st.session_state.get("last_dx")
                        if _dx:
                            _parts.append("## Жалобы")
                            _parts.append("Текст: " + str(_dx.get("text", "")))
                            _parts.append("Гипотеза: " + str(_dx.get("dx", "")) + " (сходство " + str(_dx.get("score", "")) + ")")
                            _parts.append("Топ-5 кандидатов: " + ", ".join(map(str, _dx.get("top5", []))))
                        _lc = st.session_state.get("lab_context")
                        if _lc:
                            _parts.append("## Анализы")
                            _lv = _lc.get("values", {})
                            if _lv:
                                _parts.append("Показатели: " + "; ".join(str(k) + " = " + str(v) for k, v in _lv.items()))
                            if _lc.get("patterns"):
                                _parts.append("Паттерны: " + ", ".join(map(str, _lc.get("patterns", []))))
                        _xt = st.session_state.get("xray_in") or ""
                        if _xt.strip():
                            _parts.append("## Заключение рентгена")
                            _parts.append(_xt)
                        _et = st.session_state.get("ecg_in") or ""
                        if _et.strip():
                            _parts.append("## Заключение ЭКГ")
                            _parts.append(_et)
                        if _parts:
                            _rep = "# Отчёт для врача (MedBot Duo)" + chr(10) + chr(10) + chr(10).join(_parts) + chr(10)
                            st.download_button("⬇️ Скачать отчёт (Markdown)", data=_rep, file_name="otchet_dlya_vracha.md", mime="text/markdown", key="doc_report_btn")
                        else:
                            st.caption("Отчёт появится после разбора жалоб или анализов в этой сессии.")
                    
                    if st.session_state.get("unknown_log"):
                        st.download_button("⬇️ Скачать лог текущей сессии", data=chr(10).join(st.session_state["unknown_log"]) + chr(10), file_name="unknown_queries_session.txt", mime="text/plain")
            else:
                st.info("Журнал неизвестных запросов еще не создан.")
        except Exception as e:
            st.error(f"Не удалось прочитать журнал: {e!s}")

    with col2:
        st.subheader("Статистика базы знаний")
        try:
            st.metric("Всего диагнозов", len(kb))
            st.metric("С пометкой 'Срочно'",
                      sum(1 for v in kb.values() if v.get("urgent") is True))
            st.metric("С рекомендациями",
                      sum(1 for v in kb.values() if v.get("recommendations")))
            st.metric("С красными флагами",
                      sum(1 for v in kb.values() if v.get("red_flags")))
            st.markdown("---")
            st.markdown("### Поиск по базе знаний")
            search_term = st.text_input("Введите ID диагноза или ключевое слово")
            if search_term:
                results = {k: v for k, v in kb.items()
                           if search_term.lower() in k.lower()
                           or search_term.lower() in str(v).lower()}
                if results:
                    st.json(results)
                else:
                    st.info("Ничего не найдено.")
        except Exception as e:
            st.error(f"Ошибка статистики: {e!s}")

    st.markdown("---")
    st.subheader("Отзывы пользователей")
    try:
        import pandas as _fbpd
        if os.path.exists(FEEDBACK_PATH):
            _fb = _fbpd.read_csv(FEEDBACK_PATH, sep=";")
            _up = int((_fb["оценка"] == "👍").sum())
            _down = int((_fb["оценка"] == "👎").sum())
            _mc1, _mc2, _mc3 = st.columns(3)
            _mc1.metric("👍 Полезно", _up)
            _mc2.metric("👎 Не то", _down)
            _mc3.metric("Всего отзывов", len(_fb))
            if _down:
                st.markdown("**👎 Последние негативные:**")
                st.dataframe(_fb[_fb["оценка"] == "👎"].tail(10), use_container_width=True)
        else:
            st.info("Отзывы пока не собирались.")
    except Exception as e:
        st.error(f"Ошибка отзывов: {e!s}")
