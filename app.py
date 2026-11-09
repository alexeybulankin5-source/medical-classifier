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
def load_resources():
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

kb, encoder, CLASS_NAMES, CLASS_EMBEDDINGS = load_resources()

# --- ДИСАМБИГУАЦИЯ ДАВЛЕНИЯ (правило сильнее сходств) ---
import re as _re
def _bp_override(text, top_names):
    t = text.lower()
    m = _re.search(r"(\d{2,3})\s*(?:на|/|–|-)\s*(\d{2,3})", t)
    if m:
        hi, lo = int(m.group(1)), int(m.group(2))
        if hi >= 140 or lo >= 90:
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
    (["справа внизу живота", "правая нижняя часть живота", "правый низ живота",
      "боль смещается в правую", "у пупка и смещается"],
     ["тошнит", "тошнота", "рвота", "температур", "нет аппетита"],
     "аппендицит"),
]
def _redflag_check(text, top_names):
    t = text.lower()
    for req, extra, target in _REDFLAG_RULES:
        if any(w in t for w in req) and (not extra or any(w in t for w in extra)):
            if target in top_names:
                return target, None
            return None, target   # класс вне топ-5 — только предупреждение
    return None, None


# --- ЛОГИРОВАНИЕ НЕИЗВЕСТНЫХ ЗАПРОСОВ ---
def log_unknown_query(query: str):
    if not query.strip(): 
        return False
    
    try:
        user_emb = encoder.encode(["query: " + query])[0]
        sims = cosine_similarity([user_emb], CLASS_EMBEDDINGS)[0]
        
        if np.max(sims) < 0.65:
            try:
                os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
                with open(LOG_PATH, 'a', encoding='utf-8') as f:
                    f.write(f"{query}\n")
                return True
            except Exception as e:
                print(f"Ошибка записи: {e}")
                return False
    except Exception as e:
        st.error(f"Ошибка обработки: {e!s}")
        return False
    return False

# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---
st.title("🩺 MedBot Duo")
st.markdown("Интеллектуальная система первичной диагностики")

tab1, tab2, tab3 = st.tabs(["📝 Жалобы", "🧪 Анализы", "📊 Аналитика"])

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
                    
                    best_idx = np.argmax(sims)
                    best_score = sims[best_idx]
                    found_class_id = CLASS_NAMES[best_idx]

                    top5 = [CLASS_NAMES[i] for i in np.argsort(sims)[::-1][:5]]
                    _override, _redflag_missing = _redflag_check(user_input, top5)
                    if not _override:
                        _override = _bp_override(user_input, top5)
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

                        st.markdown("---")
                        st.caption(f"Уверенность модели: {best_score:.2f}")
                    else:
                        st.info("Система не смогла определить диагноз с достаточной уверенностью. Попробуйте переформулировать жалобы или обратитесь к врачу.")
                        log_unknown_query(user_input)

                except Exception as e:
                    st.error(f"Произошла ошибка во время анализа: {e!s}")

with tab2:
    st.header("🧪 Анализы")
    st.markdown("### Загрузка результатов лабораторных исследований")
    uploaded_file = st.file_uploader("Выберите файл (PDF, JPG, PNG, TXT)", type=["pdf", "jpg", "jpeg", "png", "txt"])
    
    if uploaded_file is not None:
        st.success(f"Файл {uploaded_file.name} успешно загружен.")
        
        # Детектируем тип файла для правильного чтения
        file_bytes = uploaded_file.getvalue()
        
        try:
            if uploaded_file.type == "application/pdf":
                # Для PDF потребуется библиотека PyPDF2 или pdfplumber (не импортирована в начале)
                st.info("Для анализа PDF требуется установка дополнительных библиотек. Текст:")
                try:
                    import io

                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    st.write(text)
                except ImportError:
                    st.warning("Установите библиотеку: pip install PyPDF2")
            elif uploaded_file.type.startswith("image/"):
                # Для изображений потребуется pytesseract
                st.info("Для анализа изображений требуется установка OCR-библиотек. Текст:")
                try:
                    import io

                    import pytesseract
                    from PIL import Image
                    image = Image.open(io.BytesIO(file_bytes))
                    text = pytesseract.image_to_string(image)
                    st.write(text)
                except ImportError:
                    st.warning("Установите библиотеки: pip install pytesseract pillow")
            else:
                # Текстовые файлы
                text = file_bytes.decode('utf-8')
                st.text_area("Содержимое файла:", value=text, height=200)
                
        except Exception as e:
            st.error(f"Не удалось обработать файл: {e!s}")
    else:
        st.info("Ожидание загрузки файла...")

with tab3:
    st.header("📊 Аналитика")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Неизвестные запросы")
        try:
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, 'r', encoding='utf-8') as f:
                    logs = f.readlines()
                if logs:
                    st.text_area("Запросы, которые система не смогла распознать:", value="".join(logs), height=300)
                    if st.button("Очистить журнал"):
                        open(LOG_PATH, 'w', encoding='utf-8').close()
                        st.rerun()
                else:
                    st.success("Журнал неизвестных запросов пуст.")
            else:
                st.info("Журнал неизвестных запросов еще не создан.")
        except Exception as e:
            st.error(f"Не удалось прочитать журнал: {e!s}")

    with col2:
        st.subheader("Статистика базы знаний")
        try:
            total_classes = len(kb)
            urgent_count = sum(1 for v in kb.values() if v.get('urgent') is True)
            with_recs = sum(1 for v in kb.values() if v.get('recommendations'))
            with_flags = sum(1 for v in kb.values() if v.get('red_flags'))
            
            st.metric("Всего диагнозов", total_classes)
            st.metric("С пометкой 'Срочно'", urgent_count)
            st.metric("С рекомендациями", with_recs)
            st.metric("С красными флагами", with_flags)
            
            st.markdown("---")
            st.markdown("### Поиск по базе знаний")
            search_term = st.text_input("Введите ID диагноза или ключевое слово")
            if search_term:
                results = {k: v for k, v in kb.items() if search_term.lower() in k.lower() or search_term.lower() in str(v).lower()}
                if results:
                    st.json(results)
                else:
                    st.info("Ничего не найдено.")
        except Exception as e:
            st.error(f"Ошибка статистики: {e!s}")
