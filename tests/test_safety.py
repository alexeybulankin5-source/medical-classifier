"""Регрессионные тесты красных флагов. Запуск: python -m pytest tests -q"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety import assess

# (текст, ожидаемый id флага) — каждый позитив обязан срабатывать
POSITIVE = [
    ("острая боль в груди и тяжело дышать", "chest_pain_dyspnea"),
    ("перекосило лицо и речь стала невнятной", "stroke"),
    ("внезапно онемела левая рука", "stroke"),
    ("сильное кровотечение из раны, кровь не останавливается", "severe_bleeding"),
    ("человек потерял сознание", "unconsciousness"),
    ("температура 40 и появилась сыпь по телу", "fever_rash_stiff"),
    ("начались судороги", "seizures"),
    ("рвота не прекращается, моча тёмная и мало", "dehydration"),
]
NEGATIVE = [
    "лёгкая головная боль к вечеру",
    "боль в спине третий день, к вечеру проходит",
    "кашель неделю, температуры нет",
    "немного тошнит после еды",
    "усталость и плохой сон",
    "давит в груди, но одышки нет",
]

def test_positives():
    for text, flag in POSITIVE:
        ids = [f["id"] for f in assess(text)]
        assert flag in ids, f"флаг {flag} не сработал на: {text!r}"

def test_negatives():
    for text in NEGATIVE:
        ids = [f["id"] for f in assess(text)]
        assert ids == [], f"ложное срабатывание {ids} на: {text!r}"
