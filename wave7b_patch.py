# -*- coding: utf-8 -*-
# WAVE 7b: трёхуровневый триаж urgent -> urgency (D-008). Стратегия: добавление, не замена.
import json, shutil, sys

EMERGENCY = {
    "анафилаксия", "аппендицит", "гипертонический криз", "инсульт",
    "инфаркт миокарда", "кровотечение ЖКТ", "менингит", "тромбоэмболия лёгочной артерии",
    # кандидаты D-008 (спорные можно исключить отсюда и перезапустить)
    "сепсис", "расслоение аорты", "пневмоторакс", "кишечная непроходимость", "эпилепсия",
}
SELF_CARE = {
    "потница", "опрелости", "натоптыши и мозоли", "серная пробка", "ячмень",
    "заеды", "бородавки", "витилиго", "алопеция", "себорейный дерматит",
    "грибок ногтей", "грибок стоп", "микоз стоп", "вегетососудистая дистония",
    "переутомление", "неврастения", "бессонница", "головная боль напряжения",
    "предменструальный синдром", "младенческие колики", "прорезывание зубов",
    "диатез у грудничков", "укусы насекомых", "солнечный ожог", "метеоризм",
    "дисбактериоз", "акне", "близорукость",
}

# --- 1. Миграция knowledge.json ---
shutil.copy("knowledge.json", "knowledge.json.bak_7b")
kb = json.load(open("knowledge.json"))
for k, v in kb.items():
    v["urgency"] = ("emergency" if k in EMERGENCY
                    else "self_care" if k in SELF_CARE else "doctor_visit")
json.dump(kb, open("knowledge.json", "w"), ensure_ascii=False, indent=2)
n_em = sum(1 for v in kb.values() if v["urgency"] == "emergency")
n_sc = sum(1 for v in kb.values() if v["urgency"] == "self_care")
n_dv = sum(1 for v in kb.values() if v["urgency"] == "doctor_visit")
print(f"knowledge.json: emergency {n_em} | self_care {n_sc} | doctor_visit {n_dv} (бэкап: knowledge.json.bak_7b)")

# --- 2. Патч medbot.py: только точные замены якорей ---
src = open("medbot.py").read()
CHANGES = [
    # дефолт для неизвестных классов
    ('{"desc": "", "confirm": [], "urgent": False, "draft": True}',
     '{"desc": "", "confirm": [], "urgent": False, "urgency": "doctor_visit", "draft": True}'),
    # присвоение urgent -> рядом urgency; emergency-классы форсируют urgent
    ('    r["urgent"] = bool(info.get("urgent", False)) and r.get("confidence") in ("high", "medium")',
     '    r["urgent"] = bool(info.get("urgent", False)) and r.get("confidence") in ("high", "medium")\n'
     '    # --- волна 7б (D-008): трёхуровневый триаж; emergency-классы принудительно urgent ---\n'
     '    if info.get("urgency") == "emergency" and r.get("confidence") in ("high", "medium"):\n'
     '        r["urgent"] = True\n'
     '    r["urgency"] = "emergency" if r["urgent"] else info.get("urgency", "doctor_visit")'),
]
for old, new in CHANGES:
    if old not in src:
        print("ЯКОРЬ НЕ НАЙДЕН, патч прерван:", old[:70]); sys.exit(1)
    src = src.replace(old, new, 1)
open("medbot.py", "w").write(src)
print("medbot.py: 2 точные замены")

# --- 3. Патч run_tests.py: проверка urgency (только когда кейс её задаёт) ---
ts = open("tests/run_tests.py").read()
anchor = '        problems.append("urgent без шагов подтверждения")'
add = (anchor + '\n'
       '    if "urgency" in case:\n'
       '        got_u = "emergency" if r.get("urgent") else r.get("urgency")\n'
       '        if got_u != case["urgency"]:\n'
       "            problems.append(f\"urgency: ждали {case['urgency']}, получили {got_u}\")")
if anchor not in ts:
    print("ЯКОРЬ run_tests.py НЕ НАЙДЕН, патч прерван"); sys.exit(1)
ts = ts.replace(anchor, add, 1)
open("tests/run_tests.py", "w").write(ts)
print("run_tests.py: проверка urgency добавлена (активна только для кейсов с полем urgency)")

# --- 4. Отчёт: существующие кейсы, затронутые новыми emergency-классами ---
cases = json.load(open("tests/test_suite.json"))
text_key = max(cases[0], key=lambda c: len(str(cases[0][c])))
label_key = next((k for k, v in cases[0].items()
                  if k != text_key and isinstance(v, str) and v in kb), None)
print(f"схема кейса: текст={text_key!r}, класс={label_key!r}, ключи: {list(cases[0])}")
touched = [c for c in cases
           if label_key and c.get(label_key) in EMERGENCY and c.get("urgent") is False]
print(f"кейсов с urgent=false на emergency-классах: {len(touched)}")
for c in touched[:10]:
    print("  !", json.dumps(c, ensure_ascii=False)[:120])
print("схема первого кейса целиком:", json.dumps(cases[0], ensure_ascii=False)[:200])
