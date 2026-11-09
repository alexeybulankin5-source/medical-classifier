import json
import sys

# --- журнал дефектов ---
log = json.load(open("defects_log.json"))
if any(d.get("id") == "D-008" for d in log):
    print("D-008 уже в журнале — пропуск")
else:
    log.append({
        "id": "D-008",
        "класс": "общая логика триажа",
        "текст": "urgent бинарный: нет различия «скорая / к врачу / самолечение» — self_care-случаи визуально неотличимы от doctor_visit",
        "получено": "-",
        "верdict": "внедрён трёхуровневый триаж urgency (emergency / self_care / doctor_visit)",
        "шаг": "исправлено: миграция knowledge.json (emergency 13 / self_care 28 / doctor_visit 172), нормализация urgency перенесена за криз-гвард (urgent => emergency, D-007-наследие), трёхуровневый вывод в advice_ui.py, +2 кейса в щит (криз-гвард при low, менингит), щит 394/394"
    })
    json.dump(log, open("defects_log.json", "w"), ensure_ascii=False, indent=2)
    print("D-008 добавлена, всего записей:", len(log))

# --- README: обновляем строку про диапазон ---
readme = open("README.md").read()
if "D-001...D-008" in readme:
    print("README уже обновлён — пропуск")
elif "D-001...D-007" in readme:
    readme = readme.replace("D-001...D-007", "D-001...D-008", 1)
    if "(urgent-триггер для криза — закрыт криз-гвардом)." in readme:
        readme = readme.replace(
            "(urgent-триггер для криза — закрыт криз-гвардом).",
            "(urgent-триггер для криза — закрыт криз-гвардом) и D-008 "
            "(трёхуровневый триаж urgency: emergency 13 / self_care 28 / doctor_visit 172).", 1)
    open("README.md", "w").write(readme)
    print("README: D-008 добавлена")
else:
    print("АНКОР README НЕ НАЙДЕН — строка про диапазон не обновлена")
    sys.exit(1)
