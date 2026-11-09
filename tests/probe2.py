import warnings; warnings.filterwarnings("ignore")
import json

import medbot

print("--- справочник глазами кода ---")
kb = medbot._advice_kb()
for k in ("псориаз", "гипертония", "бессонница", "гипертонический криз"):
    v = kb.get(k)
    if v is None:
        print(f"{k}: НЕТ ЗАПИСИ")
    else:
        print(f"{k}: urgency={v.get('urgency')} | urgent={v.get('urgent')} | draft={v.get('draft')}")

print("--- полные ответы predict ---")
KEYS = ("diagnosis", "confidence", "urgent", "urgency", "guard", "description")
for t in [
    "на локтях бляшки с шелушением, чешутся не сильно, давно так",
    "давление 185 и сильно болит затылок, мутит",
    "давление поднимается до 150 на 95, подташнивает, к вечеру хуже",
    "давление 150 на 95, подташнивает и болит голова",
    "не могу уснуть уже неделю, ворочаюсь, тревожно",
]:
    r = medbot.predict(t)
    out = {k: r.get(k) for k in KEYS}
    if out.get("description") is not None:
        out["description"] = (out["description"] or "")[:30] or "<пусто>"
    print(f"{t[:55]!r}\n   -> {json.dumps(out, ensure_ascii=False)}")
