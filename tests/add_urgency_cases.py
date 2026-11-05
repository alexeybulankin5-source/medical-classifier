# -*- coding: utf-8 -*-
import json
cases = json.load(open("tests/test_suite.json"))
have = {json.dumps(c, ensure_ascii=False, sort_keys=True) for c in cases}
new = [
    {"text": "давление 185 и сильно болит затылок, мутит", "expect_cls": ["гипертонический криз"], "conf": "high", "urgent": True, "urgency": "emergency"},
    {"text": "давление 150, чуть подташнивает, слабость", "expect_cls": ["гипертония"], "conf": "high", "urgent": False, "urgency": "doctor_visit"},
    {"text": "на локтях бляшки с шелушением, чешутся не сильно, давно так", "expect_cls": ["псориаз"], "conf": "high", "urgent": False, "urgency": "self_care"},
    {"text": "сильно ломит всё тело, высокая температура под 39 и сыпь мелкая, не бледнеет", "expect_cls": ["менингит"], "conf": "high", "urgent": True, "urgency": "emergency"},
]
added = 0
for c in new:
    if json.dumps(c, ensure_ascii=False, sort_keys=True) not in have:
        cases.append(c); added += 1
json.dump(cases, open("tests/test_suite.json", "w"), ensure_ascii=False, indent=2)
print(f"добавлено кейсов: {added}, всего: {len(cases)}")
