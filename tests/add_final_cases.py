import json

NEW = [
    # псориаз: в миграции отнесён к doctor_visit (новый диагноз требует визита); conf=medium по зонду
    {"text": "на локтях и коленях годами красные бляшки с серебристым шелушением, чешутся умеренно, не проходят",
     "expect_cls": ["псориаз"], "conf": "medium", "urgent": False, "urgency": "doctor_visit"},
    # гипертония 150/95: зонд подтвердил гипертонию medium, doctor_visit (антоним-гвард)
    {"text": "давление 150 на 95, подташнивает и болит голова",
     "expect_cls": ["гипертония"], "conf": "medium", "urgent": False, "urgency": "doctor_visit"},
]
cases = json.load(open("tests/test_suite.json"))
have = {c["text"] for c in cases}
added = [c for c in NEW if c["text"] not in have]
cases += added
json.dump(cases, open("tests/test_suite.json", "w"), ensure_ascii=False, indent=2)
print(f"добавлено кейсов: {len(added)}, всего: {len(cases)}")
