import json

head = json.load(open("/tmp/suite_head.json"))
NEW = [
    # криз-гвард: срабатывает при любой уверенности — кейс охраняет сам гвард (D-007)
    {"text": "давление 185 и сильно болит затылок, мутит",
     "expect_cls": ["гипертония"], "conf": "low", "urgent": True, "urgency": "emergency"},
    # менингит: усиленная формулировка, подтверждена зондом (high / urgent / emergency)
    {"text": "голова разрывается, свет невыносим, подбородок к груди не прижать, шея как камень, "
             "мелкая сыпь не бледнеет при надавливании, температура 39",
     "expect_cls": ["менингит"], "conf": "high", "urgent": True, "urgency": "emergency"},
]
suite = head + [c for c in NEW if c["text"] not in {h["text"] for h in head}]
json.dump(suite, open("tests/test_suite.json", "w"), ensure_ascii=False, indent=2)
print(f"оригинальных: {len(head)}, добавлено новых: {len(suite) - len(head)}, всего: {len(suite)}")
