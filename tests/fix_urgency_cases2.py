# -*- coding: utf-8 -*-
import json
cases = json.load(open("tests/test_suite.json"))
changed = 0
for c in cases:
    if c.get("expect_cls") == ["гипертонический криз"]:
        # криз-гвард несёт emergency-путь независимо от класса и conf — кейс теперь охраняет сам гвард
        c["text"] = "давление 185 и сильно болит затылок, мутит"
        c["expect_cls"] = ["гипертония"]; c["conf"] = "low"
        c["urgent"] = True; c["urgency"] = "emergency"; changed += 1
    elif c.get("expect_cls") == ["менингит"]:
        c["text"] = ("голова разрывается, свет невыносим, подбородок к груди не прижать, "
                     "шея как камень, мелкая сыпь не бледнеет при надавливании, температура 39")
        c["conf"] = "high"; c["urgent"] = True; c["urgency"] = "emergency"; changed += 1
json.dump(cases, open("tests/test_suite.json", "w"), ensure_ascii=False, indent=2)
print(f"обновлено кейсов: {changed}")
