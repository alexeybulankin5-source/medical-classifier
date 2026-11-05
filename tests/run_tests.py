import importlib.util, sys, json

spec = importlib.util.spec_from_file_location("mb", "medbot.py")
mb = importlib.util.module_from_spec(spec); sys.modules["mb"] = mb
spec.loader.exec_module(mb)

suite = json.load(open("tests/test_suite.json", encoding="utf-8"))
bank = json.load(open("tests/stress_bank.json", encoding="utf-8"))
_seen = {c["text"] for c in suite}
suite = suite + [c for c in bank if c["text"] not in _seen]
passed, failed = 0, []
for case in suite:
    r = mb.predict(case["text"])
    conf = r.get("confidence")
    top = r["top"][0][0] if r.get("top") else ""
    problems = []
    want_cls = case.get("expect_cls")
    if want_cls and top not in want_cls:
        problems.append(f"класс: ждали {want_cls}, получили {top}")
    want_conf = case.get("conf") or ([case.get("expect_conf")] if case.get("expect_conf") else None)
    if want_conf and conf not in want_conf:
        problems.append(f"уверенность: ждали {want_conf}, получили {conf}")
    if "urgent" in case and bool(r.get("urgent")) != case["urgent"]:
        problems.append(f"urgent: ждали {case['urgent']}, получили {bool(r.get('urgent'))}")
    if case.get("urgent") and not r.get("confirm"):
        problems.append("urgent без шагов подтверждения")
    if "urgency" in case:
        got_u = "emergency" if r.get("urgent") else r.get("urgency")
        if got_u != case["urgency"]:
            problems.append(f"urgency: ждали {case['urgency']}, получили {got_u}")
    if case.get("advice_free") and (r.get("description") or r.get("confirm")):
        problems.append("совет навязан при low-уверенности")
    if case.get("draft_if_medium") and conf == "medium" and not r.get("draft_advice"):
        problems.append("medium-совет не помечен как черновик")
    if problems:
        failed.append((case["text"], problems, top, conf))
    else:
        passed += 1
        print(f"PASS [{conf:6}] {case['text'][:52]:52} -> {top}")

print()
for text, problems, top, conf in failed:
    print(f"FAIL [{conf:6}] {text[:52]}")
    for p in problems:
        print(f"     - {p} (факт: {top})")

verdict = "PASS" if not failed else "FAIL"
print(f"\nИТОГ: {verdict} — {passed}/{len(suite)}")
open("tests/last_verdict.txt", "w").write(verdict)
