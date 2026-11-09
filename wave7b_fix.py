# WAVE 7b-fix: гвард форсирует urgent ПОСЛЕ присвоения urgency — нормализацию переносим за гвард.
MARK = "нормализация urgency после криз-гварда"
src = open("medbot.py").read()
if MARK in src:
    print("фикс уже применён — пропуск")
else:
    anchor = 'r["guard"] = (r.get("guard") + "; " if r.get("guard") else "") + "криз-гвард: давление >= 180/120 — urgent принудительно"'
    if anchor not in src:
        print("ЯКОРЬ НЕ НАЙДЕН"); raise SystemExit(1)
    add = (anchor + '\n'
           '    # --- волна 7б: нормализация urgency после криз-гварда (гвард форсирует urgent) ---\n'
           '    r["urgency"] = "emergency" if r["urgent"] else r.get("urgency", "doctor_visit")')
    src = src.replace(anchor, add, 1)
    open("medbot.py", "w").write(src)
    print("medbot.py: нормализация перенесена за криз-гвард")
