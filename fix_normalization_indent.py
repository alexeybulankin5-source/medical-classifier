# -*- coding: utf-8 -*-
import re
src = open("medbot.py").read()
lines = src.split("\n")

MARK = "волна 7б: нормализация urgency после криз-гварда"
out, i, removed = [], 0, 0
while i < len(lines):
    if MARK in lines[i]:
        removed += 1
        i += 1
        if i < len(lines) and 'r["urgency"] = "emergency" if r["urgent"]' in lines[i]:
            i += 1
        continue
    out.append(lines[i]); i += 1
lines = out

idx = next((j for j, l in enumerate(lines)
            if 'r["guard"]' in l and "криз-гвард" in l), None)
if idx is None:
    print("СТРОКА ГВАРДА НЕ НАЙДЕНА"); raise SystemExit(1)

indent = re.match(r"\s*", lines[idx]).group(0)
lines[idx+1:idx+1] = [
    f"{indent}# --- {MARK} ---",
    f'{indent}r["urgency"] = "emergency" if r["urgent"] else r.get("urgency", "doctor_visit")',
]
open("medbot.py", "w").write("\n".join(lines))
print(f"снято старых вставок: {removed}")
print("контекст вокруг гварда:")
for l in lines[max(0, idx - 3):idx + 6]:
    print(repr(l))
