import json

log = json.load(open("defects_log.json"))
fixed = 0
for d in log:
    if d["id"] == "D-008" and "верdict" in d:
        d["вердикт"] = d.pop("верdict"); fixed += 1
json.dump(log, open("defects_log.json", "w"), ensure_ascii=False, indent=2)
print(f"исправлено ключей: {fixed}")
