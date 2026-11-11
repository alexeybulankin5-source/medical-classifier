# -*- coding: utf-8 -*-
"""Регрессионный прогон MedBot Duo: классификация жалоб + правила, без запуска интерфейса.

Запуск:  python tests_regression.py
Зависимости: numpy, sentence-transformers, файлы app.py, knowledge.json, class_embeddings.npz
"""

import ast
import re
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

CASES = [
    ("опухли губы после орехов", ["квинке", "анафилакс"]),
    ("болит горло и белый налет на миндалинах температура 37", ["тонзилл", "ангин"]),
    ("болит живот и понос", ["диар", "кишечн", "гастроэнтерит"]),
    ("депрессия и плохой сон", ["депресс"]),
    ("пульсирующая боль в половине головы, светобоязнь", ["мигрень"]),
    ("давит за грудиной при подъёме на лестницу", ["стенокард", "ибс", "инфаркт"]),
    ("постоянная жажда и частое мочеиспускание", ["диабет"]),
    ("болит колено при ходьбе по лестнице", ["артроз", "гонартроз"]),
    ("жжение и частые позывы к мочеиспусканию", ["цистит"]),
    ("свистящее дыхание и приступы одышки ночью", ["астма", "хобл"]),
    ("температура 37 и першение в горле", ["орви", "фарингит", "тонзилл", "ангин"]),
    ("asdfgh zxcvbn qwerty", ["нераспознан"], []),
    ("фывапр олдж", ["нераспознан"], []),
]

RULE_FUNCS = ("_gib_word", "is_gibberish_query", "_redflag_check", "_bp_override",
              "_allergy_override", "_tiebreak_pick", "_tonsil_override", "_symptom_override")


def collect_namespaces():
    """Извлекает из app.py функции правил и все константы, на которые они ссылаются."""
    src = open("app.py", encoding="utf-8").read()
    tree = ast.parse(src)
    top_funcs, top_assigns = {}, {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            top_funcs[node.name] = ast.get_source_segment(src, node)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            top_assigns[node.targets[0].id] = ast.get_source_segment(src, node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            top_assigns[node.target.id] = ast.get_source_segment(src, node)

    needed = set(RULE_FUNCS)
    for _ in range(4):
        used = set()
        for name in list(needed):
            if name in top_funcs:
                for n in ast.walk(ast.parse(top_funcs[name])):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                        used.add(n.id)
        new = {u for u in used if (u in top_assigns or u in top_funcs) and u not in needed}
        if not new:
            break
        needed |= new

    ns = {"re": re, "_re": re, "np": np}
    for name in sorted(needed):
        code = top_funcs.get(name) or top_assigns.get(name)
        if code:
            exec(code, ns)
    return ns


def run_all(verbose=True):
    ns = collect_namespaces()
    enc = SentenceTransformer("intfloat/multilingual-e5-small")
    emb = np.load("class_embeddings.npz", allow_pickle=True)
    X = np.asarray(emb["embeddings"]).astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    class_names = [str(x) for x in emb["names"]]
    ns["CLASS_NAMES"] = class_names
    assert len(class_names) == len(open("knowledge.json", encoding="utf-8-sig").read() and
                                 __import__("json").load(open("knowledge.json", encoding="utf-8-sig"))), \
        "индекс и база рассинхронизированы"

    gib = ns.get("is_gibberish_query")
    results = []
    for case in CASES:
        text, expects = case[0], case[1]
        if gib and gib(text):
            results.append((True, text, "нераспознан", None, None, [], 0.0))
            if verbose:
                print("PASS", repr(text), "-> нераспознан  [gibberish]")
            continue
        q = enc.encode(["query: " + text], normalize_embeddings=True)[0].astype(np.float32)
        sims = X @ q
        order = np.argsort(sims)[::-1]
        top5 = [class_names[int(i)] for i in order[:5]]
        if "_tiebreak_pick" in ns:
            best = int(ns["_tiebreak_pick"](sims, text))
        else:
            best = int(order[0])
        ovr, rf = None, None
        if "_redflag_check" in ns:
            try:
                ovr, rf = ns["_redflag_check"](text, top5)
            except Exception:
                pass
        if not ovr:
            for name in ("_bp_override", "_allergy_override", "_tonsil_override", "_symptom_override"):
                if name in ns:
                    try:
                        ovr = ns[name](text, top5)
                    except Exception:
                        ovr = None
                    if ovr:
                        break
        dx = ovr or class_names[best]
        if not expects:
            ok = bool(gib and gib(text))
        else:
            ok = any(w in dx.lower() for w in expects)
        results.append((ok, text, dx, ovr, rf, top5, float(sims[order[0]])))
        if verbose:
            tag = "PASS" if ok else "FAIL"
            note = f"  [override -> {ovr}]" if ovr and ovr != top5[0] else ""
            rfnote = f"  [redflag: {rf}]" if rf else ""
            print(tag, repr(text), "->", dx, note, rfnote)

    n_pass = sum(1 for r in results if r[0])
    n_fail = len(results) - n_pass
    print(f"ИТОГ: {n_pass} PASS / {n_fail} FAIL")
    return results


if __name__ == "__main__":
    res = run_all()
    sys.exit(0 if all(r[0] for r in res) else 1)
