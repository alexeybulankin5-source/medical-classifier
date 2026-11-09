import warnings

warnings.filterwarnings("ignore")
import medbot

TEXTS = [
    # 1) криз: текущая формулировка и усиленная
    "давление 185 и сильно болит затылок, мутит",
    "давление подскочило до 190, затылок разламывается, мутит, тошнит",
    # 2) гипертония «до 150»
    "давление 150, подташнивает, слабость, к вечеру хуже",
    # 3) псориаз
    "на локтях и коленях годами красные бляшки с серебристым шелушением, чешутся умеренно, не проходят",
    # 4) менингит: текущая и усиленная
    "сильно ломит всё тело, высокая температура под 39 и сыпь мелкая, не бледнеет",
    "голова разрывается, свет невыносим, подбородок к груди не прижать, шея как камень, "
    "мелкая сыпь не бледнеет при надавливании, температура 39",
]

fn, label = None, None
for name in ("classify", "predict", "answer", "diagnose", "respond",
             "process", "handle_message", "chat", "run"):
    f = getattr(medbot, name, None)
    if callable(f):
        fn, label = f, name
        break
if fn is None:
    print("entry point не найден, публичные имена модуля:")
    print([n for n in dir(medbot) if not n.startswith("_")])
else:
    print("entry point:", label)
    for t in TEXTS:
        try:
            r = fn(t)
        except Exception as e:
            print(f"  {t[:60]!r} -> ERROR {type(e).__name__}: {e}")
            continue
        if isinstance(r, dict):
            top = r.get("top") or []
            cls = r.get("diagnosis") or (top[0][0] if top else None)
            print(f"  {t[:60]!r} -> cls={cls} conf={r.get('confidence')} "
                  f"urgent={r.get('urgent')} urgency={r.get('urgency')}")
        else:
            print(f"  {t[:60]!r} -> {repr(r)[:150]}")
