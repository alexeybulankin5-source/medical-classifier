# -*- coding: utf-8 -*-
import os, json, urllib.request

_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

def _model_name():
    return os.environ.get("YC_MODEL", "yandexgpt-lite")

def ask_yandex_gpt(system_text, user_text, history=None,
                   temperature=0.3, max_tokens=600):
    key = os.environ.get("YC_API_KEY", "").strip()
    folder = os.environ.get("YC_FOLDER_ID", "").strip()
    if not key or not folder:
        return "Ключи не настроены: задайте YC_API_KEY и YC_FOLDER_ID в окружении и перезапустите приложение."
    msgs = [{"role": "system", "text": system_text}]
    for h in (history or []):
        msgs.append({"role": h["role"], "text": h["text"]})
    msgs.append({"role": "user", "text": user_text})
    body = {"modelUri": "gpt://" + folder + "/" + _model_name(),
            "completionOptions": {"temperature": temperature,
                                  "maxTokens": max_tokens},
            "messages": msgs}
    req = urllib.request.Request(_URL, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Api-Key " + key,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        return data["result"]["alternatives"][0]["message"]["text"]
    except urllib.error.HTTPError as e:
        return "Ошибка API: HTTP " + str(e.code) + " " + e.read().decode()[:300]
    except Exception as e:
        return "Ошибка соединения: " + str(e)