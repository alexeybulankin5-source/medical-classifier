import os, joblib, numpy as np, torch
from sentence_transformers import SentenceTransformer

# Папка с артефактами = папка, где лежит сам файл medbot.py
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class MedBot:
    def __init__(self):
        path = os.path.join(_BASE_DIR, "pipeline.joblib")
        self.pipe = joblib.load(path)
        self.router = self.pipe["router"]
        self.collo, self.mixed = self.pipe["colloquial"], self.pipe["mixed"]
        self.enc = SentenceTransformer(
            "ai-forever/sbert_large_nlu_ru",
            device="cuda" if torch.cuda.is_available() else "cpu")
        self.enc.max_seq_length = 256

    def predict_top3(self, text: str) -> dict:
        emb = self.enc.encode([text])[0]
        route = self.router.predict(emb.reshape(1, -1))[0]
        if route == 0:
            model, classes, branch = self.collo["model"], self.collo["classes"], "colloquial"
        else:
            model, classes, branch = self.mixed["model"], self.mixed["classes"], "clinical"
        p = model.predict_proba(emb.reshape(1, -1))[0]
        idx = np.argsort(p)[::-1][:3]
        top = [(classes[i], float(p[i])) for i in idx]
        margin = top[0][1] - top[1][1]
        if margin >= 0.12:
            action, verdict = "answer", f"Наиболее вероятно: {top[0][0]}"
        elif margin >= 0.08:
            action, verdict = "answer_with_caveat", f"Наиболее вероятно: {top[0][0]}. Точную причину подтвердит врач"
        elif margin >= 0.04:
            action, verdict = "top3", "Похоже на одно из: " + ", ".join(n for n, _ in top)
        else:
            action, verdict = "clarify", "Симптомы неоднозначны — нужны уточняющие вопросы"
        return {"verdict": verdict, "action": action, "branch": branch,
                "top3": top, "margin": round(margin, 3)}
