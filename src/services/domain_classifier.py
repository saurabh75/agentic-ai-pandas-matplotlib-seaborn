"""Classify document domain. LLM-mode uses one small call; keyword-mode is instant."""
from __future__ import annotations
import os, re, json
from typing import Optional

DOMAINS = ["finance", "healthcare", "legal", "technical", "research", "hr", "business", "general"]

KEYWORDS = {
    "finance":   [r"revenue", r"ebitda", r"balance sheet", r"invoice", r"portfolio", r"equity", r"p&l", r"cash flow"],
    "healthcare":[r"patient", r"clinical", r"diagnos", r"treatment", r"icd-?10", r"medic(al|ation)", r"insurance claim"],
    "legal":     [r"plaintiff", r"defendant", r"clause", r"jurisdiction", r"agreement", r"whereas", r"liab(le|ility)"],
    "technical": [r"api", r"microservice", r"kubernetes", r"latency", r"architecture", r"schema", r"deployment"],
    "research":  [r"hypothesis", r"p-?value", r"methodology", r"literature review", r"experiment", r"abstract"],
    "hr":        [r"employee", r"recruit", r"payroll", r"onboarding", r"attrition", r"performance review"],
    "business":  [r"strategy", r"market share", r"kpi", r"roadmap", r"stakeholder", r"go-?to-?market"],
}


def classify_keyword(text: str) -> str:
    t = text.lower()
    scores = {d: 0 for d in KEYWORDS}
    for d, pats in KEYWORDS.items():
        for p in pats:
            scores[d] += len(re.findall(p, t))
    top = max(scores, key=scores.get)
    return top if scores[top] > 0 else "general"


def classify_llm(text: str, ollama_client, model: str) -> str:
    sample = text[:4000]
    prompt = (
        "Classify the document domain. Reply with ONE word from this list: "
        f"{', '.join(DOMAINS)}.\n\nDocument excerpt:\n{sample}\n\nDomain:"
    )
    try:
        resp = ollama_client.generate(model=model, prompt=prompt,
                                      options={"temperature": 0, "num_predict": 8})
        out = (resp.get("response") or "").strip().lower().split()[0]
        out = re.sub(r"[^a-z]", "", out)
        return out if out in DOMAINS else classify_keyword(text)
    except Exception:
        return classify_keyword(text)


def classify(text: str, ollama_client=None, model: Optional[str] = None) -> str:
    mode = os.getenv("PERSONA_DOMAIN_CLASSIFIER", "keyword").lower()
    if mode == "llm" and ollama_client and model:
        return classify_llm(text, ollama_client, model)
    return classify_keyword(text)
