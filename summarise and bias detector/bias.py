from __future__ import annotations
from typing import List, Dict, Any
import re
from dataclasses import dataclass
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

try:
    from transformers import pipeline
    _hf_sent = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception:
    _hf_sent = None

_CHARGED_TERMS_STR = """
alarming outrageous scandal shocking disgraceful corrupt devastated chaos furious tragic failure
booming outstanding miracle rescued heroic victory unprecedented collapse disastrous catastrophic
"""
CHARGED_WORDS = set(w.lower() for w in _CHARGED_TERMS_STR.split())

@dataclass
class BiasScores:
    label: str
    vader: float
    hf_label: str
    hf_score: float
    charged_ratio: float

def _count_charged(text: str) -> float:
    toks = re.findall(r"[A-Za-z']+", text.lower())
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in CHARGED_WORDS)
    return hits / max(1, len(toks))

def _sentiment_hf(text: str) -> tuple[str, float]:
    if _hf_sent is None:
        return ("UNKNOWN", 0.0)
    try:
        res = _hf_sent(text[:4500])[0]
        return (res["label"], float(res["score"]))
    except Exception:
        return ("UNKNOWN", 0.0)

def score_bias(text: str) -> BiasScores:
    if not text.strip():
        return BiasScores(label="Unknown", vader=0.0, hf_label="UNKNOWN", hf_score=0.0, charged_ratio=0.0)
    sia = SentimentIntensityAnalyzer()
    v = sia.polarity_scores(text[:5000])["compound"]
    hf_label, hf_score = _sentiment_hf(text)
    charged = _count_charged(text)

    def bucket(compound: float) -> str:
        if abs(compound) < 0.1 and charged < 0.001:
            return "Likely Neutral"
        if compound >= 0.25:
            return "Leaning Positive"
        if compound <= -0.25:
            return "Leaning Negative"
        return "Slightly Biased"

    label = bucket(v)
    return BiasScores(label=label, vader=v, hf_label=hf_label, hf_score=hf_score, charged_ratio=charged)

def compare_sources(texts: List[str]) -> Dict[str, Any]:
    scores = [score_bias(t) for t in texts]
    if not scores:
        return {"scores": [], "disagreement": 0.0}
    import statistics
    vader_vals = [s.vader for s in scores]
    if len(vader_vals) > 1:
        try:
            disagreement = statistics.pstdev(vader_vals)
        except statistics.StatisticsError:
            disagreement = 0.0
    else:
        disagreement = 0.0
    return {"scores": [s.__dict__ for s in scores], "disagreement": float(disagreement)}
