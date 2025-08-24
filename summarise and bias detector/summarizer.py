from __future__ import annotations
from utils import clean_text
import os

try:
    from transformers import pipeline
    _hf_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
except Exception:
    _hf_summarizer = None

_USE_GEMINI = False
try:
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        genai.configure(api_key=api_key)
        _gemini = genai.GenerativeModel("gemini-1.5-flash")
        _USE_GEMINI = True
    else:
        _gemini = None
except Exception:
    _gemini = None
    _USE_GEMINI = False

def _summarize_hf(text: str, max_words: int = 140) -> str:
    if _hf_summarizer is None:
        return ""
    max_len = max(30, min(300, int(max_words * 1.3)))
    min_len = max(15, int(max_len * 0.5))
    try:
        out = _hf_summarizer(text[:4000], max_length=max_len, min_length=min_len, do_sample=False)
        return clean_text(out[0]["summary_text"])
    except Exception:
        return ""

def _summarize_gemini(text: str, tone: str, max_words: int = 140) -> str:
    if not _USE_GEMINI or _gemini is None:
        return ""
    prompt = f"""Summarize the news article below in under {max_words} words.
Tone mode: "{tone}".
- "neutral summary": balanced, concise.
- "fact-only": only verifiable facts, avoid adjectives/adverbs, no opinionated words.
- "explain to a 10-year-old": very simple words and short sentences.

Article:
{text}
"""
    try:
        resp = _gemini.generate_content(prompt)
        return clean_text(getattr(resp, "text", "") or "")
    except Exception:
        return ""

def _postprocess_fact_only(summary: str) -> str:
    if not summary:
        return summary
    import re
    summary = re.sub(r"\b(reportedly|allegedly|apparently|critics say|supporters say)\b", "", summary, flags=re.I)
    summary = re.sub(r"\b\w+ly\b", "", summary)
    summary = re.sub(r"\s{2,}", " ", summary).strip()
    return summary

def _postprocess_el10(summary: str) -> str:
    if not summary:
        return summary
    import re
    sents = re.split(r"(?<=[.!?])\s+", summary)
    sents = [s.strip() for s in sents if s.strip()]
    out = []
    for s in sents:
        words = s.split()
        if len(words) > 18:
            out.append(" ".join(words[:18]) + ".")
        else:
            out.append(s if s.endswith(('.', '!', '?')) else s + ".")
    repl = {
        "approximately": "about",
        "utilize": "use",
        "individuals": "people",
        "assistance": "help",
        "regulation": "rule",
        "authorize": "allow",
        "terminate": "end",
        "purchase": "buy",
        "children": "kids",
        "numerous": "many",
    }
    text = " ".join(out)
    for k, v in repl.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.I)
    return text

def summarize(text: str, tone: str = "neutral summary", max_words: int = 140, prefer_gemini: bool = True) -> str:
    text = clean_text(text)
    if not text:
        return ""

    if prefer_gemini and _USE_GEMINI:
        s = _summarize_gemini(text, tone, max_words=max_words)
        if s:
            if tone.lower().startswith("fact"):
                s = _postprocess_fact_only(s)
            elif "10" in tone:
                s = _postprocess_el10(s)
            return s

    base = _summarize_hf(text, max_words=max_words)
    if tone.lower().startswith("fact"):
        return _postprocess_fact_only(base)
    if "10" in tone:
        return _postprocess_el10(base)
    return base
