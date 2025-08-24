from __future__ import annotations
import requests
from bs4 import BeautifulSoup
from typing import Optional
from utils import clean_text

def _extract_with_bs4(url: str) -> Optional[str]:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, timeout=15, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        selectors = ["article", "main", "[role='main']", ".article-body", ".story-body", ".post-content", ".entry-content"]
        texts = []
        for sel in selectors:
            for node in soup.select(sel):
                ps = [p.get_text(" ", strip=True) for p in node.find_all(["p","h2","li"])]
                if ps:
                    texts.append(" ".join(ps))
        if not texts:
            ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            texts.append(" ".join(ps))
        content = max(texts, key=len) if texts else ""
        return clean_text(content)
    except Exception:
        return None

def extract_article(url: str) -> str:
    try:
        from newspaper import Article
        art = Article(url)
        art.download()
        art.parse()
        txt = art.text or ""
        if txt.strip():
            return clean_text(txt)
    except Exception:
        pass
    txt = _extract_with_bs4(url) or ""
    return clean_text(txt)

def extract_multiple(urls: list[str]) -> list[tuple[str, str]]:
    out = []
    for u in urls:
        u = u.strip()
        if not u:
            continue
        txt = extract_article(u)
        out.append((u, txt))
    return out
