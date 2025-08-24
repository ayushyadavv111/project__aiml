import re
import textwrap

def clean_text(txt: str) -> str:
    if not txt:
        return ""
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

def split_sources(raw: str) -> list[str]:
    if raw is None:
        return []
    parts = [p.strip() for p in re.split(r'\n\s*---\s*\n', raw) if p.strip()]
    return parts

def wrap(s: str, width: int = 100) -> str:
    return "\n".join(textwrap.wrap(s, width=width))
