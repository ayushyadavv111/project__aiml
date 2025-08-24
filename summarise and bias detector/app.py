import streamlit as st
from utils import wrap, split_sources
from scrape import extract_multiple
from summarizer import summarize
from bias import score_bias, compare_sources

st.set_page_config(page_title="D8 News Summarizer & Bias Detector", layout="wide")
st.title("📰 D8 News Summarizer & Bias Detector")

st.markdown("""
Paste **one or more URLs** *(one per line)* **or** paste raw article text (separate multiple articles with a line `---`).
""")
col1, col2 = st.columns([1,1])
with col1:
    urls_raw = st.text_area("URLs (one per line)", height=180, placeholder="https://example.com/news-1\nhttps://example.com/news-2")
with col2:
    pasted = st.text_area("Or paste article text (use `---` line to split multiple)", height=180)

tone = st.selectbox("Summary tone", ["neutral summary", "fact-only", "explain to a 10-year-old"])
use_gemini = st.toggle("Prefer Gemini (if API key configured)", value=True)
max_words = st.slider("Max summary length (words)", 80, 300, 140, 10)

if st.button("Summarize & Detect Bias", type="primary"):
    urls = [u.strip() for u in urls_raw.splitlines() if u.strip()]
    sources = []
    texts = []

    if urls:
        rows = extract_multiple(urls)
        for src, txt in rows:
            sources.append(src)
            texts.append(txt)
    if pasted.strip():
        for part in split_sources(pasted):
            sources.append("pasted")
            texts.append(part)

    if not texts:
        st.warning("Please provide at least one URL or paste article text.")
        st.stop()

    summaries = []
    biases = []
    for src, txt in zip(sources, texts):
        s = summarize(txt, tone=tone, max_words=max_words, prefer_gemini=use_gemini)
        b = score_bias(txt)
        summaries.append((src, s))
        biases.append(b)

    comp = compare_sources(texts)

    for i, ((src, summ), b) in enumerate(zip(summaries, biases), start=1):
        st.subheader(f"Source {i}: {src}")
        st.write("**Summary:**")
        st.write(wrap(summ or "_(no summary)_", width=100))

        st.write("**Bias signals:**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall", b.label)
        c2.metric("VADER", f"{b.vader:+.3f}")
        c3.metric("HF Label", b.hf_label)
        c4.metric("Charged word ratio", f"{b.charged_ratio:.4f}")
        st.markdown("---")

    st.subheader("Across-source comparison")
    st.write(f"Disagreement (VADER std-dev): **{comp['disagreement']:.3f}**  — higher means sources differ more in tone.")
else:
    st.info("Enter URLs or paste text, choose a tone, then click **Summarize & Detect Bias**.")

st.caption("Streamlit • Hugging Face Transformers • NLTK VADER • (optional) Gemini")
