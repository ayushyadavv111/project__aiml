# D8 News Summarizer & Bias Detector

An end-to-end Streamlit app that:
- Extracts article text from **URLs** or accepts **pasted text**
- Generates summaries in 3 tones: **Neutral**, **Fact-only**, **Explain like I'm 10**
- Flags **potential bias** by analyzing sentiment, charged wording, and comparing multiple sources
- (Optional) Uses **Gemini API** for higher-quality tone-controlled summaries

---

## 🚀 Quickstart

### 1) Create & activate a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
# NLTK resources (first run only)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```
If you get errors installing `torch`, try:
```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3) (Optional) Enable Gemini
Create a `.env` file or set an environment variable with your key:
```bash
export GEMINI_API_KEY="YOUR_KEY_HERE"   # Windows (Powershell): $Env:GEMINI_API_KEY="YOUR_KEY_HERE"
```

### 4) Run the app
```bash
streamlit run app.py
```

---

## 🧠 How bias is flagged

- **Sentiment**: Hugging Face SST-2 model + NLTK **VADER** for cross-checking.
- **Charged words**: Counts presence and density of emotionally loaded terms.
- **Comparative view**: Provide multiple sources to compare tone & sentiment across them.

The output is **heuristic** and meant to **assist** critical reading — not to declare absolute truth.
