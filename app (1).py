# ============================================================
# FAKE NEWS DETECTOR — STREAMLIT APP
# This is what runs on Render (your live website)
# ============================================================

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Load stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Load the trained model and vectorizer (the 2 .pkl files)
model      = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Same cleaning function used during training
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# ── Page setup ───────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", page_icon="🔍")
st.title("🔍 Fake News Detection System")
st.markdown("**NLP & Machine Learning · Deployed on Render**")
st.markdown("---")

# ── Input box ────────────────────────────────────────────────
news_input = st.text_area(
    "Paste a news headline or article below:",
    height=200,
    placeholder="Enter news text here..."
)

# ── Predict button ───────────────────────────────────────────
if st.button("Analyse", use_container_width=True):
    if not news_input.strip():
        st.warning("Please enter some text first.")
    else:
        # Clean → vectorize → predict
        cleaned  = clean_text(news_input)
        features = vectorizer.transform([cleaned])
        pred     = model.predict(features)[0]
        prob     = model.predict_proba(features)[0]

        st.markdown("---")
        if pred == 1:
            st.success("✅ REAL NEWS")
            st.metric("Confidence", f"{prob[1]*100:.1f}%")
        else:
            st.error("❌ FAKE NEWS")
            st.metric("Confidence", f"{prob[0]*100:.1f}%")
