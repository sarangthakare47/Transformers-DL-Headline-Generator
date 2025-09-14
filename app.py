import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

MODEL_NAME = "sshleifer/distilbart-xsum-12-6" 
@st.cache_resource 
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return pipeline("summarization", model=model, tokenizer=tokenizer)

headline_generator = load_model()

st.set_page_config(page_title="Headline Generator", page_icon="📰", layout="centered")
st.title("📰 AI-Powered Headline Generator")
st.write("Paste an article below, and I’ll generate a short headline for it.")

article = st.text_area("Enter Article Text:", height=250, placeholder="Paste your article here...")

# Length sliders
max_len = st.slider("Max Headline Length (words)", 5, 20, 12)

if st.button("✨ Generate Headline"):
    if article.strip():
        result = headline_generator(
            article,
            max_length=max_len,
            do_sample=False,
            num_beams=4,
            early_stopping=True
        )
        headline = result[0]['summary_text'].split(".")[0].strip()

        st.subheader("✅ Generated Headline:")
        st.success(headline)
    else:
        st.warning("⚠️ Please enter some text.")
