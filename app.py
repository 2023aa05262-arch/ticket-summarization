
import streamlit as st
from transformers import pipeline

# -------------------------------
# Load models from Hugging Face
# -------------------------------
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="AmitBits/t5-summary-model")
    classifier = pipeline("text-classification", model="AmitBits/routing-classifier-model")
    return summarizer, classifier

summarizer, classifier = load_models()

# -------------------------------
# UI Layout
# -------------------------------
st.set_page_config(page_title="Ticket Summarization & Routing", layout="wide")
st.title("🎫 Smart Ticket Summarizer & Router")

text_input = st.text_area("Enter your support ticket:", height=150)

if st.button("🔍 Generate Summary & Route"):
    if not text_input.strip():
        st.warning("Please enter a ticket description.")
    else:
        with st.spinner("Processing..."):
            # Generate summary
            summary = summarizer(text_input, max_length=64, min_length=10, do_sample=False)[0]["summary_text"]
            # Predict routing label
            route = classifier(text_input)[0]

        st.subheader("✂️ Summary")
        st.success(summary)

        st.subheader("📍 Routing Department")
        st.info(f"{route['label']} (score: {route['score']:.2f})")
