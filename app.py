from pathlib import Path
import re
import json
import joblib
import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "fake_news_pipeline.joblib"
METRICS_PATH = ROOT / "reports" / "evaluation.json"

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

st.title("📰 Fake News Detection System")
st.caption("A machine-learning and NLP mini project using TF-IDF and supervised classifiers.")

model = load_model()
if model is None:
    st.error("The trained model was not found.")
    st.markdown("Run `python train_model.py` after placing `Fake.csv` and `True.csv` inside the `data/` folder. See README.md.")
    st.stop()

st.info(
    "Educational demonstration only. The model learns patterns from its training dataset; "
    "it does not verify facts or consult live sources. Treat every prediction as an initial signal, "
    "not proof that a story is true or false."
)

news = st.text_area(
    "Paste a news headline or article",
    height=220,
    placeholder="Paste the headline and/or full news article here..."
)

if st.button("Analyze News", type="primary", use_container_width=True):
    cleaned = clean_text(news)
    if len(cleaned.split()) < 5:
        st.warning("Please enter a little more text (at least five words) for a meaningful demonstration.")
    else:
        prediction = int(model.predict([cleaned])[0])
        label = "Likely Real" if prediction == 1 else "Likely Fake"
        if hasattr(model, "decision_function"):
            score = float(model.decision_function([cleaned])[0])
            st.metric("Model output", label)
            st.caption(f"Classifier decision score: {score:.3f} (not a probability or factual confidence score)")
        else:
            st.metric("Model output", label)
            st.caption("This is a model prediction, not a verified fact-check.")
        st.warning("Please verify the claim using multiple reputable, independent sources before sharing.")

st.divider()
st.subheader("Model evaluation")
if METRICS_PATH.exists():
    with open(METRICS_PATH, encoding="utf-8") as f:
        report = json.load(f)
    st.write(f"**Best model:** {report['best_model']}")
    st.write(
        f"Dataset rows: {report['dataset_rows_after_cleaning']:,} · "
        f"Training rows: {report['train_rows']:,} · Test rows: {report['test_rows']:,}"
    )
    metrics_df = pd.DataFrame(report["model_results"])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
else:
    st.caption("Evaluation metrics will appear here after you run train_model.py.")

st.caption("Project: Fake News Detection using Machine Learning and Natural Language Processing.")
