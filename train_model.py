"""
Train and compare baseline ML models for the ISOT Fake/True News dataset.
Expected files: data/Fake.csv and data/True.csv
"""
from pathlib import Path
import json
import re
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

def clean_text(text):
    """Light cleaning; TF-IDF handles tokenization and n-grams."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_dataset():
    fake_path = DATA_DIR / "Fake.csv"
    true_path = DATA_DIR / "True.csv"
    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            "Dataset files missing. Download Fake.csv and True.csv and place "
            "both inside the project's data/ folder. See README.md."
        )
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    for frame in (fake, true):
        if "text" not in frame.columns:
            raise ValueError("CSV must contain a 'text' column.")
    fake["label"] = 0
    true["label"] = 1
    data = pd.concat([fake, true], ignore_index=True)
    data["title"] = data.get("title", "").fillna("").astype(str)
    data["text"] = data["text"].fillna("").astype(str)
    data["content"] = (data["title"] + " " + data["text"]).map(clean_text)
    data = data[data["content"].str.len() > 0]
    data = data.drop_duplicates(subset=["content"]).sample(frac=1, random_state=42)
    return data[["content", "label"]]

def main():
    print("Loading dataset...")
    data = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        data["content"], data["label"], test_size=0.20,
        random_state=42, stratify=data["label"]
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(class_weight="balanced")
    }

    results = []
    best_name, best_pipeline, best_f1 = None, None, -1

    for name, classifier in candidates.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english", max_features=100000,
                ngram_range=(1, 2), min_df=2, max_df=0.95,
                sublinear_tf=True
            )),
            ("classifier", classifier)
        ])
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = {
            "model": name,
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4)
        }
        results.append(metrics)
        print(metrics)
        if metrics["f1_score"] > best_f1:
            best_name, best_pipeline, best_f1 = name, pipeline, metrics["f1_score"]
            best_predictions = predictions

    joblib.dump(best_pipeline, MODEL_DIR / "fake_news_pipeline.joblib")
    summary = {
        "best_model": best_name,
        "dataset_rows_after_cleaning": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "label_mapping": {"0": "Likely Fake", "1": "Likely Real"},
        "model_results": results,
        "classification_report": classification_report(
            y_test, best_predictions, target_names=["Likely Fake", "Likely Real"],
            output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, best_predictions).tolist()
    }
    with open(REPORT_DIR / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(results).to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print(f"\nBest model: {best_name}")
    print("Saved model to model/fake_news_pipeline.joblib")
    print("Saved metrics to reports/")

if __name__ == "__main__":
    main()
