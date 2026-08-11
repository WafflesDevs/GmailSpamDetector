"""Train a bigger spam model from multiple datasets. Prefer high precision."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "spam_model.joblib"


def spam_threshold():
    return float(__import__("os").getenv("SPAM_THRESHOLD", "0.80"))


def _norm(df):
    df = df[["text", "spam"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.slice(0, 5000)
    df["spam"] = df["spam"].astype(int)
    return df[df["text"].str.len() > 15]


def load_data():
    parts = []
    data = ROOT / "data"

    # Kaggle spam-email-dataset
    parts.append(_norm(pd.read_csv(data / "emails.csv")))

    # Combined dump if present (Enron etc.)
    combined = data / "combined_train.csv"
    if combined.exists():
        parts.append(_norm(pd.read_csv(combined)))

    # Local Enron CSV
    enron = data / "enron_spam_data.csv"
    if enron.exists():
        e = pd.read_csv(enron)
        text = e["Subject"].fillna("").astype(str) + "\n" + e["Message"].fillna("").astype(str)
        spam = (e["Spam/Ham"].astype(str).str.lower() == "spam").astype(int)
        parts.append(_norm(pd.DataFrame({"text": text, "spam": spam})))

    # HuggingFace SpamAssassin if available via datasets
    try:
        from datasets import load_dataset

        sa = load_dataset("bvk/SpamAssassin-spam", split="train").to_pandas()
        # data column is a stringified list-ish; use as text
        parts.append(_norm(pd.DataFrame({"text": sa["data"].astype(str), "spam": sa["label"].astype(int)})))
    except Exception as exc:
        print("spamassassin skip:", exc)

    custom = data / "custom_spam.csv"
    if custom.exists():
        parts.append(_norm(pd.read_csv(custom)))

    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["text"])
    print(f"training rows: {len(df)}  spam={int(df.spam.sum())} ham={int((df.spam==0).sum())}")
    return df


def train():
    df = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["spam"], test_size=0.2, random_state=42, stratify=df["spam"]
    )

    # Bigger TF-IDF vocab for a stronger model
    tfidf = dict(max_features=50000, ngram_range=(1, 2), stop_words="english", min_df=2)
    models = {
        "logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0)),
        ]),
        "knn": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, stop_words="english", min_df=2)),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]),
    }

    best_name, best_model, best_score = None, None, (-1.0, -1.0)
    for name, model in models.items():
        print(f"fitting {name}...")
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        acc = accuracy_score(y_test, preds)
        # Prefer precision on spam so LLM only sees real spam
        prec = precision_score(y_test, preds, zero_division=0)
        score = (prec, acc)
        print(f"{name}: acc={acc:.4f} spam_precision={prec:.4f}")
        print(classification_report(y_test, preds, digits=3))
        if score > best_score:
            best_name, best_model, best_score = name, model, score

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"saved {best_name} precision={best_score[0]:.4f} acc={best_score[1]:.4f} -> {MODEL_PATH}")


def spam_confidence(text, model=None):
    """Return P(spam)."""
    if model is None:
        model = joblib.load(MODEL_PATH)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        classes = list(model.classes_)
        return float(proba[classes.index(1)])
    return float(model.predict([text])[0])


def is_spam(text, model=None, threshold=None):
    """True only when model is highly confident it's real spam."""
    if threshold is None:
        threshold = spam_threshold()
    return spam_confidence(text, model) >= threshold


if __name__ == "__main__":
    train()
