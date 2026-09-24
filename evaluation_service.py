import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ===========================
# Paths
# ===========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = BASE_DIR / "data" / "processed" / "preprocessed_dataset.csv"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "saved"
    / "logistic_model.pkl"
)

VECTORIZER_PATH = (
    BASE_DIR
    / "models"
    / "saved"
    / "tfidf_vectorizer.pkl"
)

RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

# ===========================
# Load Model
# ===========================

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def evaluate_model():

    # -----------------------
    # Load Dataset
    # -----------------------

    df = pd.read_csv(DATASET)

    df = df.dropna(subset=["processed_text", "label"])

    X = df["processed_text"]
    y = df["label"]

    # -----------------------
    # TF-IDF
    # -----------------------

    X = vectorizer.transform(X)

    # -----------------------
    # Same Split as Training
    # -----------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # -----------------------
    # Prediction
    # -----------------------

    y_pred = model.predict(X_test)

    # -----------------------
    # Metrics
    # -----------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n========== Evaluation ==========\n")

    print(f"Accuracy  : {accuracy*100:.2f}%")
    print(f"Precision : {precision*100:.2f}%")
    print(f"Recall    : {recall*100:.2f}%")
    print(f"F1 Score  : {f1*100:.2f}%")

    # -----------------------
    # Confusion Matrix
    # -----------------------

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake","Real"],
        yticklabels=["Fake","Real"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    cm_path = RESULT_DIR / "confusion_matrix.png"

    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------
    # Classification Report
    # -----------------------

    report = classification_report(y_test, y_pred)

    report_path = RESULT_DIR / "classification_report.txt"

    with open(report_path, "w") as f:
        f.write(report)

    # -----------------------
    # Metrics JSON
    # -----------------------

    metrics = {
        "Accuracy": round(accuracy,4),
        "Precision": round(precision,4),
        "Recall": round(recall,4),
        "F1 Score": round(f1,4)
    }

    metrics_path = RESULT_DIR / "metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nConfusion Matrix Saved:", cm_path)
    print("Classification Report Saved:", report_path)
    print("Metrics Saved:", metrics_path)

    return {
        "metrics": metrics,
        "confusion_matrix": str(cm_path),
        "classification_report": str(report_path)
    }


if __name__ == "__main__":
    evaluate_model()