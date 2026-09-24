import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from models.vectorizer.tfidf_vectorizer import create_vectorizer

from models.ml.logistic_model import create_model as logistic_model
from models.ml.naive_model import create_model as naive_model
from models.ml.svm_model import create_model as svm_model


# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = (
    BASE_DIR
    / "data"
    / "processed"
    / "preprocessed_dataset.csv"
)

RESULT_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models" / "saved"

RESULT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# Evaluation Function
# ==========================================

def evaluate(model, X_train, X_test, y_train, y_test):

    # Train model
    model.fit(X_train, y_train)

    # Prediction
    prediction = model.predict(X_test)

    # Basic metrics
    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0
    )

    # ROC-AUC
    roc_auc = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

    elif hasattr(model, "decision_function"):

        scores = model.decision_function(X_test)

        roc_auc = roc_auc_score(
            y_test,
            scores
        )

    return {

        "Accuracy": round(accuracy, 4),

        "Precision": round(
            precision,
            4
        ),

        "Recall": round(
            recall,
            4
        ),

        "F1 Score": round(
            f1,
            4
        ),

        "ROC-AUC": round(
            roc_auc,
            4
        ) if roc_auc is not None else None,

        "Confusion Matrix":
            confusion_matrix(
                y_test,
                prediction
            ).tolist(),

        "Model": model
    }


# ==========================================
# Train All Models
# ==========================================

def train_all_models():

    print("\n================================")
    print("Starting Model Training")
    print("================================\n")

    # --------------------------------------
    # Load Dataset
    # --------------------------------------

    df = pd.read_csv(DATASET)

    df = df.dropna(
        subset=[
            "processed_text",
            "label"
        ]
    )

    X = df["processed_text"]

    y = df["label"]

    print("Dataset Size:", len(df))

    # --------------------------------------
    # Train/Test Split FIRST
    # --------------------------------------

    X_train_text, X_test_text, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    print("Training Samples:", len(X_train_text))
    print("Testing Samples :", len(X_test_text))

    # --------------------------------------
    # TF-IDF
    # --------------------------------------

    vectorizer = create_vectorizer()

    # IMPORTANT:
    # Fit only on training data

    X_train = vectorizer.fit_transform(
        X_train_text
    )

    # Transform test data only

    X_test = vectorizer.transform(
        X_test_text
    )

    print(
        "TF-IDF Features:",
        X_train.shape[1]
    )

    results = {}

    # ======================================
    # Logistic Regression
    # ======================================

    print("\nTraining Logistic Regression...")

    logistic = evaluate(

        logistic_model(),

        X_train,
        X_test,

        y_train,
        y_test
    )

    joblib.dump(

        logistic["Model"],

        MODEL_DIR / "logistic_model.pkl"
    )

    results["Logistic Regression"] = {

        key: value

        for key, value in logistic.items()

        if key != "Model"
    }

    # ======================================
    # Naive Bayes
    # ======================================

    print("Training Naive Bayes...")

    naive = evaluate(

        naive_model(),

        X_train,
        X_test,

        y_train,
        y_test
    )

    joblib.dump(

        naive["Model"],

        MODEL_DIR / "naive_model.pkl"
    )

    results["Naive Bayes"] = {

        key: value

        for key, value in naive.items()

        if key != "Model"
    }

    # ======================================
    # SVM
    # ======================================

    print("Training Linear SVM...")

    svm = evaluate(

        svm_model(),

        X_train,
        X_test,

        y_train,
        y_test
    )

    joblib.dump(

        svm["Model"],

        MODEL_DIR / "svm_model.pkl"
    )

    results["SVM"] = {

        key: value

        for key, value in svm.items()

        if key != "Model"
    }

    # ======================================
    # Save TF-IDF
    # ======================================

    joblib.dump(

        vectorizer,

        MODEL_DIR / "tfidf_vectorizer.pkl"
    )

    # ======================================
    # Save JSON
    # ======================================

    metrics_path = (
        RESULT_DIR
        / "model_comparison.json"
    )

    with open(
        metrics_path,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    # ======================================
    # Save CSV
    # ======================================

    comparison = []

    for model_name, value in results.items():

        comparison.append({

            "Model": model_name,

            "Accuracy":
                value["Accuracy"],

            "Precision":
                value["Precision"],

            "Recall":
                value["Recall"],

            "F1 Score":
                value["F1 Score"],

            "ROC-AUC":
                value["ROC-AUC"]
        })

    comparison_path = (
        RESULT_DIR
        / "model_comparison.csv"
    )

    pd.DataFrame(
        comparison
    ).to_csv(

        comparison_path,

        index=False
    )

    # ======================================
    # Print Results
    # ======================================

    print("\n================================")
    print("MODEL RESULTS")
    print("================================\n")

    for model_name, value in results.items():

        print(model_name)

        print(
            "Accuracy :",
            value["Accuracy"]
        )

        print(
            "Precision:",
            value["Precision"]
        )

        print(
            "Recall   :",
            value["Recall"]
        )

        print(
            "F1 Score :",
            value["F1 Score"]
        )

        print(
            "ROC-AUC  :",
            value["ROC-AUC"]
        )

        print()

    print("Training Completed.")

    return {

        "Status":
            "Training Completed",

        "Results":
            results,

        "Comparison CSV":
            str(comparison_path),

        "Metrics JSON":
            str(metrics_path)
    }