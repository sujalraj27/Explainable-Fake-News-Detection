from lime.lime_text import LimeTextExplainer
import joblib
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load Model
MODEL_PATH = BASE_DIR / "models" / "saved" / "logistic_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "saved" / "tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# Function required by LIME
def predict_probability(texts):
    vectors = vectorizer.transform(texts)
    return model.predict_proba(vectors)


def explain_prediction(text):

    explainer = LimeTextExplainer(
        class_names=["FAKE", "REAL"]
    )

    explanation = explainer.explain_instance(
        text,
        predict_probability,
        num_features=10
    )

    prediction = model.predict(
        vectorizer.transform([text])
    )[0]

    probabilities = model.predict_proba(
        vectorizer.transform([text])
    )[0]

    fake_probability = round(probabilities[0] * 100, 2)
    real_probability = round(probabilities[1] * 100, 2)

    confidence = max(fake_probability, real_probability)

    explanation_list = []

    for word, weight in explanation.as_list():

        explanation_list.append(
            {
                "word": word,
                "weight": round(weight, 4)
            }
        )

    return {

        "prediction": "REAL" if prediction == 1 else "FAKE",

        "confidence": confidence,

        "probabilities": {

            "FAKE": fake_probability,

            "REAL": real_probability

        },

        "lime_explanation": explanation_list

    }