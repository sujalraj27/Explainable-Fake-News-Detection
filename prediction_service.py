import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from lime.lime_text import LimeTextExplainer

# Knowledge Graph
from knowledge_graph.entity_extractor import extract_entities
from knowledge_graph.graph_generator import generate_knowledge_graph

# AI Explanation
from services.llm_service import generate_explanation

# PDF Report
from services.pdf_service import generate_pdf_report


# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "saved" / "logistic_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "saved" / "tfidf_vectorizer.pkl"


# =====================================================
# Load Model
# =====================================================

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# =====================================================
# LIME Prediction Function
# =====================================================

def predict_probability(texts):
    vectors = vectorizer.transform(texts)
    return model.predict_proba(vectors)


# =====================================================
# LIME Explainer
# =====================================================

explainer = LimeTextExplainer(
    class_names=["FAKE", "REAL"]
)


# =====================================================
# Main Prediction Function
# =====================================================

def predict_news(text):

    # -----------------------------
    # TF-IDF Vectorization
    # -----------------------------

    text_vector = vectorizer.transform([text])

    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(text_vector)[0]

    probabilities = model.predict_proba(text_vector)[0]

    fake_probability = round(probabilities[0] * 100, 2)
    real_probability = round(probabilities[1] * 100, 2)

    confidence = round(
        max(fake_probability, real_probability),
        2
    )

    prediction_label = (
        "REAL"
        if prediction == 1
        else "FAKE"
    )

    # -----------------------------
    # LIME Explanation
    # -----------------------------

    explanation = explainer.explain_instance(
        text,
        predict_probability,
        num_features=10
    )

    lime_output = []

    for word, weight in explanation.as_list():

        lime_output.append(
            {
                "word": word,
                "weight": round(weight, 4)
            }
        )
            # -----------------------------
    # Knowledge Graph
    # -----------------------------

    entities = extract_entities(text)

    graph_files = generate_knowledge_graph(entities)

    graph_html = graph_files["html"]

    graph_png = graph_files["png"]

    # -----------------------------
    # Results Folder
    # -----------------------------

    results_folder = BASE_DIR / "results"

    results_folder.mkdir(exist_ok=True)

    # -----------------------------
    # Probability Chart
    # -----------------------------

    probability_chart = results_folder / "probability_chart.png"

    plt.figure(figsize=(5, 4))

    plt.bar(
        ["FAKE", "REAL"],
        [fake_probability, real_probability],
        color=["red", "green"]
    )

    plt.ylabel("Probability (%)")

    plt.title("Prediction Probability")

    plt.ylim(0, 100)

    plt.tight_layout()

    plt.savefig(probability_chart)

    plt.close()

    # -----------------------------
    # LIME Graph
    # -----------------------------

    lime_graph = results_folder / "lime_graph.png"

    words = [item["word"] for item in lime_output]

    weights = [item["weight"] for item in lime_output]

    colors = []

    for weight in weights:

        if weight >= 0:
            colors.append("green")
        else:
            colors.append("red")

    plt.figure(figsize=(8, 5))

    plt.barh(
        words,
        weights,
        color=colors
    )

    plt.xlabel("Contribution Weight")

    plt.ylabel("Words")

    plt.title("LIME Feature Importance")

    plt.tight_layout()

    plt.savefig(lime_graph)

    plt.close()

    # -----------------------------
    # AI Explanation
    # -----------------------------

    ai_explanation = generate_explanation(
        prediction_label,
        confidence,
        lime_output
    )
        # -----------------------------
    # PDF Report
    # -----------------------------

    pdf_path = generate_pdf_report(

        text,

        {

            "prediction": prediction_label,

            "confidence": confidence,

            "probabilities": {

                "FAKE": fake_probability,

                "REAL": real_probability

            },

            "probability_chart": str(probability_chart),

            "lime_graph": str(lime_graph),

            "knowledge_graph": str(graph_png),

            "lime_explanation": lime_output,

            "ai_explanation": ai_explanation

        }

    )

    # -----------------------------
    # Final Response
    # -----------------------------

    return {

        "prediction": prediction_label,

        "confidence": confidence,

        "probabilities": {

            "FAKE": fake_probability,

            "REAL": real_probability

        },

        "lime_explanation": lime_output,

        "knowledge_graph": str(graph_html),

        "knowledge_graph_image": str(graph_png),

        "probability_chart": str(probability_chart),

        "lime_graph": str(lime_graph),

        "ai_explanation": ai_explanation,

        "pdf_report": str(pdf_path)

    }