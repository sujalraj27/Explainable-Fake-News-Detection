from lime.lime_text import LimeTextExplainer
import pickle
from pathlib import Path
import matplotlib.pyplot as plt



BASE_DIR = Path(__file__).resolve().parent.parent



MODEL_PATH = (
    BASE_DIR
    /
    "models"
    /
    "logistic_model.pkl"
)


VECTORIZER_PATH = (
    BASE_DIR
    /
    "models"
    /
    "tfidf_vectorizer.pkl"
)



def load_model():

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)


    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)


    return model, vectorizer





def generate_lime_explanation(news_text):


    model, vectorizer = load_model()



    explainer = LimeTextExplainer(
        class_names=[
            "REAL",
            "FAKE"
        ]
    )



    def predict_probability(texts):

        vectors = vectorizer.transform(
            texts
        )

        return model.predict_proba(
            vectors
        )



    explanation = explainer.explain_instance(

        news_text,

        predict_probability,

        num_features=10

    )



    words = []


    for word, weight in explanation.as_list():

        words.append(

            {
                "word": word,
                "weight": round(weight,4)
            }

        )



    # ----------------------------
    # Create LIME PNG Graph
    # ----------------------------

    results_dir = BASE_DIR / "results"

    results_dir.mkdir(
        exist_ok=True
    )


    image_path = (
        results_dir
        /
        "lime_graph.png"
    )



    labels = []

    values = []


    for item in words:

        labels.append(
            item["word"]
        )

        values.append(
            item["weight"]
        )



    plt.figure(
        figsize=(8,5)
    )


    plt.barh(
        labels,
        values
    )


    plt.xlabel(
        "Contribution Weight"
    )


    plt.title(
        "LIME Explanation"
    )


    plt.tight_layout()



    plt.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



    return {


        "words":
            words,


        "graph":
            str(image_path)

    }