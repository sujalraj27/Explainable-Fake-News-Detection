import pandas as pd
from pathlib import Path

from preprocessing.text_cleaner import clean_text
from preprocessing.tokenizer import tokenize
from preprocessing.stopwords import remove_stopwords

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DATA = BASE_DIR / "data" / "processed" / "clean_dataset.csv"

OUTPUT_DATA = BASE_DIR / "data" / "processed" / "preprocessed_dataset.csv"


def preprocess_dataset():

    df = pd.read_csv(INPUT_DATA)

    processed_text = []

    for text in df["text"]:

        cleaned = clean_text(text)

        tokens = tokenize(cleaned)

        tokens = remove_stopwords(tokens)

        processed_text.append(" ".join(tokens))

    df["processed_text"] = processed_text

    df.to_csv(OUTPUT_DATA, index=False)

    return {
        "message": "Preprocessing Completed Successfully",
        "Saved To": str(OUTPUT_DATA)
    }