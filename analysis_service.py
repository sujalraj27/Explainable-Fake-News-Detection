from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "processed" / "clean_dataset.csv"

GRAPH_FOLDER = BASE_DIR / "results" / "graphs"

GRAPH_FOLDER.mkdir(parents=True, exist_ok=True)


def dataset_summary():

    df = pd.read_csv(DATASET_PATH)

    return {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Column Names": list(df.columns),
        "Missing Values": df.isnull().sum().to_dict(),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Fake News": int((df["label"] == 1).sum()),
        "Real News": int((df["label"] == 0).sum())
    }


def plot_fake_real():

    df = pd.read_csv(DATASET_PATH)

    counts = df["label"].value_counts()

    plt.figure(figsize=(6,5))

    plt.bar(["Real News", "Fake News"], [counts[0], counts[1]])

    plt.title("Fake vs Real News Distribution")

    plt.xlabel("Category")

    plt.ylabel("Count")

    plt.tight_layout()

    graph_path = GRAPH_FOLDER / "fake_real_bar.png"

    plt.savefig(graph_path)

    plt.close()

    return {
        "message":"Bar Graph Generated Successfully",
        "path":str(graph_path)
    }


def plot_pie_chart():

    df = pd.read_csv(DATASET_PATH)

    counts = df["label"].value_counts()

    plt.figure(figsize=(6,6))

    plt.pie(
        [counts[0], counts[1]],
        labels=["Real", "Fake"],
        autopct="%1.1f%%"
    )

    plt.title("Fake vs Real News")

    graph_path = GRAPH_FOLDER / "fake_real_pie.png"

    plt.savefig(graph_path)

    plt.close()

    return {
        "message":"Pie Chart Generated Successfully",
        "path":str(graph_path)
    }


def generate_wordcloud():

    df = pd.read_csv(DATASET_PATH)

    fake_news = " ".join(df[df["label"]==1]["text"].astype(str))

    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    ).generate(fake_news)

    plt.figure(figsize=(12,6))

    plt.imshow(wordcloud)

    plt.axis("off")

    graph_path = GRAPH_FOLDER / "fake_wordcloud.png"

    plt.savefig(graph_path)

    plt.close()

    return {
        "message":"WordCloud Generated Successfully",
        "path":str(graph_path)
    }