from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATASET = BASE_DIR / "data" / "raw" / "WELFake_Dataset.csv"
PROCESSED_DATASET = BASE_DIR / "data" / "processed" / "clean_dataset.csv"


def clean_dataset():

    # Load Dataset
    df = pd.read_csv(RAW_DATASET)

    # Remove unnecessary column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Remove missing values
    df = df.dropna()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Save cleaned dataset
    df.to_csv(PROCESSED_DATASET, index=False)

    return {
        "Rows After Cleaning": len(df),
        "Columns": list(df.columns),
        "Saved To": str(PROCESSED_DATASET)
    }