import pandas as pd
from project.config import DATASET_PATH, COLUMNS

def load_dataset():
    """
    Load the Mafindo hoax dataset CSV and extract relevant columns.
    Returns a cleaned pandas DataFrame.
    """
    df = pd.read_csv(DATASET_PATH)
    df = df[COLUMNS].copy()

    # Basic cleaning: drop rows with missing values in important columns
    df.dropna(subset=["title", "content"], inplace=True)

    # Optional: concatenate fields for embedding
    df["text"] = df.apply(
        lambda row: f"{row['title']}\n\n{row['content']}\n\nFact: {row['fact']}\n\nConclusion: {row['conclusion']}",
        axis=1
    )

    return df
