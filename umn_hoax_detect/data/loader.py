import pandas as pd
from umn_hoax_detect.config import DATASET_PATH, COLUMNS


def load_dataset():
    """
    Load the Mafindo hoax dataset CSV and extract relevant columns.
    Returns a cleaned pandas DataFrame.
    """
    df = pd.read_csv(DATASET_PATH)
    df = df[COLUMNS].copy()

    print("Done Reading!")

    # Basic cleaning: drop rows with missing values in important columns
    df.dropna(subset=["title", "content"], inplace=True)

    # Truncate text fields to fit Milvus VARCHAR max_length constraints
    df["title"] = df["title"].astype(str).str.slice(0, 512)
    df["content"] = df["content"].astype(str).str.slice(0, 2048)
    df["fact"] = df["fact"].astype(str).str.slice(0, 2048)
    df["conclusion"] = df["conclusion"].astype(str).str.slice(0, 512)

    # Optional: concatenate fields for embedding
    df["text"] = df.apply(
        lambda row: f"{row['title']}\n\n{row['content']}\n\nFact: {row['fact']}\n\nConclusion: {row['conclusion']}",
        axis=1,
    )

    return df
