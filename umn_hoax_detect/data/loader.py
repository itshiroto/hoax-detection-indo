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

    # Create embedding text from full content BEFORE truncation
    df["text"] = df.apply(
        lambda row: f"{row['title']}\n\n{row['content']}\n\nFact: {row['fact']}\n\nConclusion: {row['conclusion']}",
        axis=1,
    )

    # Truncate metadata fields for Milvus VARCHAR limits
    df["title"] = df["title"].astype(str).str.slice(0, 512)
    df["content"] = df["content"].astype(str).str.slice(0, 2048)
    df["fact"] = df["fact"].astype(str).str.slice(0, 2048)
    df["conclusion"] = df["conclusion"].astype(str).str.slice(0, 512)

    return df
