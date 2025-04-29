import pandas as pd
from hoax_detect.config import settings
from hoax_detect.services.vector_store import (
    connect_milvus,
    create_collection,
    batch_insert_data,
    clear_collection,
)


def load_dataset() -> pd.DataFrame:
    """Load and validate the dataset from configured CSV path."""
    try:
        df = pd.read_csv(settings.DATASET_PATH)

        # Validate required columns
        required_columns = set(settings.COLUMNS)
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Dataset missing required columns: {missing}")

        # Clean data
        df = df.dropna(subset=settings.COLUMNS)
        df = df[settings.COLUMNS]

        # Create combined text for embedding (content + fact)
        df["text"] = df["content"] + "\n\n" + df["fact"]
        df["text"] = df["text"].str.slice(0, 8192)
        df["content"] = df["content"].str.slice(0, 4096)
        df["fact"] = df["fact"].str.slice(0, 2048)
        df["conclusion"] = df["conclusion"].str.slice(0, 2048)

        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


def initialize_vector_db(clear_existing: bool = False) -> None:
    """Initialize the vector database with dataset embeddings."""
    try:
        connect_milvus()

        if clear_existing:
            print("Clearing existing collection...")
            clear_collection()

        collection = create_collection()
        print(f"Using collection: {collection.name}")

        print("Loading dataset...")
        df = load_dataset()
        print(f"Found {len(df)} valid records")

        print("Inserting embeddings...")
        inserted_count = batch_insert_data(df)
        print(f"Successfully inserted {inserted_count} records")

    except Exception as e:
        print(f"Initialization failed: {e}")
        raise


def main():
    """Standalone data loading entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Initialize vector database with hoax data"
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing collection before loading"
    )
    args = parser.parse_args()

    try:
        initialize_vector_db(clear_existing=args.clear)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
