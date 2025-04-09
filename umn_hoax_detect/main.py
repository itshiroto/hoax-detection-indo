from umn_hoax_detect.data.loader import load_dataset
from umn_hoax_detect.vector_store import insert_embeddings


def main():
    df = load_dataset()
    print(f"Loaded {len(df)} hoax entries.")
    print(df.head(3))

    print("Embedding and inserting into Milvus...")
    insert_embeddings(df)
    print("Done.")


if __name__ == "__main__":
    main()
