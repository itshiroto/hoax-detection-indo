import argparse
from umn_hoax_detect.data.loader import load_dataset
from umn_hoax_detect.vector_store import insert_embeddings, search_similar_chunks


def main():
    parser = argparse.ArgumentParser(description="Hoax News RAG System")
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding/insertion and go straight to searching",
    )
    args = parser.parse_args()

    if not args.skip_embedding:
        df = load_dataset()
        print(f"Loaded {len(df)} hoax entries.")
        print(df.head(3))

        print("Embedding and inserting into Milvus...")
        insert_embeddings(df)
        print("Done.")

    # --- Retrieval demo ---
    print("\n=== Retrieval Demo ===")
    query = input("Enter a news title or content to check for similar hoax chunks: ")
    results = search_similar_chunks(query, top_k=5)
    print("\nTop similar chunks:")
    for i, hit in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Score: {hit['score']:.4f}")
        print(f"Title: {hit['title']}")
        print(f"Content: {hit['content'][:200]}...")  # Show first 200 chars
        print(f"Fact: {hit['fact']}")
        print(f"Conclusion: {hit['conclusion']}")


if __name__ == "__main__":
    main()
