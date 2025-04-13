import argparse
from umn_hoax_detect.data.loader import load_dataset
from umn_hoax_detect.vector_store import insert_embeddings, search_similar_chunks

import requests
import os


def call_openrouter(prompt, model="google/gemini-2.0-flash-001", max_tokens=512):
    """
    Call OpenRouter API with the constructed prompt and return the response.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set in environment.")
        return None

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant for hoax detection. Given a user query and retrieved context, provide a verdict and explanation.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"OpenRouter API error: {e}")
        return None


def build_prompt(user_query, retrieved_chunks):
    """
    Build a prompt for the LLM using the user query and retrieved context,
    and include sources for each chunk.
    """
    context = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        context += (
            f"\n---\n"
            f"Hoax Chunk {i}:\n"
            f"Title: {chunk['title']}\n"
            f"Content: {chunk['content']}\n"
            f"Fact: {chunk['fact']}\n"
            f"Conclusion: {chunk['conclusion']}\n"
            f"Sumber: {chunk['title']}\n"
        )
    prompt = (
        f"User Query:\n{user_query}\n"
        f"\nRetrieved Hoax Chunks (with sources):{context}\n"
        "Based on the above, is the user query a hoax or not? Please provide a verdict, a brief explanation in Indonesian, and list the sources (Sumber) you used."
    )
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Hoax News RAG System")
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding/insertion and go straight to searching",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM generation and only show retrieved chunks",
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

    if not args.no_llm:
        prompt = build_prompt(query, results)
        print("\n=== LLM Verdict & Explanation ===")
        verdict = call_openrouter(prompt)
        print(verdict if verdict else "No response from LLM.")


if __name__ == "__main__":
    main()
