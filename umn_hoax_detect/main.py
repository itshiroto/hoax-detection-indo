import argparse
from umn_hoax_detect.data.loader import load_dataset
from umn_hoax_detect.vector_store import insert_embeddings, search_similar_chunks

import requests
import os


def call_openrouter(prompt, model="google/gemma-3-27b-it", max_tokens=512):
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


def build_prompt(user_query, retrieved_chunks, tavily_results=None):
    """
    Build a prompt for the LLM using the user query, retrieved context,
    and (optionally) Tavily news search results.
    """
    context = "Database Results:\n\n"
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
    tavily_context = ""
    if tavily_results:
        tavily_context += "\n\nWeb Search Results:"
        for i, res in enumerate(tavily_results, 1):
            tavily_context += (
                f"\nResult {i}:\n"
                f"Title: {res.get('title')}\n"
                f"URL: {res.get('url')}\n"
                f"Content: {res.get('content')}\n"
                f"Score: {res.get('score')}\n"
            )
    prompt = (
        f"User Query:\n{user_query}\n"
        f"\nRetrieved Hoax Chunks (with sources):{context}\n"
        f"{tavily_context}\n"
        "Based on the above information and your own knowledge, determine if the user query is factual or a hoax. "
        "First analyze the provided sources, then apply your own knowledge if needed. "
        "Provide:\n"
        "1. A clear verdict (HOAX or FACT)\n"
        "2. A detailed explanation in Indonesian\n"
        "3. Sources used (if any)\n"
        "4. Whether your conclusion is based on provided sources, your knowledge, or both"
    )
    return prompt


def call_tavily_api(query, max_results=3):
    """
    Call the Tavily API to search for news articles related to the query.
    Returns a list of dicts with 'title', 'url', and 'snippet'.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print("TAVILY_API_KEY not set in environment.")
        return []

    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {tavily_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "num_results": max_results,
        "search_type": "news",
        "include_domains": [
            "turnbackhoax.id", "analisadaily.com", "arungmedia.com", "balipost.com", "batamnews.co.id",
            "batampos.co.id", "batamtoday.com", "berazam.com", "beritamanado.com", "beritapagi.co.id",
            "bisnis.com", "bisnispapua.net", "bola.com", "bunaken.co.id", "bumntrack.com",
            "cumi-cumi.com", "detikkawanua.com", "detiknews.com", "detiksumsel.com", "djakalodang.co.id",
            "dream.co.id", "elshinta.com", "equator-news.com", "femina.co.id", "gatra.com",
            "goriau.com", "halloriau.com", "haluanriaupress.com", "harianhaluan.com", "harianjogja.com",
            "indopos.co.id", "indshangbao.com", "inforiau.com", "jamberita.com", "jambiindependent.co.id",
            "jawapos.com", "jektv.co.id", "jpnn.com", "kompas.com", "kontan.co.id",
            "koran-jakarta.com", "koranmerapi.com", "kr.co.id", "kronline.co", "lampungpost.com",
            "liputan6.com", "malang-post.com", "manadonews.co.id", "mediaindonesia.com", "metrojambi.com",
            "metroriau.com", "metrotvnews.com", "neraca.co.id", "netralnews.com", "newshunter.com",
            "palpres.com", "pekanbarumx.net", "pikiran-rakyat.com", "pontianakpost.com", "radarjogja.co.id",
            "radarlampung.co.id", "radarmalang.id", "radarpekanbaru.com", "radarsemarang.com", "rakyatmerdeka.co.id",
            "riaumandiri.co", "riaupos.com", "riaupotenza.com", "riauterkini.com", "sbo.co.id",
            "sinarharapan.co", "sindonews.com", "siwalimanews.com", "solopos.com", "suaramerdeka.com",
            "suarapembaruan.com", "sulutdaily.com", "sulutnews.com", "sumeks.co.id", "swa.co.id",
            "tabloidjubi.com", "tabloidsinartani.com", "tabloidwanitaindonesia.co.id", "telegrafnews.co.id", "tempo.co",
            "thejakartapost.com", "timesindonesia.co.id", "tirto.id", "transtv.co.id", "tribunjambi.com",
            "tribunjateng.com", "tribunnews.com", "trubus-online.co.id", "ummi-online.com", "uzone.co.id",
            "www.arah.com", "www.facebook.com/groups/fafhh/", "www.facebook.com/IndoHoaxBuster/", "www.facebook.com/sekoci.indo/", "www.fajar.co.id",
            "www.hariansinggalang.co.id", "www.indosiar.com", "www.kaltimpos.co.id", "www.kbr.id", "www.koran-sindo.com",
            "www.mncgroup.com", "www.okezone.com", "www.padangekspres.co.id", "www.republika.co.id", "www.rmol.co",
            "www.sctv.co.id", "www.viva.co.id"
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        print(f"Tavily API error: {e}")
        return []


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
    parser.add_argument(
        "--tavily-only",
        action="store_true",
        help="Skip vector DB and only use Tavily news search",
    )
    args = parser.parse_args()

    if not args.skip_embedding and not args.tavily_only:
        df = load_dataset()
        print(f"Loaded {len(df)} hoax entries.")
        print(df.head(3))

        print("Embedding and inserting into Milvus...")
        insert_embeddings(df)
        print("Done.")

    print("\n=== Retrieval Demo ===")
    query = input("Enter a news title or content to check for similar hoax chunks: ")

    if args.tavily_only:
        tavily_results = call_tavily_api(query, max_results=5)
        if not tavily_results:
            print("No results or error from Tavily API.")
        else:
            print("\nTavily News Search Results:")
            for i, res in enumerate(tavily_results, 1):
                print(f"\nResult {i}:")
                print(f"Title: {res.get('title')}")
                print(f"URL: {res.get('url')}")
                print(f"Snippet: {res.get('snippet')}")
        # Optionally, call LLM with only Tavily results
        if not args.no_llm:
            prompt = build_prompt(query, [], tavily_results)
            print("\n=== LLM Verdict & Explanation ===")
            verdict = call_openrouter(prompt)
            print(verdict if verdict else "No response from LLM.")
        return

    # Always use both vector db and Tavily for robust fact checking
    results = search_similar_chunks(query, top_k=5)
    print("\nTop similar chunks:")
    for i, hit in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Score: {hit['score']:.4f}")
        print(f"Title: {hit['title']}")
        print(f"Content: {hit['content'][:200]}...")  # Show first 200 chars
        print(f"Fact: {hit['fact']}")
        print(f"Conclusion: {hit['conclusion']}")

    tavily_results = call_tavily_api(query, max_results=5)
    if tavily_results:
        print("\nTavily News Search Results:")
        for i, res in enumerate(tavily_results, 1):
            print(f"\nResult {i}:")
            print(f"Title: {res.get('title')}")
            print(f"URL: {res.get('url')}")
            print(f"Content: {res.get('content')}")
            print(f"Score: {res.get('score')}")

    if not args.no_llm:
        prompt = build_prompt(query, results, tavily_results)
        print("\n=== LLM Verdict & Explanation ===")
        verdict = call_openrouter(prompt)
        print(verdict if verdict else "No response from LLM.")


if __name__ == "__main__":
    main()
