import os
import requests
from typing import List, Dict, Optional
from hoax_detect.models import HoaxChunk, NewsResult


def call_openrouter(
    prompt: str, model: str = "google/gemini-2.0-flash-lite-001", max_tokens: int = 1024
) -> Optional[str]:
    """Call OpenRouter API with the constructed prompt."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment")

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
        raise RuntimeError(f"OpenRouter API error: {e}")


def build_prompt(
    user_query: str,
    retrieved_chunks: List[HoaxChunk],
    tavily_results: List[NewsResult] = None,
) -> str:
    """Build a prompt for the LLM using the user query and retrieved context."""
    context = "Database Results:\n\n"
    for i, chunk in enumerate(retrieved_chunks, 1):
        context += (
            f"\n---\n"
            f"Title: {chunk.title}\n"
            f"Content: {chunk.content}\n"
            f"Fact: {chunk.fact}\n"
            f"Conclusion: {chunk.conclusion}\n"
        )

    if tavily_results:
        context += "\n\nWeb Search Results:"
        for i, res in enumerate(tavily_results, 1):
            context += (
                f"\nResult {i}:\n"
                f"Title: {res.title}\n"
                f"URL: {res.url}\n"
                f"Content: {res.content}\n"
                f"Score: {res.score}\n"
            )

    return (
        f"User Query:\n{user_query}\n"
        f"\nRetrieved Context:\n{context}\n"
        "Based on the above information, determine if the user query is factual or a hoax. "
        "Provide:\n"
        "1. A clear verdict (HOAKS or FAKTA)\n"
        "2. A detailed explanation in Indonesian\n"
        "3. Sources used (if any)\n"
    )
