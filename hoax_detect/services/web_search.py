import json
import os
import requests
from pathlib import Path
from typing import List
from hoax_detect.models import NewsResult


def load_trusted_domains() -> List[str]:
    """Load the list of trusted domains from JSON file."""
    config_path = Path(__file__).parent.parent / "config" / "trusted_domains.json"
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading trusted domains: {e}")


def call_tavily_api(query: str, max_results: int = 3) -> List[NewsResult]:
    """Call the Tavily API to search for news articles."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not set in environment")

    trusted_domains = load_trusted_domains()
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "num_results": max_results,
        "search_type": "news",
        "include_domains": trusted_domains if trusted_domains else None,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return [
            NewsResult(
                title=res.get("title"),
                url=res.get("url"),
                content=res.get("content"),
                score=res.get("score", 0),
            )
            for res in data.get("results", [])
        ]
    except Exception as e:
        raise RuntimeError(f"Tavily API error: {e}")
