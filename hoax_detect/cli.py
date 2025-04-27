import argparse
import requests
from typing import Optional
from hoax_detect.config import settings
from hoax_detect.models import FactCheckRequest, FactCheckResponse


def fact_check(
    query: str, use_vector_db: bool = True, use_tavily: bool = True
) -> Optional[FactCheckResponse]:
    """Call the fact checking API endpoint."""
    try:
        response = requests.post(
            settings.FACTCHECK_API_URL_CLI,
            json=FactCheckRequest(
                query=query, use_vector_db=use_vector_db, use_tavily=use_tavily
            ).model_dump(),
            timeout=120,
        )
        response.raise_for_status()
        return FactCheckResponse(**response.json())
    except Exception as e:
        print(f"API error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Hoax News Fact Checking CLI")
    parser.add_argument("query", help="News title or content to check")
    parser.add_argument(
        "--no-vector-db", action="store_true", help="Skip vector database search"
    )
    parser.add_argument(
        "--no-tavily", action="store_true", help="Skip Tavily web search"
    )
    args = parser.parse_args()

    result = fact_check(
        query=args.query,
        use_vector_db=not args.no_vector_db,
        use_tavily=not args.no_tavily,
    )

    if result:
        print("\n=== Fact Check Result ===")
        print(f"Verdict: {result.verdict}")
        print(f"\nExplanation:\n{result.explanation}")
        if result.sources:
            print("\nSources:")
            for source in result.sources:
                print(f"- {source}")
    else:
        print("Failed to get fact check result")


if __name__ == "__main__":
    main()
