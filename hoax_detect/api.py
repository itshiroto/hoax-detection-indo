from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from hoax_detect.services import (
    search_similar_chunks,
    call_tavily_api,
    call_openrouter,
    build_prompt,
)
from hoax_detect.models import (
    FactCheckRequest,
    FactCheckResponse,
    NewsResult,
    HoaxChunk,
)
from pydantic import BaseModel
import logging
import sys

load_dotenv()

app = FastAPI(
    title="Hoax News Fact Checking API",
    description="API for fact checking Indonesian news using RAG with Milvus and Tavily.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


class BatchFactCheckRequest(BaseModel):
    queries: List[str]
    use_vector_db: bool = True
    use_tavily: bool = True
    verbose: bool = False


@app.post("/fact_check", response_model=FactCheckResponse)
async def fact_check(
    request: FactCheckRequest, verbose: bool = False
) -> FactCheckResponse:
    """Main fact checking endpoint."""
    try:
        chunks, web_results = await _retrieve_context(request)

        prompt = build_prompt(request.query, chunks, web_results)

        if verbose:
            logging.info("\n=== LLM PROMPT ===\n%s\n=== END PROMPT ===", prompt)

        llm_response = call_openrouter(prompt)

        if not llm_response:
            raise HTTPException(status_code=500, detail="LLM service error")

        response = _format_response(llm_response, web_results)
        return response

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_fact_check", response_model=List[FactCheckResponse])
async def batch_fact_check(request: BatchFactCheckRequest) -> List[FactCheckResponse]:
    """Batch process multiple fact checks."""
    results = []
    for query in request.queries:
        single_request = FactCheckRequest(
            query=query,
            use_vector_db=request.use_vector_db,
            use_tavily=request.use_tavily,
        )
        results.append(await fact_check(single_request, verbose=request.verbose))
    return results


async def _retrieve_context(
    request: FactCheckRequest,
) -> tuple[List[HoaxChunk], List[NewsResult]]:
    """Retrieve both vector DB chunks and web search results."""
    chunks: List[HoaxChunk] = []
    if request.use_vector_db:
        chunks = search_similar_chunks(request.query)

    web_results: List[NewsResult] = []
    if request.use_tavily:
        web_results = call_tavily_api(request.query)

    return chunks, web_results


def _format_response(
    llm_response: str, web_results: List[NewsResult]
) -> FactCheckResponse:
    """Format the LLM response into a structured FactCheckResponse."""
    verdict = (
        "HOAX"
        if "HOAX" in llm_response.upper()
        else "FACT"
        if "FACT" in llm_response.upper()
        else "UNCERTAIN"
    )
    return FactCheckResponse(
        verdict=verdict,
        explanation=llm_response,
        sources=[res.url for res in web_results] if web_results else [],
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": ["milvus_vector_db", "tavily_web_search", "openrouter_llm"],
        "version": "1.0.0",
    }
