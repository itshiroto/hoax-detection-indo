from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from hoax_detect.services import (
    search_similar_chunks,
    call_tavily_api,
    call_openrouter,
    build_prompt
)
from hoax_detect.models import FactCheckRequest, FactCheckResponse, NewsResult, HoaxChunk
from hoax_detect.config import settings

app = FastAPI(
    title="Hoax News Fact Checking API",
    description="API for fact checking Indonesian news using RAG with Milvus and Tavily.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/fact_check", response_model=FactCheckResponse)
async def fact_check(request: FactCheckRequest) -> FactCheckResponse:
    """Main fact checking endpoint."""
    try:
        # Retrieve relevant context
        chunks: List[HoaxChunk] = []
        if request.use_vector_db:
            chunks = search_similar_chunks(request.query)
        
        web_results: List[NewsResult] = []
        if request.use_tavily:
            web_results = call_tavily_api(request.query)

        # Generate LLM response
        prompt = build_prompt(request.query, chunks, web_results)
        llm_response = call_openrouter(prompt)
        
        if not llm_response:
            raise HTTPException(status_code=500, detail="LLM service error")

        # Parse response (simplified - would need proper parsing in real implementation)
        return FactCheckResponse(
            verdict="HOAX" if "HOAX" in llm_response else "FACT",
            explanation=llm_response,
            sources=[res.url for res in web_results] if web_results else []
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
