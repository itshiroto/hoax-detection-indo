from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

from umn_hoax_detect.vector_store import search_similar_chunks
from umn_hoax_detect.main import call_openrouter, build_prompt, call_tavily_api

app = FastAPI(
    title="Hoax News Fact Checking API",
    description="API for fact checking Indonesian news using RAG with Milvus and Tavily.",
    version="1.0.0"
)

class FactCheckRequest(BaseModel):
    query: str
    use_tavily: Optional[bool] = True
    use_vector_db: Optional[bool] = True

class FactCheckResponse(BaseModel):
    verdict: str
    explanation: str
    sources: List[str]

@app.post("/fact_check", response_model=FactCheckResponse)
def fact_check(request: FactCheckRequest):
    # Retrieve from vector DB
    vector_results = []
    if request.use_vector_db:
        vector_results = search_similar_chunks(request.query, top_k=5)
    # Retrieve from Tavily
    tavily_results = []
    if request.use_tavily:
        tavily_results = call_tavily_api(request.query, max_results=5)

    # Build prompt and call LLM
    prompt = build_prompt(request.query, vector_results, tavily_results)
    llm_response = call_openrouter(prompt)
    if not llm_response:
        raise HTTPException(status_code=500, detail="No response from LLM.")

    # Extract sources from both vector DB and Tavily
    sources = []
    for chunk in vector_results:
        if chunk.get("title"):
            sources.append(chunk["title"])
    for res in tavily_results:
        if res.get("url"):
            sources.append(res["url"])

    # Try to split LLM response into verdict and explanation
    verdict = ""
    explanation = llm_response
    # Simple heuristic: look for "Verdict:" or "Putusan:" in the response
    for line in llm_response.splitlines():
        if "verdict" in line.lower() or "putusan" in line.lower():
            verdict = line.strip()
            explanation = llm_response.replace(line, "").strip()
            break

    return FactCheckResponse(
        verdict=verdict if verdict else "Tidak diketahui",
        explanation=explanation,
        sources=sources
    )
