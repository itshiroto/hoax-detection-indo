from pydantic import BaseModel
from typing import List, Optional

class FactCheckRequest(BaseModel):
    query: str
    use_vector_db: bool = True 
    use_tavily: bool = True

class FactCheckResponse(BaseModel):
    verdict: str
    explanation: str
    sources: List[str] = []

class NewsResult(BaseModel):
    title: str
    url: str 
    content: str
    score: float

class HoaxChunk(BaseModel):
    title: str
    content: str
    fact: str
    conclusion: str
    references: List[str] = []
    embedding: Optional[List[float]] = None
