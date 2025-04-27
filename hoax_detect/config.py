import os
from typing import List
from pydantic import BaseSettings

class Settings(BaseSettings):
    DATASET_PATH: str = os.getenv("DATASET_PATH", "hoax_1k.csv")
    COLUMNS: List[str] = ["title", "content", "fact", "conclusion"]
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY")
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost") 
    MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "hoax_embeddings")
    FACTCHECK_API_URL: str = os.getenv("FACTCHECK_API_URL", "http://localhost:8000/fact_check")

    class Config:
        env_file = ".env"

settings = Settings()
