import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATASET_PATH: str = os.getenv("DATASET_PATH", "hoax_1k.csv")
    COLUMNS: List[str] = ["title", "content", "fact", "conclusion", "references"]
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY")
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
    MILVUS_TOKEN: str = os.getenv("MILVUS_TOKEN", "")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "hoax_embeddings")
    FACTCHECK_API_URL: str = os.getenv(
        "FACTCHECK_API_URL", "http://localhost:8000/fact_check"
    )
    FACTCHECK_API_URL_CLI: str = os.getenv(
        "FACTCHECK_API_URL", "http://localhost:8000/fact_check?verbose=true"
    )

    class Config:
        env_file = ".env"


settings = Settings()
