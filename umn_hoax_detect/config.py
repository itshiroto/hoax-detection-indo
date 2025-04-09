import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Path to your Mafindo CSV dataset
DATASET_PATH = os.getenv("DATASET_PATH", "hoax_1k.csv")

# Columns to extract
COLUMNS = ["title", "content", "fact", "conclusion"]

# OpenRouter API key (optional if using OpenRouter)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Milvus connection settings
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "hoax_embeddings")
