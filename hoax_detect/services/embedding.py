from sentence_transformers import SentenceTransformer
from typing import List

model = SentenceTransformer("LazarusNLP/all-indobert-base-v4")

def embed_text(text: str) -> List[float]:
    """Convert text to embedding vector."""
    return model.encode(text).tolist()
