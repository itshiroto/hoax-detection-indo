from typing import List
from pydantic import BaseModel
from hoax_detect.config import settings
from hoax_detect.models import HoaxChunk

SIMILARITY_THRESHOLD = 0.3

def connect_milvus():
    """Connect to Milvus vector database."""
    from pymilvus import connections
    connections.connect(
        alias="default",
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT
    )

def create_collection():
    """Create Milvus collection if it doesn't exist."""
    from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
    connect_milvus()
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="fact", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="conclusion", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    
    schema = CollectionSchema(fields, description="Hoax news embeddings")
    return Collection(settings.MILVUS_COLLECTION, schema)

def search_similar_chunks(query: str, top_k: int = 5, threshold: float = SIMILARITY_THRESHOLD) -> List[HoaxChunk]:
    """Search for similar hoax chunks in vector database."""
    from pymilvus import Collection
    from hoax_detect.services.embedding import embed_text
    
    collection = Collection(settings.MILVUS_COLLECTION)  
    collection.load()
    
    query_embedding = embed_text(query)
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 16}
    }
    
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["title", "content", "fact", "conclusion"]
    )
    
    return [
        HoaxChunk(
            title=hit.entity.get("title"),
            content=hit.entity.get("content"),
            fact=hit.entity.get("fact"),
            conclusion=hit.entity.get("conclusion"),
        )
        for hit in results[0] 
        if hit.score >= threshold
    ]
