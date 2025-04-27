from typing import List
from pymilvus import (
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    connections,
    utility,
)
from hoax_detect.config import settings
from hoax_detect.models import HoaxChunk
from hoax_detect.services.embedding import embed_text
from tqdm import tqdm

SIMILARITY_THRESHOLD = 0.3


def connect_milvus():
    """Connect to Milvus vector database."""
    connections.connect(
        alias="default",
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
        token=settings.MILVUS_TOKEN,
    )


def create_collection() -> Collection:
    """Create or return existing Milvus collection."""
    connect_milvus()

    if utility.has_collection(settings.MILVUS_COLLECTION):
        return Collection(settings.MILVUS_COLLECTION)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="fact", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="conclusion", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="references", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    ]

    schema = CollectionSchema(fields, description="Hoax news embeddings")
    collection = Collection(settings.MILVUS_COLLECTION, schema)

    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index("embedding", index_params)

    return collection


def insert_data(entities: List[List]) -> int:
    """Insert data into Milvus collection.

    Args:
        entities: List of entity lists in format:
            [
                [titles],
                [texts],
                [contents],
                [facts],
                [conclusions],
                [references],
                [embeddings]
            ]

    Returns:
        Number of inserted entities
    """
    collection = create_collection()
    insert_result = collection.insert(entities)
    collection.flush()
    return len(insert_result.primary_keys)


def batch_insert_data(df, batch_size: int = 32) -> int:
    """Batch insert dataframe into Milvus with progress tracking.

    Args:
        df: DataFrame containing columns: title, text, content, fact, conclusion, references
        batch_size: Number of records per batch

    Returns:
        Total number of inserted records
    """
    total_inserted = 0
    for i in tqdm(range(0, len(df), batch_size), desc="Inserting data"):
        batch = df.iloc[i : i + batch_size]
        entities = [
            batch.title.tolist(),
            batch.text.tolist(),
            batch.content.tolist(),
            batch.fact.tolist(),
            batch.conclusion.tolist(),
            batch.references.tolist(),
            [embed_text(row.text) for _, row in batch.iterrows()],
        ]
        total_inserted += insert_data(entities)
    return total_inserted


def search_similar_chunks(
    query: str, top_k: int = 5, threshold: float = SIMILARITY_THRESHOLD
) -> List[HoaxChunk]:
    """Search for similar hoax chunks in vector database."""
    from hoax_detect.services.embedding import embed_text

    connect_milvus()
    collection = Collection(settings.MILVUS_COLLECTION)
    collection.load()

    query_embedding = embed_text(query)
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["title", "text", "content", "fact", "conclusion", "references"],
    )

    return [
        HoaxChunk(
            title=hit.entity.get("title"),
            text=hit.entity.get("text"),
            content=hit.entity.get("content"),
            fact=hit.entity.get("fact"),
            conclusion=hit.entity.get("conclusion"),
            references=hit.entity.get("references"),
        )
        for hit in results[0]
        if hit.score >= threshold
    ]


def clear_collection():
    """Clear all data from the collection."""
    collection = create_collection()
    collection.drop()
    return create_collection()
