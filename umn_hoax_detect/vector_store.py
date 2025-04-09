from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from umn_hoax_detect.config import (
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_COLLECTION,
)

# Load the IndoBERT model once
model = SentenceTransformer("LazarusNLP/all-indobert-base-v4")

def embed_text(text: str) -> list[float]:
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.tolist()

def connect_milvus():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

def create_collection():
    if utility.has_collection(MILVUS_COLLECTION):
        return Collection(MILVUS_COLLECTION)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="fact", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="conclusion", dtype=DataType.VARCHAR, max_length=512),
    ]
    schema = CollectionSchema(fields, description="Hoax detection embeddings")
    collection = Collection(name=MILVUS_COLLECTION, schema=schema)
    collection.load()
    return collection

def insert_embeddings(df):
    connect_milvus()
    collection = create_collection()

    embeddings = []
    for text in df["text"]:
        emb = embed_text(text)
        embeddings.append(emb)

    data = [
        embeddings,
        df["title"].tolist(),
        df["content"].tolist(),
        df["fact"].tolist(),
        df["conclusion"].tolist(),
    ]

    collection.insert(data)
    collection.flush()
