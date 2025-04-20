from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    IndexType,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from umn_hoax_detect.config import (
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_COLLECTION,
)

from tqdm import tqdm

# Load the IndoBERT model once
model = SentenceTransformer("LazarusNLP/all-indobert-base-v4")


SIMILARITY_THRESHOLD = 0.3  # Only keep results with cosine similarity >= 0.6


def search_similar_chunks(query, top_k=5, threshold=SIMILARITY_THRESHOLD):
    """
    Given a query string, embed it, search Milvus for top_k most similar chunks,
    and return the metadata for those chunks, filtered by similarity threshold.
    """
    from pymilvus import Collection

    connect_milvus()
    collection = Collection(MILVUS_COLLECTION)
    collection.load()

    # Embed the query
    query_embedding = embed_text(query)

    # Perform vector search
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    }
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["title", "content", "text", "fact", "conclusion"],
    )

    # Parse and filter results by similarity threshold
    hits = results[0]
    filtered_hits = [
        {
            "score": hit.distance,
            "title": hit.entity.get("title"),
            "content": hit.entity.get("content"),
            "text": hit.entity.get("text"),
            "fact": hit.entity.get("fact"),
            "conclusion": hit.entity.get("conclusion"),
        }
        for hit in hits
        if 1 - hit.distance >= threshold  # cosine similarity = 1 - distance
    ]
    return filtered_hits


def embed_text(text: str) -> list[float]:
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.tolist()


def connect_milvus():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)


def create_collection():
    if utility.has_collection(MILVUS_COLLECTION):
        collection = Collection(MILVUS_COLLECTION)
        # Check if index exists, if not, create it
        if not collection.has_index():
            collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 128},
                },
            )
        collection.load()
        return collection

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="fact", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="conclusion", dtype=DataType.VARCHAR, max_length=4096),
    ]
    schema = CollectionSchema(fields, description="Hoax detection embeddings")
    collection = Collection(name=MILVUS_COLLECTION, schema=schema)

    # Create index on embedding field
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        },
    )

    collection.load()
    return collection


def insert_embeddings(df):
    connect_milvus()
    collection = create_collection()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # Prepare lists for batch insert
    embeddings = []
    titles = []
    contents = []
    texts = []
    facts = []
    conclusions = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding rows"):
        # Use full concatenated text before truncation
        chunks = splitter.split_text(row["text"])

        for chunk in chunks:
            emb = embed_text(chunk)
            embeddings.append(emb)
            titles.append(str(row["title"])[:512])
            # Save original hoax content (truncated) as 'content' metadata
            contents.append(str(row["content"])[:2048])
            # Save chunk text (truncated) as 'text' metadata
            texts.append(chunk[:2048])
            facts.append(str(row["fact"])[:2048])
            conclusions.append(str(row["conclusion"])[:512])

    data = [
        embeddings,
        titles,
        contents,
        texts,
        facts,
        conclusions,
    ]

    collection.insert(data)
    collection.flush()
