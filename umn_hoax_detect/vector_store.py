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
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="fact", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="conclusion", dtype=DataType.VARCHAR, max_length=512),
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

    all_embeddings = []
    titles = []
    contents = []
    facts = []
    conclusions = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding rows"):
        # Use full content before truncation
        full_text = f"{row['title']}\n\n{row['content']}\n\nFact: {row['fact']}\n\nConclusion: {row['conclusion']}"
        chunks = splitter.split_text(full_text)

        for chunk in chunks:
            emb = embed_text(chunk)
            all_embeddings.append(emb)
            # Store truncated metadata for Milvus VARCHAR limits
            titles.append(str(row['title'])[:512])
            contents.append(str(row['content'])[:2048])
            facts.append(str(row['fact'])[:2048])
            conclusions.append(str(row['conclusion'])[:512])

    data = [
        all_embeddings,
        titles,
        contents,
        facts,
        conclusions,
    ]

    collection.insert(data)
    collection.flush()
