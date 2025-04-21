from sentence_transformers import SentenceTransformer
from pymilvus import Collection
from umn_hoax_detect.vector_store import connect_milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Load the IndoBERT model once
model = SentenceTransformer("LazarusNLP/all-indobert-base-v4")


def embed_text(text: str) -> list[float]:
    """Generate embeddings for text using the local model."""
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.tolist()


def insert_embeddings(df):
    """
    Process dataframe, chunk text, embed, and insert into Milvus.
    """
    connect_milvus()
    # Assuming create_collection is called elsewhere to ensure the collection exists
    collection = Collection("hoax_embeddings")  # Use the collection name directly
    collection.load()  # Load collection into memory for insertion

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
    references = []  # Added references list

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding rows"):
        # Use full concatenated text before truncation
        chunks = splitter.split_text(row["text"])

        for chunk in chunks:
            emb = embed_text(chunk)
            embeddings.append(emb)
            titles.append(str(row["title"])[:512])
            contents.append(str(row["content"])[:2048])
            texts.append(chunk[:2048])
            facts.append(str(row["fact"])[:2048])
            conclusions.append(str(row["conclusion"])[:512])
            references.append(str(row.get("references", ""))[:2048])  # Added references

    data = [
        embeddings,
        titles,
        contents,
        texts,
        facts,
        conclusions,
        references,  # Added references to data list
    ]

    collection.insert(data)
    collection.flush()
    collection.release()  # Release collection from memory
    print(f"Inserted {len(embeddings)} chunks into Milvus.")
