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
    Process dataframe, chunk text, embed, and insert into Milvus in batches.
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
    references = []

    batch_size = 1000
    inserted_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows for embedding"):
        chunks = splitter.split_text(row["text"])

        for chunk in chunks:
            emb = embed_text(chunk)
            embeddings.append(emb)
            titles.append(str(row["title"])[:512])
            contents.append(str(row["content"])[:2048])
            texts.append(chunk[:2048])
            facts.append(str(row["fact"])[:2048])
            conclusions.append(str(row["conclusion"])[:512])
            references.append(str(row.get("references", ""))[:2048])

            # Check if batch size is reached
            if len(embeddings) == batch_size:
                data = [
                    embeddings,
                    titles,
                    contents,
                    texts,
                    facts,
                    conclusions,
                    references,
                ]
                collection.insert(data)
                inserted_count += len(embeddings)
                print(f"Inserted batch of {len(embeddings)} chunks. Total inserted: {inserted_count}")

                # Clear lists for the next batch
                embeddings = []
                titles = []
                contents = []
                texts = []
                facts = []
                conclusions = []
                references = []

    # Insert any remaining data in the last batch
    if embeddings:
        data = [
            embeddings,
            titles,
            contents,
            texts,
            facts,
            conclusions,
            references,
        ]
        collection.insert(data)
        inserted_count += len(embeddings)
        print(f"Inserted final batch of {len(embeddings)} chunks. Total inserted: {inserted_count}")


    collection.flush()
    collection.release()  # Release collection from memory
    print(f"Finished inserting all chunks into Milvus. Total inserted: {inserted_count}")

