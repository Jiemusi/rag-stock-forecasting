import os
import glob
import uuid
import pandas as pd
from pymilvus import MilvusClient

from embedding.embed import embed_batch
from embedding.chunking import chunk_text
from db.news_schema import get_news_collection

NEWS_DIR = "data/news"
COLLECTION_NAME = "news_articles"
BATCH_SIZE = 64

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")

client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_API_KEY)

# def ensure_index(collection_name: str, field_name: str = "vector"):
#     """Create index for MilvusClient collections."""

#     try:
#         existing = client.list_indexes(collection_name)
#     except Exception:
#         existing = []

#     if existing:
#         print(f"Index already exists for {collection_name}: {existing}")
#         return

#     print(f"Creating AUTOINDEX for {collection_name} ...")

#     index_params = {
#         "index_type": "AUTOINDEX",
#         "metric_type": "COSINE",
#         "params": {}
#     }

#     client.create_index(
#         collection_name=collection_name,
#         field_name=field_name,
#         index_params=index_params
#     )
#     print("Index created successfully.")


def flush_batch(rows_batch):
    """
    Embed and insert a batch of rows into Milvus.
    rows_batch: list[dict], each dict has 'text' plus all metadata fields
    """
    if not rows_batch:
        return

    texts = [row["text"] for row in rows_batch]
    vectors = embed_batch(texts)

    for row, vec in zip(rows_batch, vectors):
        row["vector"] = vec

    # Insert all rows in one call
    result = client.insert(
        collection_name=COLLECTION_NAME,
        data=rows_batch
    )
    print("Milvus ACK:", result)


def process_csv(path: str):
    df = pd.read_csv(path)

    if df.empty:
        print(f"Skipping empty file: {path}")
        return

    symbol = str(df["company"].iloc[0])
    print(f"\nProcessing news for {symbol}: {path}  (rows={len(df)})")

    batch = []
    total_chunks = 0

    for _, row in df.iterrows():
        date = str(row.get("published_date", ""))
        title = str(row.get("title", ""))
        description = str(row.get("description", ""))
        content = str(row.get("content", ""))
        url = str(row.get("url", ""))
        source = str(row.get("source", ""))
        topic = str(row.get("topic", ""))

        # Combine text fields for embedding
        full_text = f"{title}\n{description}\n{content}".strip()
        if not full_text:
            continue  # skip empty articles

        chunks = chunk_text(full_text)
        doc_id = str(uuid.uuid4())

        for idx, chunk in enumerate(chunks):
            row_dict = {
                "symbol": symbol,
                "date": date,
                "doc_id": doc_id,
                "chunk_id": idx,
                "text": chunk,
                # dynamic fields (auto schema)
                "url": url,
                "source": source,
                "topic": topic,
                "title": title,
            }
            batch.append(row_dict)
            total_chunks += 1

            if len(batch) >= BATCH_SIZE:
                flush_batch(batch)
                print(f"  → inserted {len(batch)} chunks (running total: {total_chunks})")
                batch = []

    # Flush remaining chunks from this file
    if batch:
        flush_batch(batch)
        print(f"  → inserted final {len(batch)} chunks (total: {total_chunks})")

    print(f"Done file {path} → {total_chunks} chunks for {symbol}")


def main():
    # Create collection if needed
    if not client.has_collection(COLLECTION_NAME):
        get_news_collection(COLLECTION_NAME)

    # ensure_index(COLLECTION_NAME, field_name="vector")

    csv_files = glob.glob(f"{NEWS_DIR}/**/*.csv", recursive=True)
    print(f"Found {len(csv_files)} CSV files under {NEWS_DIR}")

    for file in csv_files:
        process_csv(file)


if __name__ == "__main__":
    main()
