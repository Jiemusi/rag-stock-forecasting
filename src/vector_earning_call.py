import os
import json
import uuid
from pymilvus import MilvusClient, connections
from embedding.chunking import chunk_text
from embedding.embed import embed_text
from db.earnings_schema import get_earnings_collection

TRANSCRIPT_DIR = "data/earning_calls"
COLLECTION_NAME = "earnings_transcripts"

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")

client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_API_KEY)

def load_json(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)

def process_company(symbol):
    path = f"{TRANSCRIPT_DIR}/{symbol}.json"
    if not os.path.exists(path):
        print(f"Missing: {path}")
        return
    
    data = load_json(path)

    # JSON is a list of records
    if not isinstance(data, list):
        print(f"Unexpected format in {path}")
        return

    for record in data:
        try:
            quarter = record["quarter"]        # e.g. "Q1"
            transcript_list = record["data"]["transcript"]
        except KeyError as e:
            print(f"Missing key in {symbol}: {e}")
            continue

        # Build raw text from transcript
        full_text = "\n".join(
            f"{entry['speaker']}: {entry['content']}"
            for entry in transcript_list
        )

        chunks = chunk_text(full_text)

        # Upload each chunk
        for idx, chunk in enumerate(chunks):
            emb = embed_text(chunk)

            unique = str(uuid.uuid4())
            doc_id = f"{symbol}_{quarter}"

            client.insert(
                collection_name=COLLECTION_NAME,
                data={
                    "symbol": symbol,
                    "quarter": quarter,
                    "chunk_id": idx,
                    "doc_id": doc_id,
                    "text": chunk,
                    "vector": emb
                }
            )

        print(f"{symbol} {quarter}: {len(chunks)} chunks uploaded.")


def main():
    # Create collection if not exists
    if not client.has_collection(COLLECTION_NAME):
        get_earnings_collection(COLLECTION_NAME)
    
    symbols = [f.replace(".json", "") for f in os.listdir(TRANSCRIPT_DIR)]
    
    for symbol in symbols:
        print(f"Processing {symbol}...")
        process_company(symbol)


if __name__ == "__main__":
    main()
