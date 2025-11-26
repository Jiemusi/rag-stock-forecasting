import os
from pymilvus import MilvusClient

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")

if not ZILLIZ_URI or not ZILLIZ_API_KEY:
    raise ValueError("ZILLIZ_URI or ZILLIZ_API_KEY not set.")


def get_earnings_collection(collection_name="earnings_transcripts"):
    client = MilvusClient(
        uri=ZILLIZ_URI,
        token=ZILLIZ_API_KEY
    )

    # If collection does NOT exist → create with only vector field
    if not client.has_collection(collection_name):
        client.create_collection(
            collection_name=collection_name,
            dimension=3072,
            metric_type="COSINE",
            auto_id=True
        )

        print(f"Created new collection: {collection_name}")

    return client