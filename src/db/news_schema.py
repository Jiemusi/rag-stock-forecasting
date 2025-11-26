import os
from pymilvus import MilvusClient

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")

if not ZILLIZ_URI or not ZILLIZ_API_KEY:
    raise ValueError("ZILLIZ_URI or ZILLIZ_API_KEY not set.")


def get_news_collection(collection_name="news_articles"):
    """
    Create a news_articles collection identical in structure to the earnings collection.
    - Single vector field named 'vector'
    - Dimension 3072 (matching text-embedding-3-large)
    - COSINE metric
    - auto_id primary key
    - dynamic fields allowed
    """

    client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_API_KEY)

    if not client.has_collection(collection_name):
        client.create_collection(
            collection_name=collection_name,
            dimension=3072,         # same embedding size
            metric_type="COSINE",
            auto_id=True            # allows dynamic fields
        )
        print(f"Created new collection: {collection_name}")

    return client