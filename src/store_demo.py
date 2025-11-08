from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection
)
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

ZILLIZ_URI = config.ZILLIZ_URI
ZILLIZ_API_KEY = config.ZILLIZ_API_KEY
COLLECTION_NAME = config.COLLECTION_NAME


# Connect to databse
connections.connect(
    alias="default",
    uri=ZILLIZ_URI,
    token=ZILLIZ_API_KEY
)
print("Connected to Zilliz")

# Define collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="ticker", dtype=DataType.VARCHAR, max_length=12),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4),
    FieldSchema(name="forward_return", dtype=DataType.FLOAT)
]
schema = CollectionSchema(fields, description="Demo collection for RAG stock forecasting")

# Create collection if not exists
collection = Collection(name=COLLECTION_NAME, schema=schema)
print(f"Collection '{COLLECTION_NAME}' created")

# Insert test data
tickers = ["AAPL", "NVDA", "MSFT"]
embeddings = [np.random.rand(4).tolist() for _ in range(3)]
returns = [0.02, -0.015, 0.03]

data = [tickers, embeddings, returns]
collection.insert(data)
print("Data successfully inserted")

# Build index
collection.create_index(
    field_name="embedding",
    index_params={
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 64}
    }
)
collection.load()
print("Index built and collection loaded")