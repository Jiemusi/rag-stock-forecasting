from pymilvus import connections, Collection
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

ZILLIZ_URI = config.ZILLIZ_URI
ZILLIZ_API_KEY = config.ZILLIZ_API_KEY
COLLECTION_NAME = config.COLLECTION_NAME

# Connect to Zilliz
connections.connect(
    alias="default",
    uri=ZILLIZ_URI,
    token=ZILLIZ_API_KEY
)
print("Connected to Zilliz")

# Load collection
collection = Collection(COLLECTION_NAME)
collection.load()
print(f"Collection '{COLLECTION_NAME}' loaded")

# Search with a random vector (replace with a real embedding later)
query_vec = np.random.rand(4).tolist()

results = collection.search(
    data=[query_vec],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=5,
    output_fields=["ticker", "forward_return"]
)

# Print results
print("\nSearch Results:")
for hit in results[0]:
    print(
        f"{hit.entity.get('ticker'):<6} "
        f"| similarity={hit.distance:.4f} "
        f"| return={hit.entity.get('forward_return')}"
    )
