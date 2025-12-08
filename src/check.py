from pymilvus import Collection
import json
import glob
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import ZILLIZ_URI, ZILLIZ_API_KEY
from pymilvus import connections

print("Connecting to Zilliz Cloud...")
try:
    connections.connect(
        alias="default",
        uri=ZILLIZ_URI,
        token=ZILLIZ_API_KEY
    )
    print("Connected.")
except Exception as e:
    print("Zilliz connection failed:", e)

print("=== CHECK 1: Zilliz Collection Vector Dimensions ===")
try:
    col = Collection("news_articles")
    col.load()
    print(col.schema)
except Exception as e:
    print("Error reading schema:", e)

print("\n=== CHECK 2: Dataset Sample Fields (PKL) ===")
import os
pkl_paths = [
    "data/tsfm_dataset/train.pkl",
    "data/tsfm_dataset/val.pkl",
    "data/tsfm_dataset/test.pkl",
]

pkl_file = None
for p in pkl_paths:
    if os.path.exists(p):
        pkl_file = p
        break

if pkl_file is None:
    print("No PKL dataset found.")
else:
    print(f"Reading PKL dataset: {pkl_file}")
    try:
        import pickle
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        print("Top-level type:", type(data))

        # Prepare a sample record
        sample = None
        if isinstance(data, dict):
            for v in data.values():
                sample = v[0] if isinstance(v, list) and len(v) > 0 else v
                break
        elif isinstance(data, list) and len(data) > 0:
            sample = data[0]
        else:
            print("Dataset is empty.")
            sample = None

        if sample is not None:
            print("Sample record type:", type(sample))
            if hasattr(sample, "keys"):
                print("Sample keys:", list(sample.keys()))

                # Check embedding-related fields
                embed_fields = ["embedding", "embeddings", "neighbor_embeddings",
                                "event_emb", "rag_emb", "vector"]
                found = [f for f in embed_fields if f in sample]

                if len(found) > 0:
                    print(f"\n⚠️ WARNING: Dataset contains embedding fields: {found} → dataset MUST be rebuilt.")
                else:
                    print("\n✅ Dataset does NOT contain embedding fields → dataset does NOT need rebuild.")
            else:
                print("Sample is not dict-like, cannot show keys.")

    except Exception as e:
        print("Error reading PKL dataset:", e)

print("\n=== CHECK COMPLETE ===")

from pymilvus import Collection

col = Collection("earnings_transcripts")
col.load()
print(col.schema)