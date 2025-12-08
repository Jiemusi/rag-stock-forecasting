from retrieve import multimodal_retrieve
from openai import OpenAI

client = OpenAI()

def embed_openai(text):
    if text is None or text.strip() == "":
        text = "no news available"
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    base = resp.data[0].embedding

    import numpy as np
    arr = np.array(base, dtype=np.float32)

    # Always return raw 3072‑dim embedding (Zilliz collections expect dim=3072)
    return arr
# emb = embed_openai("test")
# print("TEST EMBEDDING SHAPE:", len(emb))
# raise SystemExit

symbol = "AAPL"
date = "2023-08-01"

# Correct way to get embedding:
dummy_emb = embed_openai("test news")
print("FINAL QUERY EMB SHAPE:", dummy_emb.shape)
results = multimodal_retrieve(
    query_text="test news",
    query_emb=dummy_emb,
    symbol=symbol,
    as_of_date=date,
    top_k=3
)

print("RAG returned:", len(results))

for i, r in enumerate(results):
    print(f"\nNeighbor {i} keys:")
    print(r.keys())