"""
This module implements a Time‑Series Retrieval‑Augmented Generation (TS‑RAG)
retriever designed for financial event analysis and forecasting.

It supports:
• Semantic vector retrieval (Zilliz/Milvus)
• BM25 keyword retrieval
• Hybrid score fusion
• Local embedding reranking (no external API required)
• Time‑decay weighting
• Multi‑modal document retrieval (news + earnings)

The system retrieves historically similar events based on user input
or automatically generated query summaries, enabling downstream
forecasting models (e.g., TSFM) to learn from analogous past market behaviors.

───────────────────────────────────────────────
Pipeline Overview
───────────────────────────────────────────────
Query Text / LLM Summary
        ↓
OpenAI Embedding → Query Vector
        ↓
┌──────────────────────────────────────────────┐
│ 1) Vector Search (Zilliz)                    │
│ 2) BM25 Keyword Search                       │
│ 3) Hybrid Score Fusion                       │
│ 4) Time Decay                                │
│ 5) Multi‑Modal Merge (News + Earnings)        │
│ 6) Local Reranking (cosine‑similarity)  Tried LLM rerank,but costly      │
└──────────────────────────────────────────────┘
        ↓
Top‑K Most Similar Historical Events
"""


from pymilvus import connections, Collection
import numpy as np
import pandas as pd
import math
import os
import sys
from rank_bm25 import BM25Okapi
from openai import OpenAI





# ----------------------------
# Path + Config
# ----------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config.config import OPENAI_API_KEY, ZILLIZ_URI, ZILLIZ_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

NEWS_COLLECTION = "news_articles"
EARN_COLLECTION = "earnings_transcripts"

# ----------------------------
# Connect to Zilliz
# ----------------------------
connections.connect(
    alias="default",
    uri=ZILLIZ_URI,
    token=ZILLIZ_API_KEY
)
print("Connected to Zilliz")

news_col = Collection(NEWS_COLLECTION)
earn_col = Collection(EARN_COLLECTION)
news_col.load()
earn_col.load()

print(f"Collections loaded: {NEWS_COLLECTION}, {EARN_COLLECTION}")

# ----------------------------
# Tokenizer for BM25
# ----------------------------
def tokenize(text):
    return text.lower().split()

# ----------------------------
# Vector Search
# ----------------------------
def vector_search(collection, query_embedding, expr, top_k):
    results = collection.search(
        data=[query_embedding],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        expr=expr,
        output_fields=["symbol", "date", "text", "vector"]
    )

    hits = []
    for h in results[0]:
        hits.append({
            "id": h.id,
            "score_vector": float(h.score),
            "symbol": h.entity.get("symbol"),
            "date": h.entity.get("date"),
            "text": h.entity.get("text"),
            "embedding": h.entity.get("vector"),
            "source": collection.name
        })
    return hits

# ----------------------------
# BM25 Search
# ----------------------------
def bm25_search(texts, query, top_k):
    corpus = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))
    idx = np.argsort(scores)[::-1][:top_k]
    return [{"idx": int(i), "score_bm25": float(scores[i])} for i in idx]

# ----------------------------
# Local Embedding Reranker
# ----------------------------
def rerank_results(query_emb, candidates, top_k=5):
    if len(candidates) == 0:
        return []

    q = np.array(query_emb)
    reranked = []

    for c in candidates:
        emb = np.array(c["embedding"])
        sim = np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb) + 1e-6)
        reranked.append({**c, "rank_score": float(sim)})

    reranked = sorted(reranked, key=lambda x: x["rank_score"], reverse=True)
    return reranked[:top_k]

# ----------------------------
# Score Fusion
# ----------------------------
def fuse_scores(vector_score, bm25_score, alpha=0.6):
    return alpha * vector_score + (1 - alpha) * bm25_score

# ----------------------------
# Time Decay
# ----------------------------
def apply_time_decay(results, as_of_date, lambda_days=60):
    as_of = pd.to_datetime(as_of_date)

    for r in results:
        d = pd.to_datetime(r["date"])
        delta = (as_of - d).days
        decay_raw = math.exp(-delta / lambda_days)
        decay = max(0.1, decay_raw)
        r["score_final"] *= decay
        r["time_decay"] = decay

    return sorted(results, key=lambda x: x["score_final"], reverse=True)

# ----------------------------
# Hybrid Search
# ----------------------------
def hybrid_search(collection, query_text, query_emb, as_of_date, symbol=None,
                  top_k=20, alpha=0.6):

    expr = f"date <= '{as_of_date}'"
    if symbol:
        expr += f" and symbol == '{symbol}'"

    vec = vector_search(collection, query_emb, expr, top_k)
    if len(vec) == 0:
        return []

    texts = [v["text"] for v in vec]
    bm25 = bm25_search(texts, query_text, top_k)

    fused = []
    for i, v in enumerate(vec):
        b = bm25[i]["score_bm25"]
        final = fuse_scores(v["score_vector"], b, alpha)

        v["score_bm25"] = b
        v["score_final"] = final
        fused.append(v)

    return sorted(fused, key=lambda x: x["score_final"], reverse=True)

# ----------------------------
# Multi-modal Retrieval
# ----------------------------
def multimodal_retrieve(query_text, query_emb, as_of_date, symbol,
                        top_k=10,
                        alpha=0.6, lambda_days=60,
                        weight_news=0.6, weight_earn=0.4):

    top_k_news = top_k // 2
    top_k_earn = top_k - top_k_news

    # NEWS
    news_res = hybrid_search(news_col, query_text, query_emb, as_of_date,
                             symbol, top_k_news, alpha)
    news_res = apply_time_decay(news_res, as_of_date, lambda_days)
    for r in news_res:
        r["score_final"] *= weight_news
        r["source"] = "news"

    # EARNINGS
    earn_res = hybrid_search(earn_col, query_text, query_emb, as_of_date,
                             symbol, top_k_earn, alpha)
    earn_res = apply_time_decay(earn_res, as_of_date, lambda_days)
    for r in earn_res:
        r["score_final"] *= weight_earn
        r["source"] = "earnings"

    combined = news_res + earn_res
    combined = sorted(combined, key=lambda x: x["score_final"], reverse=True)

    scores = [r["score_final"] for r in combined]
    s_min, s_max = min(scores), max(scores)
    for r in combined:
        denom = (s_max - s_min) if (s_max - s_min) != 0 else 1e-6
        r["score_final"] = (r["score_final"] - s_min) / denom

    return rerank_results(query_emb, combined, top_k=top_k)

# ----------------------------
# Simple Vector Time-Aware Search
# ----------------------------
def search_time_aware(query_embedding, as_of_date, symbol=None, top_k=5):
    expr = f"date <= '{as_of_date}'"
    if symbol:
        expr += f" and symbol == '{symbol}'"

    results = news_col.search(
        data=[query_embedding],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        expr=expr,
        output_fields=["symbol", "date", "text"]
    )

    formatted = []
    for hit in results[0]:
        formatted.append({
            "id": hit.id,
            "score": float(hit.score),
            "symbol": hit.entity.get("symbol"),
            "date": hit.entity.get("date"),
            "text": hit.entity.get("text")
        })
    return formatted
