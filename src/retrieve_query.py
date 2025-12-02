# src/retrieve_query.py
'''
This module provides two ways to construct a query for the TS-RAG retrieval system:
	1.	Automatic query construction using stored news + earnings embeddings
	2.	Free-text query construction from user / LLM input

It produces:
	•	query_text → natural language text used for BM25 + LLM reranking
	•	query_emb → OpenAI embedding vector used for vector search in Zilliz/Milvus

This is the first stage of the TS-RAG forecasting pipeline.

'''
import os
import sys
from pymilvus import connections, Collection
from openai import OpenAI
import pandas as pd

# ----------------------------
# FIX IMPORT PATH
# ----------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# import config/config.py
from config.config import OPENAI_API_KEY, ZILLIZ_URI, ZILLIZ_API_KEY

# ----------------------------
# INIT OPENAI CLIENT
# ----------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# CONNECT TO ZILLIZ
# ----------------------------
connections.connect(
    alias="default",
    uri=ZILLIZ_URI,
    token=ZILLIZ_API_KEY
)

NEWS_COLLECTION = "news_articles"
EARN_COLLECTION = "earnings_transcripts"

news_col = Collection(NEWS_COLLECTION)
earn_col = Collection(EARN_COLLECTION)


# ----------------------------
# EMBEDDING
# ----------------------------
def embed_text(text, model="text-embedding-3-large"):
    """Return a vector for text using OpenAI embeddings."""
    res = client.embeddings.create(model=model, input=text)
    return res.data[0].embedding


# ----------------------------
# FETCH TEXT FROM ZILLIZ
# ----------------------------
def load_text_from_collection(collection, symbol, date):
    """
    Pull all text entries for a given symbol + date from a Milvus collection.
    """
    expr = f"symbol == '{symbol}' and date == '{date}'"
    try:
        results = collection.query(
            expr=expr,
            output_fields=["text"]
        )
        return [r["text"] for r in results]
    except Exception as e:
        print(f"[ERROR] Query failed → {e}")
        return []


# ----------------------------
# BUILD QUERY OBJECT
# ----------------------------
def build_query(symbol, date):
    """
    Combine news + earnings text for a given date and produce an embedding.
    """
    news_texts = load_text_from_collection(news_col, symbol, date)
    earn_texts = load_text_from_collection(earn_col, symbol, date)

    all_texts = news_texts + earn_texts

    if not all_texts:
        print(f"[WARNING] No text found for {symbol} on {date}")
        return None, None

    query_text = "\n\n".join(all_texts)
    query_emb = embed_text(query_text)

    return query_text, query_emb

# ----------------------------
# BUILD QUERY FROM FREE TEXT
# ----------------------------
def build_query_from_text(free_text):
    """
    Build a query embedding directly from user-provided text (e.g., LLM summary,
    price commentary, custom news). This bypasses symbol/date lookup.
    """
    if not free_text or len(free_text.strip()) == 0:
        raise ValueError("Query text is empty")
    
    query_emb = embed_text(free_text)
    return free_text, query_emb


# ----------------------------
# MAIN TEST
# ----------------------------
# if __name__ == "__main__":
    # symbol = "NVDA"
    # date = "2024-11-15"

    # query_text, query_emb = build_query(symbol, date)

    # print("\n=== QUERY TEXT PREVIEW ===")
    # print(query_text[:500], "...")

    # print("\nEmbedding dimension:", len(query_emb))

if __name__ == "__main__":
    # 自由输入测试
    custom_text = """
    NVDA surged today after strong AI demand.
    Analysts raised price targets.
    Market expects better Q4 revenue.
    """

    query_text, query_emb = build_query_from_text(custom_text)

    print("\n=== FREE TEXT QUERY PREVIEW ===")
    print(query_text[:500], "...")

    print("\nEmbedding dimension:", len(query_emb))

    # # ----------------------------
    # # RUN RETRIEVER
    # # ----------------------------
    # from retrieve import hybrid_search  # assuming hybrid_search exists in retrieve.py

    # print("\n=== RUNNING RETRIEVER ===")
    # results = hybrid_search(
    #     query_emb=query_emb,
    #     query_text=query_text,
    #     symbol="NVDA",
    #     top_k=5
    # )
    # print(results)