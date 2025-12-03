import streamlit as st
from retrieve_query import build_query_from_text
from retrieve import multimodal_retrieve

st.title("TS-RAG Market Event Search ")

symbols = ["NVDA", "AAPL", "AMD", "MSFT", "GOOGL", "TSM", "AVGO", "ASML", "QCOM", "TXN"]
selected_symbol = st.selectbox("Select stock symbol:", symbols)

st.subheader("Configure Query")

import datetime
as_of_date = st.date_input("Select as-of date:", value=datetime.date(2024, 12, 1))

# User input box
user_input = st.text_area("Enter market event, news summary, or your own analysis:")

if st.button("Search"):
    if not user_input.strip():
        st.error("Please enter some text.")
    else:
        st.write("### Building Query…")
        query_text, query_emb = build_query_from_text(user_input)

        st.write("### Query Text Preview")
        st.write(query_text[:500] + "…")

        st.write("###  Running Hybrid Retrieval…")
        results = multimodal_retrieve(
            query_text=query_text,
            query_emb=query_emb,
            as_of_date=str(as_of_date),
            symbol=selected_symbol,
            top_k=5
        )

        st.write("###  Top Results")
        for r in results:
            st.write("---")
            st.write(f"**Date:** {r['date']}")
            st.write(f"**Score:** {r['score_final']:.4f}")
            st.write(r["text"])