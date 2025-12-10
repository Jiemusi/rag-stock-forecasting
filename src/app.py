import streamlit as st
import datetime
import matplotlib.pyplot as plt
from retrieve_query import build_query_from_text
from retrieve import multimodal_retrieve
from inference import run_inference, explain_inference
from plot_backtest import (
    plot_ic_timeseries,
    plot_longshort_timeseries,
    plot_cumulative_pnl,
    plot_ic_histogram,
    plot_backtest_dashboard
)
from event_alpha import analyze_events
from backtest import run_backtest
import pickle
import pandas as pd


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("TS‑RAG Quant Dashboard")
page = st.sidebar.radio(
    "Navigate",
    [
        "🔍 RAG Search",
        "🤖 Model Prediction & Explanation",
        "📈 Backtest Results",
        "🧠 Event Alpha Analysis"
    ]
)

symbols = ["NVDA", "AAPL", "AMD", "MSFT", "GOOGL", "TSM", "AVGO", "ASML", "QCOM", "TXN"]
selected_symbol = st.sidebar.selectbox("Select stock:", symbols)
as_of_date = st.sidebar.date_input("As‑of Date:", value=datetime.date(2024, 12, 1))


# ============================================================
# 1. RAG SEARCH PAGE
# ============================================================

if page == "🔍 RAG Search":
    st.title("TS‑RAG Market Event Search")

    user_input = st.text_area("Enter market event, news summary, or your own analysis:")

    if st.button("Search"):
        if not user_input.strip():
            st.error("Please enter some text.")
        else:
            st.write("### Building Query…")
            query_text, query_emb = build_query_from_text(user_input)

            st.write("### Query Text Preview")
            st.write(query_text[:500] + "…")

            st.write("### Running Hybrid Retrieval…")
            results = multimodal_retrieve(
                query_text=query_text,
                query_emb=query_emb,
                as_of_date=str(as_of_date),
                symbol=selected_symbol,
                top_k=5
            )

            st.write("### Top Results")
            for r in results:
                st.write("---")
                st.write(f"**Date:** {r['date']}")
                st.write(f"**Score:** {r['score_final']:.4f}")
                st.write(r["text"])



# ============================================================
# 2. MODEL PREDICTION & EXPLANATION PAGE
# ============================================================

if page == "🤖 Model Prediction & Explanation":
    st.title("TSFM Model Prediction & Event Explanation")

    if st.button("Run Prediction"):
        out = run_inference(selected_symbol, str(as_of_date))

        st.subheader("Predicted 5‑Day Return Trajectory")
        fig, ax = plt.subplots()
        ax.plot(out["prediction"], linewidth=3)
        st.pyplot(fig)

        st.subheader("Attention Weights for Retrieved Events")
        fig2, ax2 = plt.subplots()
        ax2.bar(range(len(out["attention"])), out["attention"])
        st.pyplot(fig2)

        st.subheader("Neighbor Future Trajectories")
        fig3, ax3 = plt.subplots()
        for traj in out["neighbor_values"]:
            ax3.plot(traj, alpha=0.5)
        ax3.plot(out["prediction"], linewidth=3, color="black", label="Predicted")
        ax3.legend()
        st.pyplot(fig3)

        st.subheader("Predicted vs Actual (if available)")
        # This uses compare_with_actual inside inference.py (user can add)
        try:
            explain_inference(out)
        except:
            st.info("Actual data unavailable for comparison.")



# ============================================================
# 3. BACKTEST RESULTS PAGE
# ============================================================

if page == "📈 Backtest Results":
    st.title("Backtest Performance Overview")

    st.write("Running backtest on selected stock universe…")

    price_df = pickle.load(open("data/price.csv", "rb"))
    test_symbols = symbols[:8]
    dates = sorted(price_df["date"].astype(str).unique().tolist())[50:200]

    results = run_backtest(
        symbols=test_symbols,
        dates=dates,
        price_df=price_df,
        model_path="tsfm_swa_final.pt"
    )

    st.subheader("IC Time Series")
    fig_ic = plt.figure(figsize=(8,4))
    plot_ic_timeseries(results)
    st.pyplot(fig_ic)

    st.subheader("Long‑Short Spread")
    fig_ls = plt.figure(figsize=(8,4))
    plot_longshort_timeseries(results)
    st.pyplot(fig_ls)

    st.subheader("Cumulative PnL")
    fig_pnl = plt.figure(figsize=(8,4))
    plot_cumulative_pnl(results)
    st.pyplot(fig_pnl)

    st.subheader("IC Distribution")
    fig_hist = plt.figure(figsize=(6,4))
    plot_ic_histogram(results)
    st.pyplot(fig_hist)

    st.subheader("Summary Metrics")
    st.write(pd.DataFrame({
        "Mean IC": [results["IC"]["mean_ic"]],
        "Mean Rank IC": [results["IC"]["mean_rank_ic"]],
        "Long‑Short Spread": [results["LongShort"]["mean_spread"]],
        "Hit Rate": [results["HitRate"]["mean_hit_rate"]]
    }))



# ============================================================
# 4. EVENT ALPHA ANALYSIS PAGE
# ============================================================

if page == "🧠 Event Alpha Analysis":
    st.title("Event‑Driven Alpha Contribution Analysis")

    st.write("Running multiple inferences across dates for event attribution…")

    price_df = pickle.load(open("data/price.csv", "rb"))
    dates = sorted(price_df["date"].astype(str).unique().tolist())[50:80]

    all_results = []
    for d in dates:
        try:
            out = run_inference(selected_symbol, d)
            all_results.append(out)
        except:
            continue

    df_all, pos, neg, per_sym, per_dt, stats = analyze_events(all_results)

    if df_all is None:
        st.error("No event data available.")
    else:
        st.subheader("Top Positive Alpha Events")
        st.write(pos)

        st.subheader("Top Negative Alpha Events")
        st.write(neg)

        st.subheader("Alpha Contribution by Event Symbol")
        st.write(per_sym)

        st.subheader("Alpha Contribution by Event Date")
        st.write(per_dt)

        st.subheader("Summary Statistics")
        st.write(stats)