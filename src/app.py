import streamlit as st
import datetime
import matplotlib.pyplot as plt
from retrieve_query import build_query_from_text
from retrieve import multimodal_retrieve
from inference import run_inference, explain_inference
from src.plot_backtest import (
    plot_ic_timeseries,
    plot_longshort_timeseries,
    plot_cumulative_pnl,
    plot_ic_histogram,
    plot_backtest_dashboard,
    run_backtest,
)
from event_alpha import analyze_events
import pickle
import pandas as pd


# ============================================================
# PANEL DF BUILDER FOR BACKTEST
# ============================================================
def build_panel_from_results(results, horizon=5):
    import numpy as np
    rows = []
    for r in results:
        pred = float(r["prediction"][-1])

        actual = r.get("actual", None)
        if actual is not None and len(actual) > 0:
            true = float(actual[-1])
        else:
            true = np.nan

        rows.append({
            "date": pd.to_datetime(r["date"]),
            "symbol": r["symbol"],
            "pred": pred,
            "true": true
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["pred", "true"])
    return df


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
        "⚙️ End-to-End Pipeline (Benchmark + Backtest + LLM Report)",
    ],
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
        out = run_inference(
            symbol=selected_symbol,
            date_str=str(as_of_date),
            model_path="src/tsfm_swa_final.pt"
        )

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
        figs = explain_inference(out)
        if isinstance(figs, dict):
            if "neighbor_attn" in figs:
                st.pyplot(figs["neighbor_attn"])
            if "prediction" in figs:
                st.pyplot(figs["prediction"])
            if "pred_vs_actual" in figs:
                st.pyplot(figs["pred_vs_actual"])




# ============================================================
# 3. BACKTEST RESULTS (Multi‑Symbol Panel Backtest)
# ============================================================

if page == "📈 Backtest Results":
    st.title("Cross‑Sectional Backtest (Multi‑Symbol Panel)")

    st.write("Run a panel backtest across multiple symbols to compute IC, Rank IC, Sharpe, and long‑short performance.")

    symbols_input = st.text_input("Symbols (comma-separated)", "NVDA, AAPL, MSFT, AMD")
    symbols_bt = [s.strip().upper() for s in symbols_input.split(",")]

    start_date_bt = st.date_input("Start Date", datetime.date(2024, 1, 1))
    end_date_bt = st.date_input("End Date", datetime.date(2024, 3, 1))
    rank_norm_bt = st.checkbox("Use rank-normalized predictions?", True)

    if st.button("Run Backtest"):
        st.write("### ⏳ Running Inference for All Symbols…")

        date_range_bt = pd.date_range(start=start_date_bt, end=end_date_bt)
        results_bt = []

        for sym in symbols_bt:
            for d in date_range_bt:
                d_str = d.strftime("%Y-%m-%d")
                try:
                    out_bt = run_inference(
                        symbol=sym,
                        date_str=d_str,
                        model_path="src/tsfm_swa_final.pt"
                    )
                    results_bt.append(out_bt)
                except Exception as e:
                    st.error(f"Inference failed for {sym} @ {d_str}: {e}")

        st.success(f"Collected {len(results_bt)} samples across {len(symbols_bt)} symbols.")

        # --------------------------
        # Run Backtest
        # --------------------------
        st.write("### 📈 Backtest Results")

        panel_bt = build_panel_from_results(results_bt)
        bt_panel = run_backtest(panel_bt, rank_normalize=rank_norm_bt)

        plot_backtest_dashboard(bt_panel)

        st.write("### IC Timeseries")
        plot_ic_timeseries(bt_panel)

        st.write("### Long‑Short Return Timeseries")
        plot_longshort_timeseries(bt_panel)

        st.write("### Cumulative PnL")
        plot_cumulative_pnl(bt_panel)

        st.write("### IC Distribution")
        plot_ic_histogram(bt_panel)

# ============================================================
# 5. END-TO-END PIPELINE (Benchmark + Backtest + LLM Report)
# ============================================================

if page == "⚙️ End-to-End Pipeline (Benchmark + Backtest + LLM Report)":
    st.title("End-to-End Quant Pipeline + Benchmark + LLM Report")

    st.write("This page runs the entire TS-RAG system in one shot:")

    st.markdown("""
    **✔ TSFM baseline prediction**  
    **✔ TSFM + RAG prediction**  
    **✔ Backtest (IC, Rank IC, Sharpe, PnL)**  
    **✔ Event Alpha Attribution**  
    **✔ LLM interpretation of market events**  
    **✔ Auto-generated PDF research note**
    """)

    # ------------------------------
    # User Inputs
    # ------------------------------
    symbols_input = st.text_input("Symbols (comma-separated)", "NVDA, AAPL, MSFT")
    symbols = [s.strip().upper() for s in symbols_input.split(",")]

    start_date = st.date_input("Start Date", datetime.date(2024, 1, 1))
    end_date = st.date_input("End Date", datetime.date(2024, 3, 1))

    horizon = st.number_input("Prediction Horizon", value=5, min_value=3, max_value=10)
    rank_norm = st.checkbox("Use Rank-normalized Signal?", True)

    if st.button("🚀 Run Full End-to-End Pipeline"):
        st.write("### ⏳ Running Inference (Baseline & RAG)…")

        date_range = pd.date_range(start=start_date, end=end_date)

        results_baseline = []
        results_rag = []

        # 1️⃣ Run both models in parallel
        for sym in symbols:
            for d in date_range:
                d_str = d.strftime("%Y-%m-%d")

                # # Baseline TSFM (no RAG)
                # try:
                #     out_base = run_inference(sym, d_str, use_rag=False)
                #     results_baseline.append(out_base)
                # except Exception as e:
                #     st.error(f"Inference failed: {e}")

                # TSFM + RAG
                try:
                    out_rag = run_inference(
                        symbol=sym,
                        date_str=d_str,
                        model_path="src/tsfm_swa_final.pt"
                    )
                    results_rag.append(out_rag)
                except Exception as e:
                    st.error(f"Inference failed: {e}")

        st.success(f"Baseline samples = {len(results_baseline)}, RAG samples = {len(results_rag)}")

        # ------------------------------
        # 2️⃣ Backtest both models
        # ------------------------------
        st.write("### 📈 Running Backtest…")

        # bt_base = run_backtest(results_baseline, rank_normalize=rank_norm)
        panel_rag = build_panel_from_results(results_rag)
        bt_rag = run_backtest(panel_rag, rank_normalize=rank_norm)

        col1, col2 = st.columns(2)
        # with col1:
        #     st.subheader("TSFM Baseline Backtest")
        #     plot_backtest_dashboard(bt_base)

        with col2:
            st.subheader("TSFM + RAG Backtest")
            plot_backtest_dashboard(bt_rag)

        # --- Additional RAG Backtest Plots ---
        st.write("### IC Timeseries")
        plot_ic_timeseries(bt_rag)

        st.write("### Long-Short Return Timeseries")
        plot_longshort_timeseries(bt_rag)

        st.write("### Cumulative PnL")
        plot_cumulative_pnl(bt_rag)

        st.write("### IC Distribution")
        plot_ic_histogram(bt_rag)

        # ------------------------------
        # 3️⃣ Benchmark Table
        # ------------------------------
        st.subheader("📊 Performance Benchmark: Baseline vs RAG")

        benchmark_df = pd.DataFrame({
            "Metric": ["Mean IC", "Rank IC", "Sharpe", "Hit Rate"],
            # "TSFM Baseline": [
            #     bt_base["mean_ic"], bt_base["rank_ic"], bt_base["sharpe"], bt_base["hit_rate"]
            # ],
            "TSFM + RAG": [
                bt_rag["mean_ic"], bt_rag["rank_ic"], bt_rag["sharpe"], bt_rag["hit_rate"]
            ],
        })
        st.dataframe(benchmark_df)

        # ------------------------------
        # 4️⃣ Event Alpha Attribution
        # ------------------------------
        st.subheader("🧠 Event Alpha Attribution")

        df_events, pos, neg, per_sym, per_dt, stats = analyze_events(results_rag)

        st.write("### Top Positive Alpha Events")
        st.write(pos.head(5))

        st.write("### Top Negative Alpha Events")
        st.write(neg.head(5))

        # Prepare event text for LLM input
        event_context = ""
        for _, row in pos.head(5).iterrows():
            event_context += f"[POS] {row['date']} {row['symbol']} Alpha={row['alpha_contribution']:.4f}\n{row['text_excerpt']}\n\n"
        for _, row in neg.head(5).iterrows():
            event_context += f"[NEG] {row['date']} {row['symbol']} Alpha={row['alpha_contribution']:.4f}\n{row['text_excerpt']}\n\n"

        # ------------------------------
        # 5️⃣ LLM Market Interpretation
        # ------------------------------
        st.subheader("🧠 LLM Market Commentary")

        llm_prompt = f"""
Write a sell-side style market note summarizing model behavior.

Time period: {start_date} → {end_date}
Symbols: {symbols}

Include:
1. Positive alpha themes
2. Negative alpha themes
3. Sector drivers
4. Macro implications
5. Why TSFM+RAG outperformed or underperformed
6. Risks & forward outlook

Event dataset:
{event_context}
"""

        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": llm_prompt}]
        )
        llm_output = resp.choices[0].message.content
        st.write(llm_output)

        # # ------------------------------
        # # 6️⃣ Generate PDF Research Note
        # # ------------------------------
        # st.subheader("📄 Export PDF Report")

        # from pdf_report import generate_pdf_report
        # pdf_path = generate_pdf_report(
        #     start_date=start_date,
        #     end_date=end_date,
        #     symbols=symbols,
        #     benchmark_df=benchmark_df,
        #     pos_events=pos.head(5),
        #     neg_events=neg.head(5),
        #     llm_summary=llm_output
        # )

        # st.success(f"PDF Report generated: {pdf_path}")
        # st.download_button("Download PDF", data=open(pdf_path, "rb"), file_name="research_report.pdf")