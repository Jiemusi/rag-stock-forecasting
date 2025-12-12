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
        "⚙️ TS-RAG: Query → Prediction → LLM Report",
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
        st.write("Raw prediction array:", out["prediction"])
        st.write("Attention weights:", out["attention"])

        st.subheader("Predicted 5‑Day Return Trajectory")
        fig, ax = plt.subplots()
        ax.plot(out["prediction"], linewidth=3)
        st.pyplot(fig)

        # st.subheader("Attention Weights for Retrieved Events")
        # fig2, ax2 = plt.subplots()
        # ax2.bar(range(len(out["attention"])), out["attention"])
        # st.pyplot(fig2)

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
            # RAG score-based attention over neighbors
            if "attention" in figs:
                st.pyplot(figs["attention"])
            # Optional: raw TSFM attention (for debugging / comparison)
            if "model_attention" in figs:
                st.pyplot(figs["model_attention"])
            if "prediction" in figs:
                st.pyplot(figs["prediction"])
            if "pred_vs_actual" in figs:
                st.pyplot(figs["pred_vs_actual"])




# ============================================================
# 3. BACKTEST RESULTS (Multi‑Symbol Panel Backtest)
# ============================================================

if page == "📈 Backtest Results":
    st.title("Cross‑Sectional Backtest (Multi‑Symbol Panel)")

    st.write("Run a panel backtest across multiple symbols based on a pre-computed panel file "
             "(e.g., generated by src/eval_panel.py with --output_panel).")

    panel_path_bt = st.text_input("Panel file path", "data/panel_eval_rank.pkl")

    symbols_input = st.text_input("Symbols (comma-separated)", "NVDA, AAPL, MSFT, AMD")
    symbols_bt = [s.strip().upper() for s in symbols_input.split(",")]

    start_date_bt = st.date_input("Start Date", datetime.date(2024, 1, 1))
    end_date_bt = st.date_input("End Date", datetime.date(2024, 3, 1))
    rank_norm_bt = st.checkbox("Use rank-normalized predictions?", True)

    if st.button("Run Backtest"):
        # Load panel from disk
        try:
            ext = panel_path_bt.split(".")[-1].lower()
            if ext == "csv":
                panel_all = pd.read_csv(panel_path_bt, parse_dates=["date"])
            else:
                panel_all = pd.read_pickle(panel_path_bt)
        except Exception as e:
            st.error(f"Failed to load panel file: {e}")
            st.stop()

        # Ensure proper datetime type
        panel_all["date"] = pd.to_datetime(panel_all["date"])

        # Apply date + symbol filters
        mask = (panel_all["date"] >= pd.to_datetime(start_date_bt)) & (
            panel_all["date"] <= pd.to_datetime(end_date_bt)
        )
        panel_bt = panel_all[mask]

        if symbols_bt:
            panel_bt = panel_bt[panel_bt["symbol"].isin(symbols_bt)]

        if panel_bt.empty:
            st.warning("No panel rows found for the given symbols and date range.")
        else:
            st.success(f"Loaded {len(panel_bt)} panel rows for backtest.")

            st.write("### 📈 Backtest Results")

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


# 0. SINGLE-PAGE END-TO-END TS-RAG (Query → Prediction → LLM Report)
# ============================================================

if page == "⚙️ TS-RAG: Query → Prediction → LLM Report":
    st.title("TS-RAG: Query → Prediction → LLM Report")

    st.markdown("""
    1. **You describe a news event / thesis / question**  
    2. **System retrieves similar historical events (RAG)**  
    3. **TSFM+RAG model makes a forward return prediction**  
    4. **LLM writes a sell-side style note summarizing the view**
    """)

    user_query = st.text_area(
        "Describe the market event, news, or your thesis for this stock:",
        placeholder="e.g., NVDA just guided revenue above expectations due to strong AI GPU demand..."
    )

    if st.button("Run TS-RAG Pipeline"):
        if not user_query.strip():
            st.error("Please enter a description or question.")
        else:
            # 1) Build query & run RAG retrieval
            st.subheader("🔍 RAG: Similar Historical Events")

            query_text, query_emb = build_query_from_text(user_query)

            st.write("**Query Text Preview**")
            st.write(query_text[:500] + "…")

            results = multimodal_retrieve(
                query_text=query_text,
                query_emb=query_emb,
                as_of_date=str(as_of_date),
                symbol=selected_symbol,
                top_k=5
            )

            if not results:
                st.warning("No RAG results returned for this query and date window.")
            else:
                for i, r in enumerate(results, start=1):
                    st.write("---")
                    st.write(f"**[{i}] Date:** {r['date']}")
                    if "symbol" in r:
                        st.write(f"**Symbol:** {r['symbol']}")
                    st.write(f"**Score:** {r.get('score_final', 0.0):.4f}")
                    st.write(r.get("text", "")[:600] + "…")

            # 2) Run TSFM+RAG prediction for the selected symbol / date
            st.subheader("🤖 TSFM+RAG Prediction")

            try:
                out = run_inference(
                    symbol=selected_symbol,
                    date_str=str(as_of_date),
                    model_path="src/tsfm_swa_final.pt"
                )
            except Exception as e:
                st.error(f"Model inference failed: {e}")
                out = None

            if out is not None:
                # Predicted trajectory
                if "prediction" in out:
                    st.write("**Predicted H-day Return Trajectory**")
                    fig, ax = plt.subplots()
                    ax.plot(out["prediction"], linewidth=3)
                    ax.set_xlabel("Step")
                    ax.set_ylabel("Predicted Return")
                    st.pyplot(fig)

                # Attention over neighbors
                if "attention" in out:
                    st.write("**Attention Weights over Retrieved Events**")
                    fig2, ax2 = plt.subplots()
                    ax2.bar(range(len(out["attention"])), out["attention"])
                    ax2.set_xlabel("Neighbor Index")
                    ax2.set_ylabel("Attention Weight")
                    st.pyplot(fig2)

                # Neighbor trajectories
                if "neighbor_values" in out:
                    st.write("**Neighbor Future Trajectories vs Model Prediction**")
                    fig3, ax3 = plt.subplots()
                    for traj in out["neighbor_values"]:
                        ax3.plot(traj, alpha=0.4)
                    ax3.plot(out["prediction"], linewidth=3, label="Model Prediction")
                    ax3.set_xlabel("Step")
                    ax3.set_ylabel("Return")
                    ax3.legend()
                    st.pyplot(fig3)

                # Optional explanation plots from explain_inference
                st.write("**Predicted vs Actual (if available)**")
                figs = explain_inference(out)
                if isinstance(figs, dict):
                    if "attention" in figs:
                        st.pyplot(figs["attention"])
                    if "model_attention" in figs:
                        st.pyplot(figs["model_attention"])
                    if "prediction" in figs:
                        st.pyplot(figs["prediction"])
                    if "pred_vs_actual" in figs:
                        st.pyplot(figs["pred_vs_actual"])

            # 3) LLM Commentary based on query, RAG events, and prediction
            st.subheader("🧠 LLM Market Commentary")

            # Build event context from retrieved results
            event_context = ""
            for r in results[:5]:
                date_str = r.get("date", "")
                sym = r.get("symbol", selected_symbol)
                score = r.get("score_final", 0.0)
                text_snip = r.get("text", "")[:400].replace("\n", " ")
                event_context += f"[EVENT] {date_str} {sym} (score={score:.4f}) :: {text_snip}\n"

            # Extract a simple scalar prediction if available
            pred_last = None
            if out is not None and "prediction" in out and len(out["prediction"]) > 0:
                pred_last = float(out["prediction"][-1])

            prediction_str = "N/A"
            if pred_last is not None:
                prediction_str = f"{pred_last:.4f} (last-step H-day predicted return)"

            llm_prompt = f"""
You are a senior sell-side equity analyst.

User's free-form description / question:
{user_query}

Stock: {selected_symbol}
As-of date: {as_of_date}

Model summary:
- TSFM+RAG predicted H-day return (last step): {prediction_str}

Retrieved historical events (RAG memory):
{event_context}

Write a concise but professional market note that:
1. Explains the model's directional view (bullish / bearish / neutral) and how strong the conviction is.
2. Links the current situation to the most relevant historical events above.
3. Highlights key upside catalysts and downside risks.
4. Discusses how macro / sector context might affect this trade.
5. Ends with a clear risk-reward assessment and what would invalidate the thesis.
"""

            try:
                from openai import OpenAI
                client = OpenAI()
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": llm_prompt}]
                )
                llm_output = resp.choices[0].message.content
                st.write(llm_output)
            except Exception as e:
                st.error(f"LLM call failed: {e}")



# ============================================================





# ============================================================
# 5. END-TO-END PIPELINE (Benchmark + Backtest + LLM Report)
# ============================================================

if page == "⚙️ End-to-End Pipeline (Benchmark + Backtest + LLM Report)":
    st.title("End-to-End Quant Pipeline + Benchmark + LLM Report")

    st.write("This page runs the entire TS-RAG system in one shot, using a pre-computed panel "
             "of predictions and returns (e.g., from src/eval_panel.py).")

    st.markdown("""
    **✔ TSFM + RAG panel loaded from disk**  
    **✔ Backtest (IC, Rank IC, Sharpe, PnL)**  
    **✔ Event Alpha Attribution (from panel)**  
    **✔ LLM interpretation of model behavior**  
    """)

    # ------------------------------
    # User Inputs
    # ------------------------------
    panel_path = st.text_input("Panel file path", "data/panel_eval_rank.pkl")

    symbols_input = st.text_input("Symbols (comma-separated)", "NVDA, AAPL, MSFT")
    symbols = [s.strip().upper() for s in symbols_input.split(",")]

    start_date = st.date_input("Start Date", datetime.date(2024, 1, 1))
    end_date = st.date_input("End Date", datetime.date(2024, 3, 1))

    horizon = st.number_input("Prediction Horizon", value=5, min_value=3, max_value=10)
    rank_norm = st.checkbox("Use Rank-normalized Signal?", True)

    if st.button("🚀 Run Full End-to-End Pipeline"):
        st.write("### ⏳ Loading panel and running backtest…")

        # 1️⃣ Load panel from disk
        try:
            ext = panel_path.split(".")[-1].lower()
            if ext == "csv":
                panel_all = pd.read_csv(panel_path, parse_dates=["date"])
            else:
                panel_all = pd.read_pickle(panel_path)
        except Exception as e:
            st.error(f"Failed to load panel file: {e}")
            st.stop()

        panel_all["date"] = pd.to_datetime(panel_all["date"])

        # Filter by date + symbols
        mask = (panel_all["date"] >= pd.to_datetime(start_date)) & (
            panel_all["date"] <= pd.to_datetime(end_date)
        )
        panel_sub = panel_all[mask]
        if symbols:
            panel_sub = panel_sub[panel_sub["symbol"].isin(symbols)]

        if panel_sub.empty:
            st.warning("No panel rows found for the selected symbols and date range.")
            st.stop()

        st.success(f"Loaded {len(panel_sub)} panel rows for selected symbols and dates.")

        # ------------------------------
        # 2️⃣ Backtest TSFM + RAG
        # ------------------------------
        st.write("### 📈 Running Backtest…")

        bt_rag = run_backtest(panel_sub, rank_normalize=rank_norm)

        col1, col2 = st.columns(2)
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
        # 4️⃣ Event Alpha Attribution (from panel)
        # ------------------------------
        st.subheader("🧠 Event Alpha Attribution")

        # Construct event alpha dataframe from panel_sub
        # Assume panel_sub contains 'date', 'symbol', 'pred', 'true', and optionally 'text_excerpt' and 'alpha_contribution'
        # If not, we create a simple version with alpha contribution as pred * true (just as a placeholder)
        if 'alpha_contribution' not in panel_sub.columns:
            panel_sub['alpha_contribution'] = panel_sub['pred'] * panel_sub['true']

        if 'text_excerpt' not in panel_sub.columns:
            panel_sub['text_excerpt'] = "No event text available."

        # Aggregate positive and negative alpha events
        pos = panel_sub[panel_sub['alpha_contribution'] > 0].sort_values(by='alpha_contribution', ascending=False)
        neg = panel_sub[panel_sub['alpha_contribution'] < 0].sort_values(by='alpha_contribution')

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