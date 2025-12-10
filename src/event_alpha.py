

"""
event_alpha.py — Event-Driven Alpha Contribution Analysis
Author: Han + ChatGPT

This module analyzes how much each retrieved event (RAG neighbor)
contributes to the model's predicted alpha using:

    alpha_contribution = attention_weight * neighbor_future_return

It supports:
- Per-inference event-level explanation
- Aggregated alpha contribution across all dates & symbols in a backtest
- Ranking the most positive and negative event contributors
"""

import numpy as np
import pandas as pd


# ============================================================
# 1. SINGLE-INFERENCE EVENT CONTRIBUTION
# ============================================================

def compute_event_contribution(result):
    """
    Input result from run_inference():
    {
        "attention": np.array shape (5,),
        "neighbor_values": np.array shape (5,5),
        "rag_results": list of 5 dicts
    }

    Returns DataFrame of event-level alpha contributions.
    """

    attn = result["attention"]            # shape (5,)
    neighbor_returns = result["neighbor_values"][:, -1]  # use day-5 return
    rag = result["rag_results"]

    contributions = attn * neighbor_returns

    rows = []
    for i in range(len(contributions)):
        rows.append({
            "neighbor_id": i,
            "symbol": rag[i]["symbol"],
            "date": rag[i]["date"],
            "similarity": rag[i].get("score", np.nan),
            "attention": attn[i],
            "future_return": neighbor_returns[i],
            "alpha_contribution": contributions[i],
            "text_excerpt": rag[i]["text"][:200] + "..."
        })

    return pd.DataFrame(rows)


# ============================================================
# 2. AGGREGATION OVER BACKTEST
# ============================================================

def aggregate_event_contributions(all_results):
    """
    all_results: list of run_inference() outputs across dates × symbols.

    Returns a DataFrame of all event contributions combined.
    """

    all_rows = []

    for out in all_results:
        df = compute_event_contribution(out)
        df["source_symbol"] = out["symbol"]
        df["source_date"] = out["date"]
        all_rows.append(df)

    if len(all_rows) == 0:
        return pd.DataFrame()

    df_all = pd.concat(all_rows, ignore_index=True)
    return df_all


# ============================================================
# 3. TOP EVENTS ANALYSIS
# ============================================================

def top_positive_events(df, n=10):
    """Return the top n events that contributed most positively to alpha."""
    return df.sort_values("alpha_contribution", ascending=False).head(n)


def top_negative_events(df, n=10):
    """Return the top n events that contributed most negatively to alpha."""
    return df.sort_values("alpha_contribution").head(n)


def event_summary_stats(df):
    """Basic performance statistics."""
    return {
        "mean_contribution": df["alpha_contribution"].mean(),
        "median_contribution": df["alpha_contribution"].median(),
        "std_contribution": df["alpha_contribution"].std(),
        "positive_pct": (df["alpha_contribution"] > 0).mean(),
        "num_events": len(df),
    }


# ============================================================
# 4. SYMBOL-LEVEL EVENT CONTRIBUTION
# ============================================================

def per_symbol_contribution(df):
    """
    Returns contribution aggregated by the symbol that the event originated from.
    """
    return df.groupby("symbol")["alpha_contribution"].sum().sort_values(ascending=False)


# ============================================================
# 5. DATE-LEVEL CONTRIBUTION
# ============================================================

def per_date_contribution(df):
    """
    Returns contribution aggregated by the event's date.
    """
    return df.groupby("date")["alpha_contribution"].sum().sort_values(ascending=False)


# ============================================================
# 6. FULL PIPELINE (READY FOR STREAMLIT)
# ============================================================

def analyze_events(all_results):
    """
    all_results: list of inference outputs across the entire backtest.

    Returns:
        df_all : event-level table
        pos     : top positive events
        neg     : top negative events
        per_sym : symbol-level contribution
        per_dt  : date-level contribution
        stats   : summary statistics
    """

    df_all = aggregate_event_contributions(all_results)

    if df_all.empty:
        return None, None, None, None, None

    pos = top_positive_events(df_all)
    neg = top_negative_events(df_all)
    per_sym = per_symbol_contribution(df_all)
    per_dt = per_date_contribution(df_all)
    stats = event_summary_stats(df_all)

    return df_all, pos, neg, per_sym, per_dt, stats