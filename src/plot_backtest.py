import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from src.eval_panel import compute_ic, compute_long_short, compute_hit_rate
import pandas as pd

def plot_ic_timeseries(bt):
    ic = bt.get("ic_series")
    if ic is None or len(ic)==0:
        st.warning("No IC series.")
        return
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(ic.index, ic.values)
    ax.set_title("IC Timeseries")
    ax.set_xlabel("Date")
    ax.set_ylabel("IC")
    st.pyplot(fig)

def plot_longshort_timeseries(bt):
    ls = bt.get("ls_series")
    if ls is None or len(ls)==0:
        st.warning("No Long-Short series.")
        return
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(ls.index, ls.values)
    ax.set_title("Long-Short Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    st.pyplot(fig)

def plot_cumulative_pnl(bt):
    ls = bt.get("ls_series")
    if ls is None or len(ls)==0:
        st.warning("No LS series for PnL.")
        return
    pnl = np.cumsum(ls.values)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(ls.index, pnl)
    ax.set_title("Cumulative PnL")
    ax.set_xlabel("Date")
    ax.set_ylabel("PnL")
    st.pyplot(fig)

def plot_ic_histogram(bt):
    ic = bt.get("ic_series")
    if ic is None or len(ic)==0:
        st.warning("No IC series.")
        return
    fig, ax = plt.subplots(figsize=(6,3))
    sns.histplot(ic.values, kde=True, ax=ax)
    ax.set_title("IC Distribution")
    ax.set_xlabel("IC")
    st.pyplot(fig)

def plot_backtest_dashboard(bt):
    mean_ic = bt.get("mean_ic")
    rank_ic = bt.get("rank_ic")
    sharpe = bt.get("sharpe")
    hit = bt.get("hit_rate")
    if mean_ic is None:
        st.write("**Mean IC:** N/A")
    else:
        st.write(f"**Mean IC:** {mean_ic:.4f}")

    if rank_ic is None:
        st.write("**Rank IC:** N/A")
    else:
        st.write(f"**Rank IC:** {rank_ic:.4f}")

    if sharpe is None:
        st.write("**Sharpe:** N/A")
    else:
        st.write(f"**Sharpe:** {sharpe:.4f}")

    if hit is None:
        st.write("**Hit Rate:** N/A")
    else:
        st.write(f"**Hit Rate:** {hit:.4f}")

def run_backtest(results, rank_normalize=False, quantile=0.2):
    """
    results: list of inference dicts, each with fields:
      'date', 'symbol', 'prediction' (vector), 'actual' (vector)
    We take last-day prediction and last-day actual for each entry.
    """
    # results must be a list of inference dicts
    if results is None:
        return {}

    # If results is a DataFrame (should NOT happen in app), reject it
    if isinstance(results, pd.DataFrame):
        results = results.to_dict(orient="records")

    # Now results should be a list
    if not isinstance(results, list) or len(results) == 0:
        return {}

    rows = []
    for r in results:
        if not isinstance(r, dict):
            continue
        pred = r.get("prediction")
        act = r.get("actual")
        if pred is None or act is None or len(pred)==0 or len(act)==0:
            continue
        pred_ret = float(pred[-1])
        true_ret = float(act[-1])
        rows.append({
            "date": r["date"],
            "symbol": r["symbol"],
            "pred": pred_ret,
            "true": true_ret
        })

    if rows is None or len(rows) == 0:
        return {}

    panel_df = pd.DataFrame(rows)
    panel_df["date"] = pd.to_datetime(panel_df["date"])
    panel_df = panel_df.sort_values(["date","symbol"])

    if rank_normalize:
        panel_df["pred"] = panel_df.groupby("date")["pred"].rank(pct=True)

    ic_series, ic_summary = compute_ic(panel_df)
    ls_series, ls_summary = compute_long_short(panel_df, quantile=quantile)
    hit_rate, n_used = compute_hit_rate(panel_df)

    bt = {
        "ic_series": ic_series,
        "mean_ic": ic_summary["mean_ic"],
        "rank_ic": ic_summary["mean_ic"],
        "ls_series": ls_series,
        "sharpe": ls_summary["sharpe"],
        "hit_rate": hit_rate
    }
    return bt
