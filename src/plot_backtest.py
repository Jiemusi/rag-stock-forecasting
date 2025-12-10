"""
plot_backtest.py — Visualization Toolkit for Backtesting Results
Compatible with backtest.py output:
results = {
    "pred_df": pred_df,
    "real_df": real_df,
    "IC": ic_res,
    "LongShort": ls_res,
    "HitRate": hit_res
}
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ============================================================
# 1. PLOT IC TIME SERIES
# ============================================================

def plot_ic_timeseries(results):
    ic = results["IC"]["daily_ic"]
    rank_ic = results["IC"]["daily_rank_ic"]

    plt.figure(figsize=(10, 4))
    plt.plot(ic, label="IC", linewidth=2)
    plt.plot(rank_ic, label="Rank IC", linewidth=2)
    plt.axhline(0, color="black", linestyle="--")
    plt.title("IC / Rank IC Time Series")
    plt.xlabel("Time")
    plt.ylabel("Correlation")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 2. PLOT LONG-SHORT SPREAD
# ============================================================

def plot_longshort_timeseries(results):
    spread = results["LongShort"]["daily_spread"]

    plt.figure(figsize=(10, 4))
    plt.plot(spread, label="Long-Short Spread", linewidth=2)
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Daily Long-Short Spread")
    plt.xlabel("Time")
    plt.ylabel("Return Difference")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 3. PLOT CUMULATIVE PNL
# ============================================================

def plot_cumulative_pnl(results):
    cum_pnl = results["LongShort"]["cumulative_pnl"]

    plt.figure(figsize=(10, 4))
    plt.plot(cum_pnl, label="Cumulative PnL", linewidth=3)
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Cumulative Long-Short PnL")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. IC DISTRIBUTION HISTOGRAM
# ============================================================

def plot_ic_histogram(results):
    ic = results["IC"]["daily_ic"]

    plt.figure(figsize=(6, 4))
    sns.histplot(ic, kde=True, bins=20)
    plt.title("Distribution of Daily IC")
    plt.xlabel("IC")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. FULL BACKTEST DASHBOARD
# ============================================================

def plot_backtest_dashboard(results):
    plt.figure(figsize=(16, 10))

    # ---- IC ----
    plt.subplot(2, 2, 1)
    plt.plot(results["IC"]["daily_ic"], label="IC", linewidth=2)
    plt.plot(results["IC"]["daily_rank_ic"], label="Rank IC", linewidth=2)
    plt.axhline(0, color="black", linestyle="--")
    plt.title("IC / Rank IC Time Series")
    plt.legend()

    # ---- Spread ----
    plt.subplot(2, 2, 2)
    plt.plot(results["LongShort"]["daily_spread"], label="Daily Spread")
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Long-Short Spread")

    # ---- Cumulative PnL ----
    plt.subplot(2, 2, 3)
    plt.plot(results["LongShort"]["cumulative_pnl"], label="Cumulative PnL", linewidth=2)
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Cumulative PnL")

    # ---- IC Histogram ----
    plt.subplot(2, 2, 4)
    sns.histplot(results["IC"]["daily_ic"], kde=True, bins=20)
    plt.title("IC Distribution")

    plt.tight_layout()
    plt.show()
