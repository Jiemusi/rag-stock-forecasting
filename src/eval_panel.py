# src/eval_panel.py

"""
Panel (multi-stock) evaluation for TSFM + RAG model.

This script:
- Loads ALL symbols from data/processed_company_dataset/*.csv
- For each (date, symbol) within a date range:
    - Calls run_inference(symbol, date, model_path)
    - Extracts predicted H-th day return as signal
    - Computes true H-th day log return from prices
- Builds a panel DataFrame: [date, symbol, pred, true]
- Computes:
    * Information Coefficient (IC) across symbols per date
    * Long-short portfolio performance (cross-sectional)
    * Global hit-rate (sign(pred) == sign(true))

Usage:
    python3 -m src.eval_panel --model_path tsfm_best_val_74.pt \
        --start_date 2024-01-01 --end_date 2024-12-31
"""

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

from .inference import run_inference


# -------------------------------------------------------------
# 1. Load price panel
# -------------------------------------------------------------

def load_price_panel():
    """
    Load all symbols from data/processed_company_dataset/*.csv.

    Returns a DataFrame with at least:
        ['date', 'symbol', 'close', 'date_str']
    """
    files = glob("data/processed_company_dataset/*.csv")
    if not files:
        raise FileNotFoundError("No CSV files found in data/processed_company_dataset/*.csv")

    dfs = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["date"])
        if "symbol" not in df.columns or "close" not in df.columns:
            continue
        df = df[["date", "symbol", "close"]].copy()
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No usable CSVs: need columns ['date', 'symbol', 'close'].")

    panel = pd.concat(dfs, ignore_index=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["date_str"] = panel["date"].dt.strftime("%Y-%m-%d")
    return panel


# -------------------------------------------------------------
# 2. Compute true H-day last-day log return for (symbol, date)
# -------------------------------------------------------------

def compute_true_last_return(symbol_df: pd.DataFrame, idx: int, horizon: int = 5) -> float:
    """
    Given a single-symbol df with columns ['date', 'symbol', 'close', 'date_str'],
    compute the LAST-day log return over horizon H:

        r_{t+H} = log(P_{t+H} / P_{t+H-1})

    Returns np.nan if we can't compute it (e.g., not enough data).
    """
    closes = symbol_df["close"].to_numpy()
    n = len(closes)

    # Need t+H and t+H-1  => idx + horizon <= n-1
    if idx + horizon >= n:
        return np.nan

    p0 = closes[idx + horizon - 1]
    p1 = closes[idx + horizon]
    if p0 <= 0 or p1 <= 0:
        return np.nan

    return float(np.log(p1 / p0))


# -------------------------------------------------------------
# 3. Build panel of predictions & truths
# -------------------------------------------------------------

def build_panel_predictions(
    model_path: str,
    start_date: str = None,
    end_date: str = None,
    horizon: int = 5,
):
    """
    For ALL symbols and dates in the panel:
      - compute true last-day return over horizon H
      - call run_inference(symbol, date) to get predicted trajectory
      - take last element as predicted last-day return

    Returns:
        panel_df with columns: ['date', 'symbol', 'pred', 'true']
    """
    panel = load_price_panel()
    symbols = sorted(panel["symbol"].unique())
    print(f"Found {len(symbols)} symbols: {symbols}")

    records = []

    for sym in symbols:
        df_s = panel[panel["symbol"] == sym].sort_values("date").reset_index(drop=True)

        # Optional date filter applied on the anchor date
        if start_date is not None:
            df_s = df_s[df_s["date_str"] >= start_date]
        if end_date is not None:
            df_s = df_s[df_s["date_str"] <= end_date]
        df_s = df_s.reset_index(drop=True)

        if df_s.empty:
            continue

        print(f"\nEvaluating symbol {sym} over {len(df_s)} dates...")
        for idx in tqdm(range(len(df_s)), desc=f"{sym}", leave=False):
            date_str = df_s.loc[idx, "date_str"]

            # Need access to the full unfiltered symbol df for true return calc
            # (we re-filter above, so recompute index on the original panel slice)
            # Simpler: re-grab the full symbol series for true calc:
            full_s = panel[panel["symbol"] == sym].sort_values("date").reset_index(drop=True)
            # Find the matching index in full_s by date
            j = full_s.index[full_s["date_str"] == date_str]
            if len(j) == 0:
                continue
            j = int(j[0])

            true_ret = compute_true_last_return(full_s, j, horizon=horizon)
            if np.isnan(true_ret):
                continue

            try:
                out = run_inference(sym, date_str, model_path=model_path)
                pred_vec = np.asarray(out["prediction"], dtype=float)
                if pred_vec.size == 0:
                    continue
                pred_ret = float(pred_vec[-1])  # LAST-day forecast
            except Exception as e:
                print(f"[WARN] Inference failed for {sym} @ {date_str}: {e}")
                continue

            records.append({
                "date": date_str,
                "symbol": sym,
                "pred": pred_ret,
                "true": true_ret,
            })

    if not records:
        raise RuntimeError("No (date, symbol) pairs produced predictions and truths.")

    df_panel = pd.DataFrame.from_records(records)
    df_panel["date"] = pd.to_datetime(df_panel["date"])
    df_panel = df_panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    return df_panel


# -------------------------------------------------------------
# 4. IC, Long-Short, Hit Rate
# -------------------------------------------------------------

def compute_ic(panel_df: pd.DataFrame):
    """
    Compute cross-sectional Information Coefficient (IC) per date.

    For each date:
        IC_t = corr( pred_i(t), true_i(t) ) across symbols i

    Returns:
        ic_series: pd.Series indexed by date
        summary:   dict with mean, std, t-stat, N
    """
    ics = []
    dates = []

    for date, g in panel_df.groupby("date"):
        g = g.dropna(subset=["pred", "true"])
        if len(g) < 2:
            continue
        x = g["pred"].to_numpy()
        y = g["true"].to_numpy()
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        ic = float(np.corrcoef(x, y)[0, 1])
        ics.append(ic)
        dates.append(date)

    if not ics:
        raise RuntimeError("No valid dates for IC computation.")

    ic_series = pd.Series(ics, index=pd.to_datetime(dates)).sort_index()
    mean_ic = float(ic_series.mean())
    std_ic = float(ic_series.std(ddof=1))
    n = len(ic_series)
    if std_ic > 0 and n > 1:
        t_stat = mean_ic / (std_ic / np.sqrt(n))
    else:
        t_stat = np.nan

    summary = {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "t_stat": float(t_stat),
        "n_dates": n,
    }
    return ic_series, summary


def compute_long_short(panel_df: pd.DataFrame, quantile: float = 0.2):
    """
    Long-short evaluation.

    For each date:
      - sort symbols by 'pred'
      - Long = top quantile
      - Short = bottom quantile
      - r_LS(t) = mean(true_long) - mean(true_short)

    Returns:
        ls_series: pd.Series of long-short returns indexed by date
        summary:   dict with mean, std, sharpe, hit_rate
    """
    ls_returns = []
    dates = []

    for date, g in panel_df.groupby("date"):
        g = g.dropna(subset=["pred", "true"])
        if len(g) < 5:  # need enough names for quantiles
            continue

        q_lo = g["pred"].quantile(quantile)
        q_hi = g["pred"].quantile(1 - quantile)

        long_basket = g[g["pred"] >= q_hi]
        short_basket = g[g["pred"] <= q_lo]

        if long_basket.empty or short_basket.empty:
            continue

        long_ret = float(long_basket["true"].mean())
        short_ret = float(short_basket["true"].mean())
        ls_ret = long_ret - short_ret

        ls_returns.append(ls_ret)
        dates.append(date)

    if not ls_returns:
        raise RuntimeError("No valid dates for long-short computation.")

    ls_series = pd.Series(ls_returns, index=pd.to_datetime(dates)).sort_index()
    mean_ls = float(ls_series.mean())
    std_ls = float(ls_series.std(ddof=1))
    if std_ls > 0:
        sharpe = mean_ls / std_ls
    else:
        sharpe = np.nan

    hit_rate = float((ls_series > 0).mean())

    summary = {
        "mean_ls": mean_ls,
        "std_ls": std_ls,
        "sharpe": float(sharpe),
        "hit_rate": hit_rate,
        "n_dates": len(ls_series),
    }
    return ls_series, summary


def compute_hit_rate(panel_df: pd.DataFrame, eps: float = 0.0):
    """
    Global sign hit-rate across ALL (date, symbol) pairs:

        sign(pred) == sign(true)

    Ignoring entries where |true| <= eps (tiny or zero moves).

    Returns:
        hit_rate: float
        n_used:   number of pairs used
    """
    df = panel_df.dropna(subset=["pred", "true"]).copy()
    df = df[np.abs(df["true"]) > eps]

    if df.empty:
        return np.nan, 0

    sign_pred = np.sign(df["pred"].to_numpy())
    sign_true = np.sign(df["true"].to_numpy())
    hits = (sign_pred == sign_true)
    hit_rate = float(hits.mean())
    return hit_rate, int(len(df))


# -------------------------------------------------------------
# 5. CLI
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Panel (multi-stock) evaluation for TSFM.")
    parser.add_argument("--model_path", type=str, default="tsfm_swa_final.pt",
                        help="Path to TSFM checkpoint.")
    parser.add_argument("--start_date", type=str, default=None,
                        help="Start date (YYYY-MM-DD) for anchor dates, optional.")
    parser.add_argument("--end_date", type=str, default=None,
                        help="End date (YYYY-MM-DD) for anchor dates, optional.")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forecast horizon H (days). Used for true last-day return.")
    parser.add_argument("--quantile", type=float, default=0.2,
                        help="Quantile for long-short (e.g., 0.2 = top/bottom 20%).")
    args = parser.parse_args()

    print("=== BUILDING PANEL PREDICTIONS ===")
    panel_df = build_panel_predictions(
        model_path=args.model_path,
        start_date=args.start_date,
        end_date=args.end_date,
        horizon=args.horizon,
    )

    print(f"\nPanel size: {len(panel_df)} rows, "
          f"{panel_df['symbol'].nunique()} symbols, "
          f"{panel_df['date'].nunique()} dates.")

    # ----- IC -----
    ic_series, ic_summary = compute_ic(panel_df)

    # ----- Long-short -----
    ls_series, ls_summary = compute_long_short(panel_df, quantile=args.quantile)

    # ----- Global hit-rate -----
    hit_rate, n_hit = compute_hit_rate(panel_df, eps=0.0)

    # ----- Print summary -----
    print("\n================ PANEL EVAL SUMMARY ================")
    print(f"Dates used (IC):           {ic_summary['n_dates']}")
    print(f"Mean IC:                   {ic_summary['mean_ic']:.4f}")
    print(f"IC std:                    {ic_summary['std_ic']:.4f}")
    print(f"IC t-stat:                 {ic_summary['t_stat']:.2f}")

    print("\nLong-Short (top/bottom quantile):")
    print(f"  Quantile:                {args.quantile:.2f}")
    print(f"  Dates used (LS):         {ls_summary['n_dates']}")
    print(f"  Mean LS return:          {ls_summary['mean_ls']:.6f}")
    print(f"  LS std:                  {ls_summary['std_ls']:.6f}")
    print(f"  LS Sharpe (per-period):  {ls_summary['sharpe']:.2f}")
    print(f"  LS hit-rate (r_LS>0):    {ls_summary['hit_rate']:.3f}")

    print("\nGlobal sign hit-rate (all date,symbol pairs):")
    print(f"  Hit-rate:                {hit_rate:.3f}")
    print(f"  N pairs used:            {n_hit}")


if __name__ == "__main__":
    main()
