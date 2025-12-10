

"""
backtest.py — Full End-to-End Backtesting Pipeline for TSFM RAG Model
Author: Han + ChatGPT
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from inference import run_inference
from eval import compute_ic, compute_long_short, compute_hit_rate

# ============================================================
# 1. LOAD PRICE DATA FOR GROUND TRUTH RETURNS
# ============================================================

def compute_future_return(price_df, symbol, date, horizon=5):
    df = price_df[price_df["symbol"] == symbol].sort_values("date")
    closes = df["close"].values
    dates = df["date"].astype(str).tolist()

    if date not in dates:
        return np.nan

    idx = dates.index(date)
    if idx + horizon >= len(closes):
        return np.nan

    base = closes[idx]
    future = closes[idx + horizon]
    return (future / base - 1)


# ============================================================
# 2. GENERATE PREDICTION MATRIX USING INFERENCE
# ============================================================

def generate_predictions(symbols, dates, model_path):
    """
    Returns pred_df: DataFrame [date × symbol]
    """

    pred_df = pd.DataFrame(index=dates, columns=symbols, dtype=float)

    print("\n=== Running Model Inference for Backtest ===")
    for d in tqdm(dates):
        for sym in symbols:
            try:
                out = run_inference(
                    sym,
                    d,
                    model_path=model_path
                )
                pred_df.loc[d, sym] = out["prediction"][-1]  # use 5‑day prediction
            except Exception:
                pred_df.loc[d, sym] = np.nan

    return pred_df


# ============================================================
# 3. GENERATE GROUND‑TRUTH RETURNS MATRIX
# ============================================================

def generate_real_return_matrix(price_df, symbols, dates, horizon=5):
    real_df = pd.DataFrame(index=dates, columns=symbols, dtype=float)

    print("\n=== Computing Ground Truth Returns ===")
    for d in tqdm(dates):
        for sym in symbols:
            real_df.loc[d, sym] = compute_future_return(price_df, sym, d, horizon=horizon)

    return real_df


# ============================================================
# 4. FULL BACKTEST ROUTINE
# ============================================================

def run_backtest(
    symbols,
    dates,
    price_df,
    model_path="tsfm_swa_final.pt"
):

    # Step 1 — model predictions
    pred_df = generate_predictions(symbols, dates, model_path)

    # Step 2 — real returns
    real_df = generate_real_return_matrix(price_df, symbols, dates)

    # Step 3 — evaluation metrics
    print("\n=== Computing Backtest Metrics ===")
    ic_res = compute_ic(pred_df, real_df)
    ls_res = compute_long_short(pred_df, real_df)
    hit_res = compute_hit_rate(pred_df, real_df)

    return {
        "pred_df": pred_df,
        "real_df": real_df,
        "IC": ic_res,
        "LongShort": ls_res,
        "HitRate": hit_res
    }


# ============================================================
# 5. EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    print("\n=== BACKTEST START ===")

    # load price data
    import pickle
    price_df = pickle.load(open("data/price.csv", "rb"))

    # universe selection
    symbols = sorted(price_df["symbol"].unique().tolist())[:8]   # take first 8
    dates = sorted(price_df["date"].astype(str).unique().tolist())[50:200]

    results = run_backtest(
        symbols=symbols,
        dates=dates,
        price_df=price_df,
        model_path="tsfm_swa_final.pt"
    )

    print("\n=== RESULTS SUMMARY ===")
    print("Mean IC:", results["IC"]["mean_ic"])
    print("Mean Rank IC:", results["IC"]["mean_rank_ic"])
    print("Mean Long/Short Spread:", results["LongShort"]["mean_spread"])
    print("Hit Rate:", results["HitRate"]["mean_hit_rate"])

    print("\nBacktest Completed.\n")