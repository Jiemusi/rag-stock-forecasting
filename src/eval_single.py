# src/eval_custom.py

"""
Custom evaluation script for TSFM + RAG model.

- Uses run_inference(symbol, date, model_path=...)
- Uses close prices from data/processed_company_dataset/*.csv
- Computes predictive metrics for a single symbol:
    * Full H-day trajectory: MSE / RMSE / MAE vs true returns
    * Last-day (day H):      MSE / RMSE / MAE
    * Direction accuracy (last day)
    * Zero baseline comparison
"""

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

from .inference import run_inference


# -------------------------------------------------------------
# 1. Load price data
# -------------------------------------------------------------

def load_price_df(symbol: str) -> pd.DataFrame:
    """
    Load price data for one symbol from data/processed_company_dataset/*.csv.

    Expects columns at least: ['date', 'symbol', 'close'].
    """
    files = glob("data/processed_company_dataset/*.csv")
    if not files:
        raise FileNotFoundError("No CSV files found in data/processed_company_dataset/*.csv")

    dfs = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["date"])
        if "symbol" not in df.columns or "close" not in df.columns:
            continue
        dfs.append(df[["date", "symbol", "close"]])

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df[all_df["symbol"] == symbol].copy()
    if all_df.empty:
        raise ValueError(f"No rows found for symbol {symbol} in processed_company_dataset.")

    all_df = all_df.sort_values("date")
    all_df["date_str"] = all_df["date"].dt.strftime("%Y-%m-%d")
    all_df = all_df.reset_index(drop=True)
    return all_df


# -------------------------------------------------------------
# 2. Compute true H-day trajectory of daily returns
# -------------------------------------------------------------

def compute_true_trajectory(df: pd.DataFrame, idx: int, horizon: int = 5) -> np.ndarray:
    """
    Given a symbol-specific price df with columns ['date', 'symbol', 'close', 'date_str'],
    return the true H-day *daily* log returns starting from index idx:

        r_t+1 = log(P_{t+1} / P_t)
        ...
        r_t+H = log(P_{t+H} / P_{t+H-1})

    Returns a vector of length H, or np.nan vector if not enough future data.
    """
    closes = df["close"].to_numpy()
    n = len(closes)

    if idx + horizon >= n:
        return np.full(horizon, np.nan, dtype=float)

    rets = []
    for i in range(horizon):
        p0 = closes[idx + i]
        p1 = closes[idx + i + 1]
        if p0 <= 0 or p1 <= 0:
            rets.append(np.nan)
        else:
            rets.append(float(np.log(p1 / p0)))
    return np.array(rets, dtype=float)


# -------------------------------------------------------------
# 3. Evaluation loop
# -------------------------------------------------------------

def evaluate_symbol(
    symbol: str,
    model_path: str,
    start_date: str = None,
    end_date: str = None,
    horizon: int = 5,
):
    """
    Loop over dates for a single symbol:
      - run_inference(symbol, date, model_path)
      - get prediction vector (length H)
      - compute true H-day log-return trajectory from prices
      - accumulate metrics

    Returns a dict of summary stats.
    """
    df = load_price_df(symbol)

    # filter by date range if provided
    if start_date is not None:
        df = df[df["date_str"] >= start_date]
    if end_date is not None:
        df = df[df["date_str"] <= end_date]

    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("No data left after applying date filters.")

    preds_list = []
    trues_list = []
    used_dates = []

    print(f"Evaluating {symbol} on {len(df)} anchor dates (will drop ones too close to the end).")

    for idx in tqdm(range(len(df)), desc="Eval dates"):
        date_str = df.loc[idx, "date_str"]

        # true trajectory from prices
        true_vec = compute_true_trajectory(df, idx, horizon=horizon)
        if np.isnan(true_vec).any():
            # skip if we can't form a full H-day trajectory
            continue

        try:
            out = run_inference(symbol, date_str, model_path=model_path)
            pred_vec = np.asarray(out["prediction"], dtype=float)

            if pred_vec.shape[0] < horizon:
                # pad or fall back to last value repeated
                last = float(pred_vec[-1])
                pad = np.full(horizon, last, dtype=float)
                pad[: pred_vec.shape[0]] = pred_vec
                pred_vec = pad
            else:
                pred_vec = pred_vec[:horizon]
        except Exception as e:
            print(f"[WARN] Inference failed for {symbol} @ {date_str}: {e}")
            continue

        preds_list.append(pred_vec)
        trues_list.append(true_vec)
        used_dates.append(date_str)

    if len(preds_list) == 0:
        raise RuntimeError("No successful evaluation samples collected.")

    preds = np.stack(preds_list, axis=0)  # [N, H]
    trues = np.stack(trues_list, axis=0)  # [N, H]

    # ----- metrics -----
    # Full trajectory error
    diff = preds - trues
    mse_full = float(np.mean(diff ** 2))
    rmse_full = float(np.sqrt(mse_full))
    mae_full = float(np.mean(np.abs(diff)))

    # Last day only
    diff_last = preds[:, -1] - trues[:, -1]
    mse_last = float(np.mean(diff_last ** 2))
    rmse_last = float(np.sqrt(mse_last))
    mae_last = float(np.mean(np.abs(diff_last)))

    # Zero baseline (predict all zeros)
    zero_full = np.zeros_like(trues)
    zero_diff = zero_full - trues
    zero_mse_full = float(np.mean(zero_diff ** 2))
    zero_rmse_full = float(np.sqrt(zero_mse_full))
    zero_mae_full = float(np.mean(np.abs(zero_diff)))

    zero_diff_last = zero_full[:, -1] - trues[:, -1]
    zero_mse_last = float(np.mean(zero_diff_last ** 2))
    zero_rmse_last = float(np.sqrt(zero_mse_last))
    zero_mae_last = float(np.mean(np.abs(zero_diff_last)))

    # Directional accuracy on last day (ignoring tiny true moves)
    eps = 1e-8
    mask = np.abs(trues[:, -1]) > eps
    if mask.any():
        sign_hit = np.sign(preds[mask, -1]) == np.sign(trues[mask, -1])
        dir_acc = float(np.mean(sign_hit))
    else:
        dir_acc = float("nan")

    # per-day correlation (pred vs true) across all eval dates
    day_corr = []
    for h in range(horizon):
        x = preds[:, h]
        y = trues[:, h]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            day_corr.append(np.nan)
        else:
            day_corr.append(float(np.corrcoef(x, y)[0, 1]))

    # summary dict
    summary = {
        "symbol": symbol,
        "horizon": horizon,
        "n_samples": int(preds.shape[0]),
        "dates": used_dates,
        "preds": preds,
        "trues": trues,
        "metrics": {
            "full": {
                "mse": mse_full,
                "rmse": rmse_full,
                "mae": mae_full,
            },
            "last": {
                "mse": mse_last,
                "rmse": rmse_last,
                "mae": mae_last,
            },
            "baseline_full": {
                "mse": zero_mse_full,
                "rmse": zero_rmse_full,
                "mae": zero_mae_full,
            },
            "baseline_last": {
                "mse": zero_mse_last,
                "rmse": zero_rmse_last,
                "mae": zero_mae_last,
            },
            "directional_accuracy_last": dir_acc,
            "per_day_corr": day_corr,
        },
    }

    return summary


# -------------------------------------------------------------
# 4. CLI
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Custom TSFM evaluation.")
    parser.add_argument("--symbol", type=str, default="AAPL",
                        help="Ticker to evaluate (must exist in processed_company_dataset).")
    parser.add_argument("--model_path", type=str, default="tsfm_swa_final.pt",
                        help="Path to TSFM checkpoint.")
    parser.add_argument("--start_date", type=str, default=None,
                        help="Start date (YYYY-MM-DD), optional.")
    parser.add_argument("--end_date", type=str, default=None,
                        help="End date (YYYY-MM-DD), optional.")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forecast horizon in days.")
    args = parser.parse_args()

    summary = evaluate_symbol(
        symbol=args.symbol,
        model_path=args.model_path,
        start_date=args.start_date,
        end_date=args.end_date,
        horizon=args.horizon,
    )

    m = summary["metrics"]
    print("\n================ CUSTOM EVAL SUMMARY ================")
    print(f"Symbol:        {summary['symbol']}")
    print(f"Horizon:       {summary['horizon']} days")
    print(f"Samples used:  {summary['n_samples']}")
    print("----------------------------------------------------")
    print("Full H-day trajectory:")
    print(f"  MODEL  MSE:  {m['full']['mse']:.6f}")
    print(f"  MODEL  RMSE: {m['full']['rmse']:.6f}")
    print(f"  MODEL  MAE:  {m['full']['mae']:.6f}")
    print(f"  ZERO   MSE:  {m['baseline_full']['mse']:.6f}")
    print(f"  ZERO   RMSE: {m['baseline_full']['rmse']:.6f}")
    print(f"  ZERO   MAE:  {m['baseline_full']['mae']:.6f}")
    print("----------------------------------------------------")
    print("Last-day (day H) only:")
    print(f"  MODEL  MSE:  {m['last']['mse']:.6f}")
    print(f"  MODEL  RMSE: {m['last']['rmse']:.6f}")
    print(f"  MODEL  MAE:  {m['last']['mae']:.6f}")
    print(f"  ZERO   MSE:  {m['baseline_last']['mse']:.6f}")
    print(f"  ZERO   RMSE: {m['baseline_last']['rmse']:.6f}")
    print(f"  ZERO   MAE:  {m['baseline_last']['mae']:.6f}")
    print("----------------------------------------------------")
    print(f"Directional accuracy (last day, |true|>eps): {m['directional_accuracy_last']:.3f}")
    print("Per-day correlation (pred vs true):")
    for i, c in enumerate(m["per_day_corr"], start=1):
        print(f"  Day {i}: {c:.3f}")


if __name__ == "__main__":
    main()
