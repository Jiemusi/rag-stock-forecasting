# src/eval_baseline_from_test.py
"""
Baseline model evaluation on held-out test set.

Assumptions:
- You have a baseline dataset pickle at `--test_path` with keys:
    - "x_struct": np.ndarray of shape (N, T, D_feat)
    - "y":        np.ndarray of shape (N, H)  (future H-day returns)
    - "symbol":   array-like of shape (N,) with tickers (e.g., "AAPL")
    - "date":     array-like of shape (N,) with YYYY-MM-DD strings
- You have a trained baseline model checkpoint at `--model_path`,
  whose architecture is defined in `src/model_baseline.py` as:

      class BaselineTSFM(nn.Module):
          def forward(self, x_struct):  # (B, T, D)
              return pred               # (B, H)

This script:
  1) Runs the baseline model on the test set.
  2) Computes:
      - Cross-sectional IC (information coefficient) per date.
      - Long-short performance (top vs bottom quantile).
      - Global sign hit-rate.
  3) Prints a text summary to stdout (no plots / no saving).
"""

import argparse
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from model_baseline import BaselineTSFM  # adjust if your class is named differently


# -------------------------
# Metrics
# -------------------------
def compute_ic_by_date(dates, symbols, y_true_last, y_pred_last):
    """
    Compute cross-sectional IC (Pearson correlation) for each date,
    using the last-day return per (date, symbol).
    """
    per_date_true = defaultdict(list)
    per_date_pred = defaultdict(list)

    for d, yt, yp in zip(dates, y_true_last, y_pred_last):
        per_date_true[d].append(float(yt))
        per_date_pred[d].append(float(yp))

    ic_list = []
    used_dates = []

    for d in sorted(per_date_true.keys()):
        y_t = np.array(per_date_true[d], dtype=float)
        y_p = np.array(per_date_pred[d], dtype=float)

        if len(y_t) < 2:
            continue
        if np.std(y_t) < 1e-8 or np.std(y_p) < 1e-8:
            continue

        corr = np.corrcoef(y_p, y_t)[0, 1]
        if np.isfinite(corr):
            ic_list.append(corr)
            used_dates.append(d)

    ic_arr = np.array(ic_list, dtype=float)
    return used_dates, ic_arr


def compute_long_short(dates, y_true_last, y_pred_last, quantile=0.2):
    """
    For each date:
      - sort symbols by predicted last-day return
      - go long top q, short bottom q
      - compute (mean true return of long) - (mean true return of short)

    Returns:
      dates_ls: list of dates used
      ls_returns: np.array of long-short returns per date
    """
    per_date_true = defaultdict(list)
    per_date_pred = defaultdict(list)

    for d, yt, yp in zip(dates, y_true_last, y_pred_last):
        per_date_true[d].append(float(yt))
        per_date_pred[d].append(float(yp))

    dates_ls = []
    ls_vals = []

    for d in sorted(per_date_true.keys()):
        y_t = np.array(per_date_true[d], dtype=float)
        y_p = np.array(per_date_pred[d], dtype=float)

        n = len(y_t)
        if n < 3:
            continue

        k = int(np.floor(quantile * n))
        if k < 1:
            k = 1

        idx_sort = np.argsort(y_p)  # ascending
        bottom_idx = idx_sort[:k]
        top_idx = idx_sort[-k:]

        r_top = np.mean(y_t[top_idx])
        r_bottom = np.mean(y_t[bottom_idx])
        r_ls = r_top - r_bottom

        dates_ls.append(d)
        ls_vals.append(r_ls)

    return dates_ls, np.array(ls_vals, dtype=float)


def compute_hit_rate(y_true_last, y_pred_last, eps=0.0):
    """
    Global sign hit-rate over all (date, symbol) pairs where |true| > eps.
    """
    y_t = np.array(y_true_last, dtype=float)
    y_p = np.array(y_pred_last, dtype=float)

    mask = np.abs(y_t) > eps
    if mask.sum() == 0:
        return np.nan, 0

    sign_true = np.sign(y_t[mask])
    sign_pred = np.sign(y_p[mask])

    hits = (sign_true * sign_pred > 0).astype(float)
    return hits.mean(), mask.sum()


# -------------------------
# Main eval code
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation on test.pkl")
    parser.add_argument(
        "--test_path",
        type=str,
        default="../data/tsfm_dataset_baseline/test.pkl",
        help="Path to baseline test.pkl (with x_struct, y, symbol, date).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="tsfm_baseline_best_val.pt",
        help="Path to baseline model checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.2,
        help="Top/bottom quantile for long-short portfolios.",
    )
    args = parser.parse_args()

    # ----------------- Load dataset -----------------
    with open("../data/tsfm_dataset/test.pkl", "rb") as f:   # or your actual path
        data_list = pickle.load(f)   # this is a list, NOT a dict

    print(f"Loaded {len(data_list)} samples from test.pkl")
    print("Sample keys:", data_list[0].keys())

    # Stack into arrays
    x_struct = np.stack([d["x_struct"] for d in data_list], axis=0)  # (N, T, D)
    y        = np.stack([d["y"]        for d in data_list], axis=0)  # (N, H)
    symbols  = np.array([d["symbol"]   for d in data_list])
    dates    = np.array([d["date"]     for d in data_list])

    N, T, D = x_struct.shape
    H = y.shape[1]
    print("x_struct shape:", x_struct.shape)
    print("y shape:", y.shape)

    # ----------------- Build model -----------------
    device = torch.device(args.device)

    # You must ensure these hyperparameters match how you trained
    model = BaselineTSFM(
        input_dim=D,
        num_layers=2,
        horizon=H,
    ).to(device)

    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # ----------------- Run inference -----------------
    all_pred = []

    with torch.no_grad():
        for start in range(0, N, args.batch_size):
            end = min(N, start + args.batch_size)
            xs = torch.tensor(x_struct[start:end], dtype=torch.float32, device=device)
            pred = model(xs)  # (B, H)
            all_pred.append(pred.cpu().numpy())

    y_pred = np.concatenate(all_pred, axis=0)  # (N, H)
    assert y_pred.shape == y.shape

    # Use last day (H-1) for IC / LS / hit-rate
    y_true_last = y[:, -1]
    y_pred_last = y_pred[:, -1]

    # ----------------- Metrics -----------------
    # 1) IC
    ic_dates, ic_vals = compute_ic_by_date(dates, symbols, y_true_last, y_pred_last)
    if len(ic_vals) > 0:
        ic_mean = float(ic_vals.mean())
        ic_std = float(ic_vals.std(ddof=1)) if len(ic_vals) > 1 else 0.0
        ic_t = ic_mean / (ic_std / np.sqrt(len(ic_vals))) if ic_std > 0 and len(ic_vals) > 1 else np.nan
    else:
        ic_mean, ic_std, ic_t = np.nan, np.nan, np.nan

    # 2) Long-short
    ls_dates, ls_vals = compute_long_short(dates, y_true_last, y_pred_last, quantile=args.quantile)
    mask = np.isfinite(ls_vals)
    ls_vals = ls_vals[mask]
    ls_dates = [d for d, m in zip(ls_dates, mask) if m]
    if len(ls_vals) > 0:
        ls_mean = float(ls_vals.mean())
        ls_std = float(ls_vals.std(ddof=1)) if len(ls_vals) > 1 else 0.0
        ls_sharpe = ls_mean / ls_std if ls_std > 0 else np.nan
        ls_hit = float((ls_vals > 0).mean())
    else:
        ls_mean = ls_std = ls_sharpe = ls_hit = np.nan

    # 3) Global hit-rate
    hit_rate, n_pairs = compute_hit_rate(y_true_last, y_pred_last, eps=0.0)

    # ----------------- Print summary -----------------
    print("\n================ BASELINE PANEL EVAL (test.pkl) ================")
    print(f"Test path:                {args.test_path}")
    print(f"Model path:               {args.model_path}")
    print(f"Samples (N):              {N}")
    print(f"Window length (T):        {T}")
    print(f"Horizon (H):              {H}")
    print("----------------------------------------------------")
    print("Cross-sectional IC (last day returns):")
    print(f"  Dates used (IC):        {len(ic_vals)}")
    print(f"  Mean IC:                {ic_mean:.4f}" if np.isfinite(ic_mean) else "  Mean IC:                nan")
    print(f"  IC std:                 {ic_std:.4f}" if np.isfinite(ic_std) else "  IC std:                 nan")
    print(f"  IC t-stat:              {ic_t:.2f}" if np.isfinite(ic_t) else "  IC t-stat:              nan")
    print("----------------------------------------------------")
    print("Long-Short (top/bottom quantile, last day returns):")
    print(f"  Quantile:               {args.quantile:.2f}")
    print(f"  Dates used (LS):        {len(ls_vals)}")
    print(f"  Mean LS return:         {ls_mean:.6f}" if np.isfinite(ls_mean) else "  Mean LS return:         nan")
    print(f"  LS std:                 {ls_std:.6f}" if np.isfinite(ls_std) else "  LS std:                 nan")
    print(f"  LS Sharpe (per-period): {ls_sharpe:.2f}" if np.isfinite(ls_sharpe) else "  LS Sharpe (per-period): nan")
    print(f"  LS hit-rate (r_LS>0):   {ls_hit:.3f}" if np.isfinite(ls_hit) else "  LS hit-rate (r_LS>0):   nan")
    print("----------------------------------------------------")
    print("Global sign hit-rate (all date,symbol pairs):")
    if np.isfinite(hit_rate):
        print(f"  Hit-rate:               {hit_rate:.3f}")
    else:
        print("  Hit-rate:               nan")
    print(f"  N pairs used:           {n_pairs}")
    print("====================================================\n")


if __name__ == "__main__":
    main()
