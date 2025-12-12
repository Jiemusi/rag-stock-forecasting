"""
construct_dataset_baseline.py

Builds a baseline TSFM dataset **without** any RAG / news components.

Each sample contains:
    - symbol : stock ticker (str)
    - date   : anchor date for the prediction window (str, 'YYYY-MM-DD')
    - x_struct : np.ndarray of shape (WINDOW_SIZE, n_features)
                 rolling window of daily structural features
    - y        : np.ndarray of shape (FORECAST_HORIZON,)
                 future relative return trajectory

Splits:
    - Train : 2021-07-03  <= date <= 2023-05-31
    - Val   : 2023-07-01  <= date <= 2023-12-31
    - Test  : date >= 2024-02-01

Input:
    data/processed_company_dataset/processed_company_dataset/*.csv

Output:
    data/tsfm_dataset_baseline/{train,val,test}.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# =====================================================
# 1. CONFIG
# =====================================================

INPUT_DIR = "data/processed_company_dataset"
SAVE_DIR  = "data/tsfm_dataset_baseline"

WINDOW_SIZE = 30           # number of past days used as input
FORECAST_HORIZON = 5       # number of future days to predict

os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================
# 2. TARGET TRAJECTORY
# =====================================================

def compute_future_trajectory(df, horizon=FORECAST_HORIZON):
    """
    For each row i, compute future horizon-day relative returns:

        y[i, j] = close[i + 1 + j] / close[i] - 1,   j = 0..h-1

    If we don't have enough future data (i + horizon >= len(df)),
    we fill that row with NaNs and later drop it.
    """
    closes = df["close"].values
    trajs = []
    for i in range(len(df)):
        if i + horizon >= len(df):
            trajs.append([np.nan] * horizon)
        else:
            base = closes[i]
            future = (closes[i + 1 : i + 1 + horizon] / base - 1).tolist()
            trajs.append(future)
    return np.array(trajs, dtype=np.float32)

# =====================================================
# 3. PROCESS SINGLE COMPANY FILE
# =====================================================

def process_file(filepath):
    """
    Build samples for a single symbol CSV in processed_company_dataset.

    Returns a list of dicts:
        {
            "symbol": str,
            "date":   'YYYY-MM-DD',
            "x_struct": (WINDOW_SIZE, n_features) float32,
            "y":        (FORECAST_HORIZON,) float32
        }
    """
    df = pd.read_csv(filepath, parse_dates=["date"])
    symbol = df["symbol"].iloc[0]

    # Features MUST match your training setup
    FEATURE_COLS = [
        "log_ret",
        "volatility_20d",
        "volume_change",
        "ps_ratio",
        "pe_ratio",
        "rev_growth_qoq",
        "real_rate",
        "yield_curve",
        "unemployment_change",
        "close",
        "is_trading_day",
    ]

    # Compute and attach future trajectories
    trajectories = compute_future_trajectory(df)
    df["target_traj"] = list(trajectories)
    df = df.dropna(subset=["target_traj"])

    # Precompute matrices for speed
    feat_matrix = df[FEATURE_COLS].values.astype(np.float32)
    dates = df["date"].dt.strftime("%Y-%m-%d").values

    samples = []

    # Sliding window over time
    for t in tqdm(
        range(WINDOW_SIZE, len(df)),
        desc=f"Building {symbol}",
        leave=False,
    ):
        curr_date = dates[t]

        # Optional: skip very early dates if you want
        if curr_date < "2022-01-01":
            continue

        # 1) Structured input window
        x_struct = feat_matrix[t - WINDOW_SIZE : t]  # (30, n_features)

        # 2) Future target trajectory
        y = np.array(df["target_traj"].iloc[t], dtype=np.float32)  # (5,)

        samples.append(
            {
                "symbol": symbol,
                "date": curr_date,
                "x_struct": x_struct,
                "y": y,
            }
        )

    return samples

# =====================================================
# 4. SPLIT & SAVE
# =====================================================

def main():
    files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.endswith(".csv")
    ]

    all_samples = []

    print(f"Processing {len(files)} companies from {INPUT_DIR}...")

    for f in files:
        try:
            s = process_file(f)
            all_samples.extend(s)
            print(f"  ✓ {os.path.basename(f)} -> {len(s)} samples")
        except Exception as e:
            print(f"  ✗ Error processing {f}: {e}")

    print(f"\nTotal samples collected: {len(all_samples)}")

    # Chronological split by date (string comparison works with YYYY-MM-DD)
    train, val, test = [], [], []
    for s in all_samples:
        d = s["date"]
        if "2021-07-03" <= d <= "2023-05-31":
            train.append(s)
        elif "2023-07-01" <= d <= "2023-12-31":
            val.append(s)
        elif d >= "2024-02-01":
            test.append(s)
        # dates in the gaps are ignored (e.g., 2023-06, 2024-01)

    print(f"\nStats: Train={len(train)}, Val={len(val)}, Test={len(test)}")

    # Save pickles
    with open(os.path.join(SAVE_DIR, "train.pkl"), "wb") as f:
        pickle.dump(train, f)
    with open(os.path.join(SAVE_DIR, "val.pkl"), "wb") as f:
        pickle.dump(val, f)
    with open(os.path.join(SAVE_DIR, "test.pkl"), "wb") as f:
        pickle.dump(test, f)

    print(f"Baseline dataset saved to {SAVE_DIR}")
    print("Done.")

if __name__ == "__main__":
    main()
