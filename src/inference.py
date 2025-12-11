import os
import sys
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from openai import OpenAI

# ------------------------------------------------------------------
# Path setup: make sure we can import model.py and retrieve.py
# ------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.model import TSFM
from src.retrieve import multimodal_retrieve

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
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

EMBED_DIM = 3072      # text-embedding-3-large
STRUCT_WINDOW = 30    # as used in TSFM encoder
HORIZON = 5           # model predicts 5-day trajectory
TOP_K_NEIGHBORS = 5

DATA_DIR = os.path.join(ROOT_DIR, "data", "processed_company_dataset")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
client = OpenAI()     # expects OPENAI_API_KEY in env


# ------------------------------------------------------------------
# Embedding helper
# ------------------------------------------------------------------
def embed_text(text: str) -> np.ndarray:
    """
    Get a 3072-d embedding for the given text.
    """
    if text is None or text.strip() == "":
        text = "no news available"

    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    arr = np.array(resp.data[0].embedding, dtype=np.float32)
    if arr.shape[0] != EMBED_DIM:
        raise ValueError(f"Expected embedding dim {EMBED_DIM}, got {arr.shape}")
    return arr


# ------------------------------------------------------------------
# Structured features from CSV
# ------------------------------------------------------------------
def load_struct_window(symbol: str, date_str: str, window: int = STRUCT_WINDOW) -> np.ndarray:
    """
    Load the last `window` days of structured features for `symbol`
    up to and including `date_str`.

    Returns: (window, F) float32 array.
    """
    csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No CSV found for {symbol}: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date")

    as_of = pd.to_datetime(date_str)
    df = df[df["date"] <= as_of]

    if len(df) < window:
        raise ValueError(f"Not enough history for {symbol} before {date_str} (have {len(df)}, need {window})")

    window_df = df.iloc[-window:]
    x = window_df[FEATURE_COLS].values.astype(np.float32)
    return x  # (window, F)


# ------------------------------------------------------------------
# Future trajectory for a given date (neighbor values & actual)
# ------------------------------------------------------------------
# def compute_future_trajectory(symbol: str, date_str: str, horizon: int = HORIZON) -> np.ndarray:
#     """
#     Compute future HORIZON-day relative return trajectory based on 'close'.

#     Returns: (horizon,) float32 array; zeros if insufficient data.
#     """
#     csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
#     if not os.path.exists(csv_path):
#         return np.zeros(horizon, dtype=np.float32)

#     df = pd.read_csv(csv_path, parse_dates=["date"])
#     df = df.sort_values("date")

#     closes = df["close"].values
#     dates = df["date"].dt.strftime("%Y-%m-%d").tolist()

#     if date_str not in dates:
#         return np.zeros(horizon, dtype=np.float32)

#     idx = dates.index(date_str)
#     if idx + horizon >= len(closes):
#         return np.zeros(horizon, dtype=np.float32)

#     base = closes[idx]
#     future = closes[idx + 1 : idx + 1 + horizon]

#     return (future / base - 1.0).astype(np.float32)

def compute_future_trajectory(symbol, date_str, horizon=HORIZON):
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    rets = df["log_ret"].values
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()

    if date_str not in dates:
        return np.zeros(horizon, dtype=np.float32)

    idx = dates.index(date_str)
    if idx + horizon >= len(rets):
        return np.zeros(horizon, dtype=np.float32)

    return rets[idx+1:idx+1+horizon].astype(np.float32)



# ------------------------------------------------------------------
# Build model inputs from symbol + date
# ------------------------------------------------------------------
def build_inference_inputs(
    symbol: str,
    date_str: str,
    top_k: int = TOP_K_NEIGHBORS
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, List[Dict[str, Any]]]:
    """
    Returns:
        x_struct : (1, 30, F)
        x_query  : (1, EMBED_DIM)
        x_keys   : (1, K, EMBED_DIM)
        x_values : (1, K, HORIZON)
        query_text : str
        rag_results : list of retrieved docs
    """

    # ----- 1) Query text + embedding -----
    query_text = f"Recent financial news and events for {symbol} up to {date_str}"
    query_emb = embed_text(query_text)  # (EMBED_DIM,)

    # ----- 2) Structured window -----
    x_struct_np = load_struct_window(symbol, date_str, window=STRUCT_WINDOW)  # (30, F)

    # ----- 3) Retrieve neighbors from Zilliz/Milvus -----
    rag_results = multimodal_retrieve(
        query_text=query_text,
        query_emb=query_emb,
        as_of_date=date_str,
        symbol=symbol,
        top_k=top_k,
    )

    # ----- 4) Build neighbor keys (embeddings) & values (trajectories) -----
    keys: List[np.ndarray] = []
    values: List[np.ndarray] = []

    for r in rag_results:
        emb = np.array(r["embedding"], dtype=np.float32)
        if emb.shape[0] != EMBED_DIM:
            raise ValueError(f"Neighbor embedding dim mismatch: {emb.shape}")
        keys.append(emb)

        # use the same symbol as the main one (retriever is already filtered by symbol)
        traj = compute_future_trajectory(symbol, r["date"])
        values.append(traj)

    # If no neighbors, create one dummy
    if len(keys) == 0:
        keys.append(np.zeros(EMBED_DIM, dtype=np.float32))
        values.append(np.zeros(HORIZON, dtype=np.float32))

    # Pad to top_k
    while len(keys) < top_k:
        keys.append(np.zeros_like(keys[0]))
        values.append(np.zeros_like(values[0]))

    # Truncate if more than top_k
    keys = keys[:top_k]
    values = values[:top_k]

    x_keys_np = np.stack(keys)      # (K, EMBED_DIM)
    x_values_np = np.stack(values)  # (K, HORIZON)

    # ----- 5) Convert to tensors with correct shapes -----
    x_struct = torch.tensor(x_struct_np, dtype=torch.float32).unsqueeze(0)  # (1, 30, F)
    x_query = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0)     # (1, EMBED_DIM)
    x_keys = torch.tensor(x_keys_np, dtype=torch.float32).unsqueeze(0)      # (1, K, EMBED_DIM)
    x_values = torch.tensor(x_values_np, dtype=torch.float32).unsqueeze(0)  # (1, K, HORIZON)

    return x_struct, x_query, x_keys, x_values, query_text, rag_results


# ------------------------------------------------------------------
# Model loading (supports both best_val and SWA checkpoints)
# ------------------------------------------------------------------
def load_tsfm(model_path: str, device: str = DEVICE) -> nn.Module:
    model = TSFM().to(device)
    state = torch.load(model_path, map_location=device)

    # Handle SWA state dict with "module." prefix and "n_averaged"
    if any(k.startswith("module.") for k in state.keys()):
        cleaned = {
            k.replace("module.", ""): v
            for k, v in state.items()
            if k != "n_averaged"
        }
        state = cleaned

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ------------------------------------------------------------------
# Run inference
# ------------------------------------------------------------------
def run_inference(
    symbol: str,
    date_str: str,
    model_path: str = "tsfm_best_val_74.pt",
    top_k: int = TOP_K_NEIGHBORS,
    device: str = DEVICE,
) -> Dict[str, Any]:
    print(f"\n=== Running inference for {symbol} on {date_str} ===")
    print(f"Using model checkpoint: {model_path}")
    print(f"Device: {device}")

    # Build inputs
    x_struct, x_query, x_keys, x_values, query_text, rag_results = build_inference_inputs(
        symbol, date_str, top_k=top_k
    )

    print(f"Retrieved {len(rag_results)} neighbor events.")
    if len(rag_results) > 0:
        top = rag_results[0]
        print("\nTop retrieved event:")
        print(f"- Date:   {top['date']}")
        print(f"- Source: {top['source']}")
        print(f"- Score:  {top['score_final']:.4f}")
        print(f"- Text:   {top['text'][:200]}...")

    # Load model
    model = load_tsfm(model_path, device=device)

    x_struct = x_struct.to(device)
    x_query = x_query.to(device)
    x_keys = x_keys.to(device)
    x_values = x_values.to(device)

    with torch.no_grad():
        y_pred, attn = model(x_struct, x_query, x_keys, x_values)

    # Shapes:
    # y_pred: (1, HORIZON)
    # attn:   (1, 1, K)
    y_pred_np = y_pred.squeeze(0).cpu().numpy()          # (HORIZON,)
    attn_np = attn.squeeze(0).squeeze(0).cpu().numpy()   # (K,)

    # Also compute "actual" trajectory for this date if possible
    actual_np = compute_future_trajectory(symbol, date_str)

    return {
        "symbol": symbol,
        "date": date_str,
        "query_text": query_text,
        "prediction": y_pred_np,
        "attention": attn_np,
        "neighbor_values": x_values.cpu().numpy().squeeze(0),  # (K, HORIZON)
        "rag_results": rag_results,
        "actual": actual_np,
    }


# ------------------------------------------------------------------
# Simple explainability / visualization
# ------------------------------------------------------------------
def explain_inference(result: Dict[str, Any]) -> None:
    symbol = result["symbol"]
    date = result["date"]
    y_pred = result["prediction"]
    actual = result["actual"]
    attn = result["attention"]
    neighbors = result["rag_results"]

    print("\n=== EXPLAINABILITY REPORT ===")
    print(f"Symbol: {symbol} | Date: {date}")
    print("\nQuery text used for RAG:")
    print(result["query_text"])
    print("\nPredicted 5-day trajectory (relative returns):")
    print(np.round(y_pred, 4))

    print("\nActual 5-day trajectory (if available):")
    print(np.round(actual, 4))

    print("\nNeighbor attention weights:")
    for i, w in enumerate(attn):
        if i < len(neighbors):
            r = neighbors[i]
            print(f"  Neighbor {i}: w={w:.3f}, date={r['date']}, source={r['source']}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    out = run_inference("AAPL", "2022-12-30")
    explain_inference(out)
