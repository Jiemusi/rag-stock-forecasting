import torch
import pickle
import numpy as np
from model import TSFM
from retrieve import multimodal_retrieve
import matplotlib.pyplot as plt


from openai import OpenAI
client = OpenAI()

def embed_text(text):
    if text is None or text.strip() == "":
        text = "no news available"
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    base = resp.data[0].embedding
    arr = np.array(base, dtype=np.float32)
    pooled = np.concatenate([arr, arr, arr, arr]).astype(np.float32)
    return pooled


# ============================
# Load Structured Window
# ============================
def get_struct_window(df, symbol, date, window=30):
    df_sym = df[df["symbol"] == symbol].sort_values("date")
    dates = df_sym["date"].astype(str).tolist()

    if date not in dates:
        raise ValueError(f"Date {date} not found for symbol {symbol}")

    idx = dates.index(date)

    if idx < window:
        raise ValueError(f"Not enough past data for {symbol} on {date}")

    window_df = df_sym.iloc[idx - window : idx]

    return window_df.drop(columns=["symbol", "date"]).values


# ============================
# Compute Future Trajectory
# ============================
def compute_future_trajectory(df, symbol, date, horizon=5):
    df_sym = df[df["symbol"] == symbol].sort_values("date")
    closes = df_sym["close"].values
    dates = df_sym["date"].astype(str).tolist()

    if date not in dates:
        return np.zeros(horizon, dtype=np.float32)

    idx = dates.index(date)
    if idx + horizon >= len(closes):
        return np.zeros(horizon, dtype=np.float32)

    base = closes[idx]
    future = closes[idx + 1 : idx + 1 + horizon]

    return (future / base - 1).astype(np.float32)


# ============================
# Build Inference Input
# ============================
def build_inference_inputs(news_df, struct_df, price_df, symbol, date):
    # ---- Query text ----
    rows = news_df[(news_df["symbol"] == symbol) & (news_df["date"].astype(str) == date)]
    news_text = " ".join(rows["news_text"].astype(str).tolist()) if len(rows) > 0 else "no news"
    query_emb = embed_text(news_text)

    # ---- Structured Window ----
    x_struct = get_struct_window(struct_df, symbol, date)

    # ---- Retrieve Neighbors ----
    rag_results = multimodal_retrieve(
        query_text=news_text,
        query_emb=query_emb,
        symbol=symbol,
        as_of_date=date,
        top_k=5
    )

    keys = []
    values = []

    for r in rag_results:
        keys.append(r["embedding"])
        traj = compute_future_trajectory(price_df, r["symbol"], r["date"])
        values.append(traj)

    # pad if <5 retrieved
    while len(keys) < 5:
        keys.append(np.zeros_like(keys[0]))
        values.append(np.zeros(5, dtype=np.float32))

    x_keys = np.stack(keys)
    x_values = np.stack(values)

    return (
        torch.tensor(x_struct, dtype=torch.float32).unsqueeze(0),
        torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0),
        torch.tensor(x_keys, dtype=torch.float32).unsqueeze(0),
        torch.tensor(x_values, dtype=torch.float32).unsqueeze(0),
        news_text,
        rag_results
    )


# ============================
# Run Inference
# ============================
def run_inference(
    symbol,
    date,
    news_df_path="data/news.csv",
    struct_df_path="data/struct.csv",
    price_df_path="data/price.csv",
    model_path="tsfm_best.pt",
    device="cuda" if torch.cuda.is_available() else "cpu"
):

    # Load data
    news_df = pickle.load(open(news_df_path, "rb"))
    struct_df = pickle.load(open(struct_df_path, "rb"))
    price_df = pickle.load(open(price_df_path, "rb"))

    # Build inputs
    x_struct, x_query, x_keys, x_values, text, rag_results = build_inference_inputs(
        news_df, struct_df, price_df, symbol, date
    )

    # Load model
    model = TSFM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x_struct = x_struct.to(device)
    x_query = x_query.to(device)
    x_keys = x_keys.to(device)
    x_values = x_values.to(device)

    with torch.no_grad():
        y_pred, attn = model(x_struct, x_query, x_keys, x_values)

    return {
        "symbol": symbol,
        "date": date,
        "news_used": text,
        "prediction": y_pred.cpu().numpy().flatten(),
        "attention": attn.cpu().numpy().flatten(),
        "neighbor_values": x_values.cpu().numpy().squeeze(0),
        "rag_results": rag_results,
        "price_df": price_df
    }


def explain_inference(result):
    print("\n=== EXPLAINABILITY REPORT ===")
    print("Symbol:", result["symbol"])
    print("Date:", result["date"])
    print("\nNews Used:\n", result["news_used"])
    print("\nPrediction (5‑day trajectory):", result["prediction"])
    print("Attention Weights:", result["attention"])

    attn = result["attention"]
    plt.figure(figsize=(6,3))
    plt.bar(range(len(attn)), attn)
    plt.title("Neighbor Attention Weights")
    plt.xlabel("Neighbor Index")
    plt.ylabel("Attention Weight")
    plt.tight_layout()
    plt.show()

    # Advanced plots
    plot_neighbor_trajectories(result["prediction"], result["neighbor_values"])
    plot_attention_heatmap(attn)


import seaborn as sns

def plot_neighbor_trajectories(pred, neighbors):
    plt.figure(figsize=(7,4))
    plt.plot(pred, label="Predicted", linewidth=3, color="black")
    for i, traj in enumerate(neighbors):
        plt.plot(traj, alpha=0.5, label=f"Neighbor {i}")
    plt.title("Predicted vs Neighbor Future Trajectories")
    plt.xlabel("Day")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_attention_heatmap(attn):
    plt.figure(figsize=(4,2))
    sns.heatmap(attn.reshape(1,-1), annot=True, cmap="Blues")
    plt.title("Attention Heatmap")
    plt.yticks([])
    plt.xlabel("Neighbor Index")
    plt.tight_layout()
    plt.show()


def summarize_neighbors(rag_results, values):
    print("\n=== Neighbor Summaries ===")
    for i, r in enumerate(rag_results):
        print(f"\nNeighbor {i}")
        print("Symbol:", r["symbol"])
        print("Date:", r["date"])
        print("Trajectory:", values[i])


def compare_with_actual(price_df, symbol, date, pred, horizon=5):
    df_sym = price_df[price_df["symbol"] == symbol].sort_values("date")
    closes = df_sym["close"].values
    dates = df_sym["date"].astype(str).tolist()

    if date not in dates:
        print("\nNo actual future data available.")
        return

    idx = dates.index(date)
    if idx + horizon >= len(closes):
        print("\nNot enough actual future data available.")
        return

    base = closes[idx]
    future = closes[idx+1:idx+1+horizon]
    actual = (future / base - 1)

    plt.figure(figsize=(7,4))
    plt.plot(pred, label="Predicted", linewidth=3)
    plt.plot(actual, label="Actual", linewidth=3)
    plt.title("Predicted vs Actual Trajectory")
    plt.xlabel("Day")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nActual trajectory:", actual)


# ============================
# Example Usage
# ============================
if __name__ == "__main__":
    out = run_inference("AAPL", "2023-08-01")
    explain_inference(out)
