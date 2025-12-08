import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add root path to find retrieve.py
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from retrieve_query import embed_text
from retrieve import multimodal_retrieve

# ================================
# CONFIGURATION
# ================================
# Input: The CLEAN CSVs from your notebook
INPUT_DIR = "data/processed_company_dataset/processed_company_dataset"
# Output: The Final Tensors
SAVE_DIR = "data/tsfm_dataset"
CACHE_DIR = "data/cache"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Hyperparameters
WINDOW_SIZE = 30        # Lookback Window
FORECAST_HORIZON = 5    # Predict 5-day return
TOP_K_EVENTS = 5        # RAG Context Size
EMBED_DIM = 3072        # Text Embedding Dimension

# ================================
# CACHING (Saves time/money)
# ================================
EMBED_CACHE_FILE = f"{CACHE_DIR}/embed_cache.pkl"
RAG_CACHE_FILE   = f"{CACHE_DIR}/rag_cache.pkl"

if os.path.exists(EMBED_CACHE_FILE):
    with open(EMBED_CACHE_FILE, "rb") as f: EMBED_CACHE = pickle.load(f)
else:
    EMBED_CACHE = {}

if os.path.exists(RAG_CACHE_FILE):
    with open(RAG_CACHE_FILE, "rb") as f: RAG_CACHE = pickle.load(f)
else:
    RAG_CACHE = {}

def save_caches():
    with open(EMBED_CACHE_FILE, "wb") as f: pickle.dump(EMBED_CACHE, f)
    with open(RAG_CACHE_FILE, "wb") as f: pickle.dump(RAG_CACHE, f)

# =====================================================
# 1. HELPER: FAST EMBEDDING
# =====================================================
def fast_embed(text):
    if text in EMBED_CACHE: return EMBED_CACHE[text]
    try:
        emb = embed_text(text)
        # Convert list to numpy immediately to save memory
        emb_np = np.array(emb, dtype=np.float32)
        EMBED_CACHE[text] = emb_np
        return emb_np
    except:
        return np.zeros(EMBED_DIM, dtype=np.float32)

# =====================================================
# 2. HELPER: RAG RETRIEVAL (NO LEAKAGE)
# =====================================================
def get_rag_context(symbol, date_str, df):
    """
    Retrieves (Keys, Values) for the ARM Model.
    STRICTLY enforces date <= date_str.
    """
    cache_key = (symbol, date_str)
    if cache_key in RAG_CACHE: return RAG_CACHE[cache_key]

    # A. Query Embedding (Today's News)
    # This represents "The Market Context Now"
    news_rows = df[df["date"] == date_str]
    if "news_text" in news_rows.columns:
        query_text = " ".join(news_rows["news_text"].astype(str).tolist())
    else:
        query_text = f"{symbol} event on {date_str}"
    query_emb = fast_embed(query_text)

    # B. Retrieve History (The "Memory")
    try:
        # CRITICAL: as_of_date=date_str ensures we don't fetch next week's news
        results = multimodal_retrieve(
            query_text=query_text,
            query_emb=query_emb,
            symbol=symbol,
            as_of_date=date_str, 
            top_k=TOP_K_EVENTS
        )
    except:
        results = []

    # C. Format for Model
    keys_list = []   # The Embeddings
    values_list = [] # The Price Trajectories

    for r in results:
        # Get Vector (Key)
        v = r.get("embedding") or r.get("vector")
        if v is not None:
            keys_list.append(np.array(v, dtype=np.float32))
        else:
            keys_list.append(np.zeros(EMBED_DIM, dtype=np.float32))
            
        # Get Trajectory (Value)
        symbol_r = r.get("symbol")
        date_r = r.get("date")
        traj = None
        if symbol_r is not None and date_r is not None:
            try:
                df_sym = df[df["symbol"] == symbol_r].sort_values("date")
                closes = df_sym["close"].values
                dates_sym = df_sym["date"].dt.strftime("%Y-%m-%d").values
                if date_r in dates_sym:
                    idx = list(dates_sym).index(date_r)
                    if idx + FORECAST_HORIZON < len(closes):
                        base = closes[idx]
                        traj = (closes[idx+1:idx+1+FORECAST_HORIZON] / base - 1).astype(np.float32)
            except:
                traj = None
        if traj is None:
            traj = np.zeros(FORECAST_HORIZON, dtype=np.float32)
        values_list.append(traj)

    # Pad if we found fewer than K events
    while len(keys_list) < TOP_K_EVENTS:
        keys_list.append(np.zeros(EMBED_DIM, dtype=np.float32))
        values_list.append(np.zeros(FORECAST_HORIZON, dtype=np.float32))

    # Stack into Tensors
    rag_packet = {
        "q_emb": query_emb,                                      # (3072,)
        "k_embs": np.stack(keys_list[:TOP_K_EVENTS]),            # (K, 3072)
        "v_trajs": np.stack(values_list[:TOP_K_EVENTS])          # (K, 5)
    }
    
    RAG_CACHE[cache_key] = rag_packet
    return rag_packet

def compute_future_trajectory(df, horizon=FORECAST_HORIZON):
    closes = df["close"].values
    trajs = []
    for i in range(len(df)):
        if i + horizon >= len(df):
            trajs.append([np.nan]*horizon)
        else:
            base = closes[i]
            future = (closes[i+1:i+1+horizon] / base - 1).tolist()
            trajs.append(future)
    return np.array(trajs, dtype=np.float32)

# =====================================================
# 3. MAIN PROCESSING LOOP
# =====================================================
def process_file(filepath):
    # Load the ALREADY PROCESSED csv from your notebook
    df = pd.read_csv(filepath, parse_dates=['date'])
    symbol = df['symbol'].iloc[0]
    
    # Define features (Must match your notebook output)
    FEATURE_COLS = [
        'log_ret', 'volatility_20d', 'volume_change',
        'ps_ratio', 'pe_ratio',
        'rev_growth_qoq',
        'real_rate', 'yield_curve', 'unemployment_change',
        'close', 'is_trading_day'
    ]
    
    trajectories = compute_future_trajectory(df)
    df["target_traj"] = list(trajectories)
    df = df.dropna(subset=["target_traj"])
    
    # Prepare Numpy arrays for speed
    feat_matrix = df[FEATURE_COLS].values.astype(np.float32)
    dates = df['date'].dt.strftime("%Y-%m-%d").values
    
    samples = []
    
    # Sliding Window
    for t in tqdm(range(WINDOW_SIZE, len(df)), desc=f"Building {symbol}", leave=False):
        
        curr_date = dates[t]
        
        # Strict Date Filter (Don't train on 2021 warmup data)
        if curr_date < "2022-01-01":
            continue
            
        # 1. Structured Data (Window)
        x_struct = feat_matrix[t-WINDOW_SIZE : t]
        
        # 2. Unstructured Data (RAG)
        rag_data = get_rag_context(symbol, curr_date, df)
        
        # 3. Target
        y = np.array(df["target_traj"].iloc[t], dtype=np.float32)
        
        # 4. Assemble Sample
        samples.append({
            "symbol": symbol,
            "date": curr_date,
            "x_struct": x_struct,               # (30, 11)
            "x_query": rag_data['q_emb'],       # (3072,)
            "x_keys": rag_data['k_embs'],       # (5, 3072)
            "x_values": rag_data['v_trajs'],    # (5, 5)
            "y": y                              # (5,)
        })
        
    return samples

# =====================================================
# 4. SPLIT & SAVE
# =====================================================
def main():
    files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    
    all_samples = []
    
    print(f"Processing {len(files)} companies from {INPUT_DIR}...")
    
    for f in files:
        try:
            s = process_file(f)
            all_samples.extend(s)
            
            # Save cache periodically
            if len(all_samples) % 500 == 0: save_caches()
                
        except Exception as e:
            print(f"Error on {f}: {e}")

    # Split (Purged Walk-Forward)
    train, val, test = [], [], []
    for s in all_samples:
        d = s['date']
        if "2021-07-03" <= d <= "2023-05-31": train.append(s)
        elif "2023-07-01" <= d <= "2023-12-31": val.append(s)
        elif d >= "2024-02-01": test.append(s)

    # Save
    print(f"\nStats: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    
    pickle.dump(train, open(f"{SAVE_DIR}/train.pkl", "wb"))
    pickle.dump(val, open(f"{SAVE_DIR}/val.pkl", "wb"))
    pickle.dump(test, open(f"{SAVE_DIR}/test.pkl", "wb"))
    
    save_caches()
    print("Dataset Generation Complete.")

if __name__ == "__main__":
    main()