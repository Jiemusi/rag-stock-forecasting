import pickle
import numpy as np
import os
import random

# CONFIG
DATA_FILE = "data/tsfm_dataset/train.pkl"

def check_samples():
    print(f"--- Loading {DATA_FILE} ---")
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: File not found at {DATA_FILE}")
        return

    with open(DATA_FILE, "rb") as f:
        dataset = pickle.load(f)
        
    total_len = len(dataset)
    print(f"✅ Loaded {total_len} samples.")
    
    if total_len == 0:
        print("⚠️ Warning: Dataset is empty!")
        return

    # Pick 3 Random Samples
    indices = random.sample(range(total_len), 3)
    
    print("\n" + "="*50)
    print(f"🔍 INSPECTING 3 RANDOM SAMPLES")
    print("="*50)

    for i, idx in enumerate(indices):
        s = dataset[idx]
        print(f"\nSample #{i+1} (Index {idx})")
        print(f"   🔹 Symbol: {s['symbol']}")
        print(f"   🔹 Date:   {s['date']}")
        print("-" * 30)
        
        # 1. Check Shapes
        print(f"   [Inputs]")
        print(f"   x_struct shape:   {s['x_struct'].shape}  (Expect: 30, 11)")
        print(f"   x_query shape:    {s['x_query'].shape}    (Expect: 3072,)")
        print(f"   x_keys shape:     {s['x_keys'].shape}     (Expect: 5, 3072)")
        print(f"   x_values shape:   {s['x_values'].shape}   (Expect: 5, 5)")
        
        # 2. Check Target
        print(f"   [Target]")
        print(f"   y value:          {s['y']:.6f}          (5-Day Return)")
        
        # 3. Sanity Checks
        # Check for NaNs
        has_nans = np.isnan(s['x_struct']).any() or np.isnan(s['x_query']).any()
        print(f"   [Health Check]")
        print(f"   Contains NaNs?    {'❌ YES' if has_nans else '✅ NO'}")
        
        # Check if Embedding is all zeros (indicates API failure or cache miss)
        is_empty_emb = np.all(s['x_query'] == 0)
        print(f"   Query Embedding?  {'⚠️ All Zeros (Failed)' if is_empty_emb else '✅ Valid (Non-Zero)'}")
        
        # Check Retrieval
        is_empty_keys = np.all(s['x_keys'] == 0)
        print(f"   Retrieved Keys?   {'⚠️ All Zeros (No History)' if is_empty_keys else '✅ Valid'}")

    print("\n" + "="*50)
    print("VERDICT:")
    print("1. If 'x_struct' is (30, 11) -> ✅ Feature Engineering worked.")
    print("2. If 'x_query' is Valid       -> ✅ OpenAI API worked.")
    print("3. If 'x_keys' is Valid        -> ✅ RAG Retrieval worked.")
    print("="*50)

if __name__ == "__main__":
    check_samples()