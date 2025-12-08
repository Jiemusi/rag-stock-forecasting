import pickle
import numpy as np

PATH = "data/tsfm_dataset/train.pkl"

print(f"Loading: {PATH}")
with open(PATH, "rb") as f:
    data = pickle.load(f)

print(f"Total samples: {len(data)}")

sample = data[0]
print("\n=== SAMPLE KEYS ===")
print(sample.keys())

print("\n=== STRUCT FEATURES (x_struct) ===")
x_struct = sample["x_struct"]
print("type:", type(x_struct))
print("shape:", np.array(x_struct).shape)

print("\n=== QUERY TEXT EMBEDDING (x_query) ===")
x_query = sample["x_query"]
print("type:", type(x_query))
print("shape:", np.array(x_query).shape)

print("\n=== RAG KEYS (x_keys) ===")
x_keys = sample["x_keys"]
print("shape:", np.array(x_keys).shape)

print("\n=== RAG VALUES (x_values) ===")
x_values = sample["x_values"]
print("shape:", np.array(x_values).shape)

print("\n=== TARGET (y) ===")
y = sample["y"]
print("shape:", np.array(y).shape)

print("\n=== FULL SAMPLE PRINT ===")
for k, v in sample.items():
    try:
        print(f"{k}: shape={np.array(v).shape}")
    except:
        print(f"{k}: {type(v)}")
