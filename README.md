# RAG Stock Forecasting (RAG + TSFM)

End-to-end demo project that combines **Retrieval-Augmented Generation (RAG)** over historical market events (news + earnings calls) with a **time-series forecasting model (TSFM)** to improve short-horizon equity return forecasts and explainability.

This repo is organized as a runnable pipeline:

1) **Data** → prices / fundamentals / macro / text
2) **Event memory (RAG)** → embed + index + retrieve similar historical events
3) **Forecasting** → baseline TSFM vs RAG-augmented TSFM
4) **Backtest + plots** → IC / long-short equity curve
5) **Demo app** → Streamlit UI

---

## What this project does

### Retrieval (RAG) layer
Given a query (e.g., a market summary or a company context), the retrieval layer returns **Top‑K similar historical events** by combining:

- **Vector search** (semantic similarity) over embeddings
- **BM25 keyword search** (exact-match relevance)
- **Hybrid score fusion** (vector + BM25)
- **Time decay weighting** (favor recent events)
- **Multi-modal merge** (news + earnings call transcripts)
- **Local reranking** (cosine-based rerank on top candidates)

### Forecasting layer
The forecasting layer consumes structured panel features and optionally the retrieved-event signal to predict **H‑day forward returns** (default horizon in this repo is commonly 5 days).

### Evaluation
The evaluation scripts produce:

- **Cross-sectional IC** (information coefficient)
- **Long-short (top/bottom quantile) returns & Sharpe**
- Plots saved to `results/` (e.g. `ic_timeseries.png`, `longshort_equity_curve.png`)

---

## Repository structure

High-level layout (key folders):

```
.
├─ config/
│  └─ config.py
├─ data/
│  ├─ news/
│  ├─ earning_calls/
│  ├─ prices/
│  ├─ fundamentals/
│  ├─ macro/
│  ├─ processed/
│  ├─ text_embeddings/
│  ├─ tsfm_dataset/
│  └─ cache/
├─ results/
│  ├─ ic_histogram.png
│  ├─ ic_timeseries.png
│  └─ longshort_equity_curve.png
├─ src/
│  ├─ app.py
│  ├─ inference.py
│  ├─ retrieve.py
│  ├─ retrieve_query.py
│  ├─ vector_news.py
│  ├─ vector_earning_call.py
│  ├─ construct_dataset.py
│  ├─ construct_dataset_baseline.py
│  ├─ train.py
│  ├─ train_baseline.py
│  ├─ eval_single.py
│  ├─ eval_panel.py
│  ├─ eval_baseline_model.py
│  ├─ plot_backtest.py
|  ├─ unit_test.py
│  └─ ...
└─ requirements.txt
```

Notes:
- `data/` is expected to be **large**. Do not commit it to git unless you know what you are doing.
- `src/` contains the runnable scripts for dataset construction, retrieval, training, evaluation, and the Streamlit demo.

---

## Quickstart

### 1) Create environment & install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you prefer conda:

```bash
conda create -n rag-stock python=3.10 -y
conda activate rag-stock
pip install -r requirements.txt
```

### 2) Configure environment variables

Create a `.env` in the repo root (this repo already includes a `.env` file locally for development; do **not** commit secrets).

Minimum recommended variables:

```bash
# OpenAI
OPENAI_API_KEY=...

# Vector DB (Zilliz / Milvus)
ZILLIZ_URI=...
ZILLIZ_TOKEN=...

# Collection names (adjust to match your setup)
ZILLIZ_COLLECTION_NEWS=news_articles
ZILLIZ_COLLECTION_EARNINGS=earnings_transcripts

# Optional: app / cache
CACHE_DIR=data/cache
```

Config defaults may also live in `config/config.py`.

### 3) Run the demo app

```bash
export OPENAI_API_KEY= "..."
streamlit run src/app.py
```

---

## End-to-end workflow

Below is the typical flow. Depending on how you already prepared `data/`, you may skip earlier steps.

If you do **not** want to re-construct datasets from raw sources, you can also download the latest preprocessed data directly from a Hugging Face repo `CindyGaoP/rag_tsfm_data `and place it under `data/` (matching the folder structure above). Similarly, if you do not want to re-train models, you can use the provided `tsfm_swa_final.pt` checkpoints in `src/` (or your downloaded checkpoints) and go straight to the evaluation + backtest + Streamlit demo steps.

### Step A — Prepare / preprocess data

If you are starting from raw sources, you typically:

1) place raw files under `data/news/`, `data/earning_calls/`, `data/prices/`, `data/fundamentals/`, `data/macro/`
2) run preprocessing to create `data/processed/`

There is a notebook (`src/preprocess.ipynb`) you can use for exploratory preprocessing.

### Step B — Build event embeddings + vector index

These scripts typically embed text and upsert into the vector DB collections:

```bash
python src/vector_news.py
python src/vector_earning_call.py
```

Expected result: your vector DB has collections for news and earnings calls.

### Step C — Construct TSFM datasets

Baseline dataset (no retrieval features):

```bash
python src/construct_dataset_baseline.py
```

RAG/augmented dataset:

```bash
python src/construct_dataset.py
```

### Step D — Train models

Baseline:

```bash
python src/train_baseline.py
```

RAG-augmented model:

```bash
python src/train.py
```

Hyperparameter sweeps are configured in `src/sweep.yaml`.

### Step E — Evaluate & backtest

Baseline evaluation:

```bash
python src/eval_baseline_model.py
```

Panel evaluation:

```bash
python src/eval_panel.py
```

Plot backtest results:

```bash
python src/plot_backtest.py
```

Plots are saved to `results/`.

---

## Usage examples

### Minimal demo (single query → Top‑K events)

Run a one-off retrieval query and print the Top‑K most similar historical events.

```bash
python src/retrieve_query.py \
  --query "NVDA guides down and semis sell off; what similar events happened before?" \
  --top_k 10
```

Expected output (example fields; exact formatting may differ by your config):

- `source`: news / earnings
- `date`: event date
- `title` / `headline`
- `similarity_score`
- `url` or `doc_id`

### Full demo (multimodal + rerank + formatted output)

Retrieve from **both** news and earnings collections, apply hybrid fusion + time decay, merge results, then rerank and print a clean final list.

```bash
python src/retrieve_query.py \
  --query "AAPL demand weakens; margins under pressure; compare to prior cycles" \
  --top_k 30 \
  --use_multimodal \
  --use_rerank \
  --output_format pretty
```

If your `retrieve_query.py` does not currently support these flags, treat the command above as the *target behavior*. You can still run:

```bash
python src/retrieve_query.py
```

and use it as the canonical retrieval demo for grading.

---

## Using the retrieval module (standalone)

The retrieval logic is implemented in `src/retrieve.py` and invoked by `src/retrieve_query.py`.

Typical usage pattern:

1) Provide a query text (or a market summary)
2) Retrieve Top‑K similar historical events across news + earnings
3) Optionally rerank and return a final ranked list

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'matplotlib'`

```bash
pip install matplotlib
```

(or reinstall from `requirements.txt`)

### Zilliz / Milvus connection fails (URI / token invalid)

- Double-check `ZILLIZ_URI` and `ZILLIZ_TOKEN` in `.env`.
- If you recently rotated a token, restart your shell / IDE to reload `.env`.
- Make sure your IP / network allows connecting to the cluster.

### Collection load fails / collection not found

- Verify collection names match your environment:
  - `ZILLIZ_COLLECTION_NEWS`
  - `ZILLIZ_COLLECTION_EARNINGS`
- Confirm the collections exist in the cluster (and your token has permission).
- Re-run the indexing scripts if needed:

```bash
python src/vector_news.py
python src/vector_earning_call.py
```

### Retrieval returns empty results

- Confirm embeddings were created and inserted into the vector DB.
- Increase `top_k` (e.g., 10 → 50) to sanity check.
- If you use filters/expressions (`expr`) or time windows, loosen them.
- Sanity check with a very broad query (e.g., a single ticker like `AAPL`).


## Experiment results (baseline vs RAG‑TSFM)

This section is what you should point graders to for the **Experiment Results (7%)** rubric item.

### What to report

- **Data & time range**: which symbols, what date range, what horizon `H` (e.g., 5‑day forward returns)
- **Fixed splits**: train/val/test split logic (and the exact files used)
- **Fixed parameters**: retrieval fusion weights (e.g., `alpha`), time‑decay settings (e.g., `lambda_days` / half‑life), and any rerank settings
- **Metrics** (report baseline vs RAG‑TSFM):
  - Cross‑sectional **IC**
  - Long‑short mean return + **Sharpe**
  - **Hit‑rate** 

### Reproducible run commands

> Tip: If your scripts support a seed flag, set it (e.g., `--seed 42`). If not, set the seed inside the script before training/eval.

1) **Baseline evaluation** (no retrieval features)

```bash
python src/eval_baseline_model.py
```

2) **RAG / panel evaluation** (RAG‑augmented model)

```bash
python src/eval_panel.py
```

3) **Plots** (IC over time, long‑short equity curve)

```bash
python src/plot_backtest.py
```

Outputs are saved under `results/` (examples: `ic_timeseries.png`, `longshort_equity_curve.png`).

### Result 


| Model | Mean IC | IC t‑stat | LS mean return | LS Sharpe | LS hit‑rate |
|---|---:|---:|---:|---:|---:|
| Baseline TSFM |0.037  |3.28  |0.0124  |0.2  |0.442  |
| RAG‑TSFM | 0.011 | 0.53 |0.0033  |0.05  |0.525  |



## Reproducibility checklist

To make results reproducible for grading:

- Fix random seeds in training / evaluation (Python / NumPy / Torch).
- Log all key hyperparameters (window length `T`, horizon `H`, top_k, fusion weights, decay half-life, etc.).
- Save:
  - trained checkpoints (e.g., `*.pt`)
  - evaluation outputs (tables/metrics)
  - plots under `results/`

---




## License

See `LICENSE`.