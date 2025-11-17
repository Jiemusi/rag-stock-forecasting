

"""
News harvesting pipeline for TheNewsAPI.
- Multi-subquery search per company (e.g., ["NVIDIA", "AI GPU", "data center"]).
- Industry topics expressed as compact keywords / boolean-friendly phrases.
- Cleans and saves snippet-level news to CSV (no LLM summarization or embeddings).
- Uses TheNewsAPI advanced params per docs: search_fields, locale, categories, sort, page.
"""

from __future__ import annotations

import os
import time
from datetime import timedelta
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import requests
from dateutil import parser as dateparser
from tqdm import tqdm

# =====================
# Configuration
# =====================
CONFIG: Dict[str, Any] = {
    # ---- API & time ----
    "THENEWSAPI_KEY": os.getenv("THENEWSAPI_KEY", "KaryXB902PSNjjmfamCQimLFEWE8fFLJ0lUo62K4"),
    "FROM_DATE": os.getenv("NEWS_FROM", "2022-01-01"),
    "TO_DATE": os.getenv("NEWS_TO", "2022-12-31"),
    "RANGE_SPLIT_DAYS": int(os.getenv("NEWS_RANGE_SPLIT_DAYS", "15")),
    "PAGE_SIZE": int(os.getenv("NEWS_PAGE_SIZE", "100")),
    "SLEEP_BETWEEN_CALLS_SEC": float(os.getenv("NEWS_SLEEP_SEC", "1.5")),
    "LANGUAGE": os.getenv("NEWS_LANG", "en"),

    # ---- Filters per new docs ----
    # Search across these fields only (per docs: title|description|keywords|main_text)
    "SEARCH_FIELDS": "title,description,keywords,main_text",
    # Limit to English-speaking locales most relevant to you
    "LOCALE": "us,ca,gb",
    # Keep business + tech only for relevance
    "CATEGORIES": "business,tech",
    # Sort by relevance when search is used
    "SORT": "relevance_score",
    # Limit pagination to at most 2 pages per query window
    "MAX_PAGES": 2,
    # Optionally exclude noisy domains (comma-separated); leave empty to skip
    "EXCLUDE_DOMAINS": os.getenv("NEWS_EXCLUDE_DOMAINS", ""),
    # Include similar? Default false to reduce duplicates
    "INCLUDE_SIMILAR": "false",

    # ---- Output ----
    "OUT_DIR_COMPANIES": os.getenv("NEWS_OUT_CO", "data/companies"),
    "OUT_DIR_INDUSTRY": os.getenv("NEWS_OUT_IND", "data/industry"),
}

# =====================
# Company queries (list-based for multi-subquery search)
# Keep these short & distinct to maximize recall in TheNewsAPI.
# =====================
COMPANY_QUERIES: Dict[str, List[str]] = {
    # "NVDA": [
    #     "NVIDIA", "NVDA", "AI GPU", "H100", "Blackwell", "data center"
    # ],
    # "TSM": [
    #     "TSMC", "TSM", "Taiwan Semiconductor", "foundry", "wafer fab"
    # ],
    # "AVGO": [
    #     "Broadcom", "AVGO", "VMware", "custom silicon", "networking chip"
    # ],
    # "ASML": [
    #     "ASML", "ASML Holding", "EUV lithography", "lithography tools"
    # ],
    # "AMD": [
    #     "AMD", "Advanced Micro Devices", "AI GPU", "Instinct", "EPYC", "data center"
    # ],
    # "MSFT": [
    #     "Microsoft", "MSFT", "Azure", "Copilot", "OpenAI"
    # ],
    # "GOOGL": [
    #     "Google", "Alphabet", "GOOGL", "Gemini", "TPU", "Google Cloud"
    # ],
    # "AAPL": [
    #     "Apple", "AAPL", "Vision Pro", "on-device AI", "M-series chip"
    # ],
    "QCOM": [
        "Qualcomm", "QCOM", "Snapdragon", "5G", "edge AI", "automotive"
    ],
    "TXN": [
        "Texas Instruments", "TXN", "analog chips", "embedded", "industrial semiconductors"
    ],
}

# =====================
# Industry topics (compact keyword/boolean-friendly strings)
# =====================
# INDUSTRY_TOPICS: Dict[str, str] = {
#     # AI infra / GPU 需求
#     "ai_infrastructure": '"AI GPU" + (data center | cloud | inference | training | "accelerated computing")',

#     # 半导体供应链（制造 + 包装 + 物流）
#     "semiconductor_supply_chain": '"semiconductor supply chain" + (foundry | packaging | shortage | logistics | capacity | lead time)',

#     # 制程节点 / 光刻
#     "process_nodes": '"2nm" | "3nm" | "5nm" | "EUV lithography" | "advanced lithography" | "advanced node"',

#     # 封装 / HBM / chiplet
#     "chip_packaging": 'CoWoS | chiplet | chiplets | "3D packaging" | HBM | "advanced packaging"',

#     # 地缘政治 / 政策
#     "geopolitical_policy": '"export controls" | "CHIPS Act" | "US China" | sanctions | "tech war"',

#     # 云厂商自研芯片 / TPU / Graviton
#     "custom_silicon": '"custom silicon" | ASIC | "AI accelerator" | "cloud chip" | TPU | Graviton',

#     # M&A / JV / 长期供货
#     "mna_partnerships": 'acquisition | merger | "strategic investment" | partnership | "joint venture" | "supply agreement"',

#     # Cloud + AI 服务（补 AWS）
#     "cloud_ai_services": 'Azure | "Google Cloud" | AWS | "AI services" | "AI cloud" | "data center investment"',

#     # 消费端 AI 设备（补 AI PC）
#     "consumer_ai_devices": '"on-device AI" | "AI smartphone" | "AI PC" | "AI laptop" | "Vision Pro" | "mixed reality"',

#     # Edge / 汽车 / IoT，单独抽出来给 QCOM/TXN
#     "edge_auto_iot": '"edge AI" | "automotive chips" | ADAS | "autonomous driving" | "IoT devices"',

#     # 市场情绪 / 估值相关
#     "financial_market_trends": '"AI rally" | "semiconductor stocks" | "market valuation" | "earnings revision" | guidance',
# }

# =====================
# Helpers
# =====================

def ensure_dirs() -> None:
    os.makedirs(CONFIG["OUT_DIR_COMPANIES"], exist_ok=True)
    os.makedirs(CONFIG["OUT_DIR_INDUSTRY"], exist_ok=True)


def month_ranges(start_date: str, end_date: str, step_days: int) -> List[Tuple[str, str]]:
    """Split a date range into [from, to] chunks of length step_days."""
    start = dateparser.parse(start_date).date()
    end = dateparser.parse(end_date).date()
    ranges: List[Tuple[str, str]] = []
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=step_days - 1), end)
        ranges.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt + timedelta(days=1)
    return ranges


def normalize_article(a: Dict[str, Any], company: Optional[str] = None, topic: Optional[str] = None) -> Dict[str, Any]:
    """Map TheNewsAPI fields to a consistent schema suitable for CSV."""
    published_at = a.get("published_at", "")
    try:
        ts = dateparser.parse(published_at) if published_at else None
        published_iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ") if ts else ""
        published_date = ts.date().isoformat() if ts else ""
    except Exception:
        published_iso = published_at
        published_date = published_at[:10]
    return {
        "company": company,
        "topic": topic,
        "title": a.get("title"),
        "url": a.get("url"),
        "source": a.get("source"),  # per docs: domain of the source
        "author": a.get("author"),
        "published_at": published_iso,
        "published_date": published_date,
        "description": a.get("description") or a.get("snippet"),
        "content": a.get("main_text") or a.get("content") or a.get("description") or a.get("snippet"),
    }


def dedup_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = (r.get("title"), r.get("url"))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def write_csv(rows: List[Dict[str, Any]], outfile: str) -> None:
    if not rows:
        print(f"[INFO] No rows to write for {outfile}")
        return
    df = pd.DataFrame(rows)
    if "published_at" in df.columns:
        df.sort_values("published_at", inplace=True, ascending=False)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_csv(outfile, index=False)
    print(f"[OK] Wrote {len(df)} rows → {outfile}")


# =====================
# TheNewsAPI caller (updated per docs you shared)
# =====================

def fetch_thenewsapi_articles(query: str, from_date: str, to_date: str, page_size: int) -> List[Dict[str, Any]]:
    """Fetch articles for a single query string within a date window."""
    url = "https://api.thenewsapi.com/v1/news/all"
    all_articles: List[Dict[str, Any]] = []
    page = 1
    while True:
        params = {
            "api_token": CONFIG["THENEWSAPI_KEY"],
            "search": query,
            "search_fields": CONFIG["SEARCH_FIELDS"],
            "categories": CONFIG["CATEGORIES"],
            "locale": CONFIG["LOCALE"],
            "language": CONFIG["LANGUAGE"],
            "published_after": from_date,
            "published_before": to_date,
            "sort": CONFIG["SORT"],
            "include_similar": CONFIG["INCLUDE_SIMILAR"],
            "page": page,
            "limit": page_size,
        }
        if CONFIG.get("EXCLUDE_DOMAINS"):
            params["exclude_domains"] = CONFIG["EXCLUDE_DOMAINS"]

        resp = requests.get(url, params=params)
        try:
            data = resp.json()
        except Exception:
            print(f"[WARN] Non-JSON response for query '{query}': status={resp.status_code}")
            break

        articles = data.get("data", [])
        if not articles:
            print(f"[INFO] No results for query: {query[:60]}... ({from_date}→{to_date})")
            break

        all_articles.extend(articles)

        # Stop if less than a full page returned or page cap reached
        if len(articles) < page_size or page >= int(CONFIG["MAX_PAGES"]):
            break

        page += 1
        time.sleep(CONFIG["SLEEP_BETWEEN_CALLS_SEC"])

    return all_articles


# =====================
# Harvesting (multi-subquery per label)
# =====================

def harvest_simple(label: str, queries, out_csv: str, company: Optional[str] = None, topic: Optional[str] = None) -> None:
    """Collect and save snippet-level news for a company/topic across date slices."""
    ranges = month_ranges(CONFIG["FROM_DATE"], CONFIG["TO_DATE"], CONFIG["RANGE_SPLIT_DAYS"])
    rows: List[Dict[str, Any]] = []
    for (st, ed) in tqdm(ranges, desc=f"[{label}] date slices"):
        sub_queries = queries if isinstance(queries, list) else [queries]
        for i, q in enumerate(sub_queries, 1):
            print(f"[{label}] Sub-query {i}/{len(sub_queries)}: {q}")
            try:
                arts = fetch_thenewsapi_articles(q, st, ed, CONFIG["PAGE_SIZE"])
            except Exception as e:
                print(f"[WARN] Query failed for {st}→{ed}: {e}")
                continue
            for a in arts:
                rows.append(normalize_article(a, company=company, topic=topic))
            print(f"[{label}] Sub-query '{q}' → {len(arts)} articles")
            time.sleep(CONFIG["SLEEP_BETWEEN_CALLS_SEC"])

    rows = dedup_rows(rows)
    write_csv(rows, out_csv)


# =====================
# Entrypoint
# =====================

def main() -> None:
    ensure_dirs()
    print("\n=== Starting News Harvest (Simplified Multi-query) ===")
    print(f"Date range: {CONFIG['FROM_DATE']} → {CONFIG['TO_DATE']}")
    print(f"Range split: every {CONFIG['RANGE_SPLIT_DAYS']} days\n")

    # Companies
    for ticker, q_list in COMPANY_QUERIES.items():
        out_csv = os.path.join(CONFIG['FROM_DATE'],CONFIG['TO_DATE'],CONFIG["OUT_DIR_COMPANIES"], f"{ticker}.csv")
        print(f"\n=== Harvesting company news: {ticker} ===")
        harvest_simple(label=ticker, queries=q_list, out_csv=out_csv, company=ticker)

    # # Industry topics
    # for topic, q in INDUSTRY_TOPICS.items():
    #     out_csv = os.path.join(
    #     CONFIG["FROM_DATE"],
    #     CONFIG["TO_DATE"],
    #     CONFIG["OUT_DIR_COMPANIES"],
    #     f"{ticker}.csv",
    #     )
    #     print(f"\n=== Harvesting industry topic: {topic} ===")
    #     harvest_simple(label=topic, queries=[q], out_csv=out_csv, topic=topic)

    print("\n All done. CSVs saved in data/companies and data/industry.\n")


if __name__ == "__main__":
    main()