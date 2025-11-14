import os
import json
import time
import requests

API_KEY = os.getenv("ALPHAVANTAGE_KEY")
BASE_URL = "https://www.alphavantage.co/query"

COMPANIES = [
    "NVDA", "TSM", "AVGO", "ASML", "AMD",
    "MSFT", "GOOGL", "AAPL", "QCOM", "TXN"
]

YEARS = [2022, 2023, 2024]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

OUTPUT_DIR = "data/earning_calls"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Safe interval: 75 calls/min → ~0.8s per call
REQUEST_DELAY = 0.85
# Retry delay if API responds with a note or empty data (rate limit)
RETRY_DELAY = 5


def request_transcript(symbol, year, quarter):
    """
    Make one API request for a single transcript.
    If Alpha Vantage rate limits, automatically retry until success or confirmed no data.
    """
    params = {
        "function": "EARNINGS_CALL_TRANSCRIPT",
        "symbol": symbol,
        "quarter": f"{year}{quarter}",
        "apikey": API_KEY
    }

    while True:
        response = requests.get(BASE_URL, params=params, timeout=20)
        time.sleep(REQUEST_DELAY)

        if response.status_code != 200:
            time.sleep(RETRY_DELAY)
            continue

        data = response.json()

        # Alpha Vantage returns "Note" or empty result when rate-limited
        if ("Note" in data) or ("Error Message" in data):
            time.sleep(RETRY_DELAY)
            continue

        # If truly no transcript, the response is an empty dict
        if data == {}:
            return None

        return data


def download_company(symbol):
    results = []

    for year in YEARS:
        for q in QUARTERS:
            print(f"{symbol} {year}{q}...", end=" ", flush=True)

            data = request_transcript(symbol, year, q)

            if data is None:
                print("no data")
            else:
                print("saved")
                results.append({
                    "symbol": symbol,
                    "year": year,
                    "quarter": q,
                    "data": data
                })

    out_path = os.path.join(OUTPUT_DIR, f"{symbol}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"{symbol}: {len(results)} transcripts written to {out_path}")


def main():
    if not API_KEY:
        print("ALPHAVANTAGE_KEY environment variable not set.")
        return

    for symbol in COMPANIES:
        print(f"\nFetching transcripts for {symbol}")
        download_company(symbol)


if __name__ == "__main__":
    main()
