import os
import json
import time
import requests

API_KEY = os.getenv("ALPHAVANTAGE_KEY")

TICKERS = [
    "AAPL", "AMD", "ASML", "AVGO", 
    "GOOGL", "MSFT", "NVDA", 
    "QCOM", "TSM", "TXN"
]

OUTPUT_DIR = "data/prices"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_daily_adjusted(symbol):
    print(f"\nFetching {symbol} ...")

    url = (
        "https://www.alphavantage.co/query?"
        f"function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}"
        f"&outputsize=full&apikey={API_KEY}"
    )

    r = requests.get(url)
    data = r.json()

    if "Time Series (Daily)" not in data:
        print(f"Failed for {symbol}: {data}")
        return None

    ts = data["Time Series (Daily)"]

   
    cleaned = []
    for date, values in ts.items():
        cleaned.append({
            "date": date,
            "open": float(values["1. open"]),
            "high": float(values["2. high"]),
            "low": float(values["3. low"]),
            "close": float(values["4. close"]),
            "adjusted_close": float(values["5. adjusted close"]),
            "volume": int(values["6. volume"])
        })

    cleaned_sorted = sorted(cleaned, key=lambda x: x["date"])

    return cleaned_sorted


def save_json(symbol, data):
    path = os.path.join(OUTPUT_DIR, f"{symbol}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved → {path}")


def main():
    for ticker in TICKERS:
        data = fetch_daily_adjusted(ticker)
        if data:
            save_json(ticker, data)
        time.sleep(12)  # AlphaVantage limit: 5 calls/min → sleep 12s

    print("\nAll done!")


if __name__ == "__main__":
    main()
