import os
import time
import requests
import pandas as pd

API_KEY = os.getenv("ALPHAVANTAGE_KEY")
BASE_URL = "https://www.alphavantage.co/query"

OUTPUT_DIR = "data/prices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMPANIES = [
    "NVDA", "TSM", "AVGO", "ASML", "AMD",
    "MSFT", "GOOGL", "AAPL", "QCOM", "TXN"
]

START_DATE = "2022-01-01"
END_DATE   = "2024-12-31"


def fetch_prices(symbol):
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "apikey": API_KEY,
        "outputsize": "full"
    }

    print(f"Fetching {symbol}...")

    r = requests.get(BASE_URL, params=params)
    if r.status_code != 200:
        print(f"HTTP {r.status_code} for {symbol}")
        return

    data = r.json()
    if "Time Series (Daily)" not in data:
        print(f"No price data found for {symbol}")
        return

    ts = data["Time Series (Daily)"]

    rows = []
    for date, values in ts.items():
        if date < START_DATE or date > END_DATE:
            continue

        rows.append({
            "date": date,
            "open": float(values["1. open"]),
            "high": float(values["2. high"]),
            "low": float(values["3. low"]),
            "close": float(values["4. close"]),
            "adjusted_close": float(values["5. adjusted close"]),
            "volume": int(values["6. volume"]),
            "dividend_amount": float(values["7. dividend amount"]),
            "split_coefficient": float(values["8. split coefficient"])
        })

    if not rows:
        print(f"No rows for {symbol} in date range")
        return

    df = pd.DataFrame(rows).sort_values("date")
    df.to_csv(f"{OUTPUT_DIR}/{symbol}.csv", index=False)

    print(f"Saved {symbol}: {len(df)} rows")


def main():
    for symbol in COMPANIES:
        fetch_prices(symbol)
        time.sleep(1)


if __name__ == "__main__":
    main()
