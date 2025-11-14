import os
import json
import time
import requests

API_KEY = os.getenv("ALPHAVANTAGE_KEY")
BASE_URL = "https://www.alphavantage.co/query"

OUTPUT_PATH = "data/macro/macro.json"
os.makedirs("data/macro", exist_ok=True)

MACRO_FUNCTIONS = {
    "real_gdp": "REAL_GDP",
    "real_gdp_per_capita": "REAL_GDP_PER_CAPITA",
    "cpi": "CPI",
    "inflation": "INFLATION",
    "unemployment": "UNEMPLOYMENT",
    "federal_funds_rate": "FEDERAL_FUNDS_RATE",
    "consumer_sentiment": "CONSUMER_SENTIMENT",
    "retail_sales": "RETAIL_SALES",
    "industrial_production": "INDUSTRIAL_PRODUCTION",
    "treasury_yield_10y": "TREASURY_YIELD"
}

def fetch_macro_indicator(name: str, function: str):
    """Fetch a single macro series."""
    url = f"{BASE_URL}?function={function}&apikey={API_KEY}"
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    macro_data = {}

    for name, function in MACRO_FUNCTIONS.items():
        print(f"Fetching {name} ...")
        data = fetch_macro_indicator(name, function)

        if data:
            macro_data[name] = data
        else:
            macro_data[name] = None

        time.sleep(0.8)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(macro_data, f, indent=2)

    print(f"Saved macro fundamentals → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
