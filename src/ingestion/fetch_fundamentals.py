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

OUTPUT_DIR = "data/fundamentals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REQUEST_DELAY = 0.85
RETRY_DELAY = 5 


# Utility: Safe API call with retry

def fetch_alpha_vantage(function, symbol):
    """
    Generic function to retrieve data from Alpha Vantage.
    Automatically retries when rate-limited.
    """
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": API_KEY
    }

    while True:
        response = requests.get(BASE_URL, params=params, timeout=20)
        time.sleep(REQUEST_DELAY)

        if response.status_code != 200:
            time.sleep(RETRY_DELAY)
            continue

        data = response.json()

        if ("Note" in data) or ("Error Message" in data):
            time.sleep(RETRY_DELAY)
            continue

        # If AV returns an empty dict, accept it
        return data


# Clean extraction of quarterly records

def index_by_quarter(raw_list):
    """
    Convert Alpha Vantage quarterly data list into:
    { "2022Q1": {...}, "2022Q2": {...}, ... }
    """
    result = {}
    if not isinstance(raw_list, list):
        return result

    for entry in raw_list:
        fiscal_date = entry.get("fiscalDateEnding")  # e.g., "2022-03-31"
        if not fiscal_date:
            continue

        year = fiscal_date[:4]
        month = int(fiscal_date[5:7])

        if month in (1, 2, 3):
            quarter = "Q1"
        elif month in (4, 5, 6):
            quarter = "Q2"
        elif month in (7, 8, 9):
            quarter = "Q3"
        else:
            quarter = "Q4"

        key = f"{year}{quarter}"
        result[key] = entry

    return result


# Extract selected metrics for a given quarter

def extract_metrics(is_data, bs_data, cf_data):
    """
    Extract the minimal set of fundamentals for one quarter.
    Each input is a dict representing that quarter or None.
    """
    if is_data is None and bs_data is None and cf_data is None:
        return None

    def g(d, k):
        if d is None:
            return None
        try:
            return float(d.get(k))
        except:
            return None

    return {
        "revenue": g(is_data, "totalRevenue"),
        "net_income": g(is_data, "netIncome"),
        "eps": g(is_data, "reportedEPS"),
        "operating_income": g(is_data, "operatingIncome"),
        "gross_profit": g(is_data, "grossProfit"),

        "total_assets": g(bs_data, "totalAssets"),
        "total_liabilities": g(bs_data, "totalLiabilities"),
        "long_term_debt": g(bs_data, "longTermDebt"),
        "cash_and_equivalents": g(bs_data, "cashAndCashEquivalentsAtCarryingValue"),

        "operating_cashflow": g(cf_data, "operatingCashflow"),
        "capital_expenditures": g(cf_data, "capitalExpenditures")
    }


# Download fundamentals for one company

def download_company(symbol):
    print(f"\nFetching fundamentals for {symbol}")

    income_raw = fetch_alpha_vantage("INCOME_STATEMENT", symbol)
    balance_raw = fetch_alpha_vantage("BALANCE_SHEET", symbol)
    cashflow_raw = fetch_alpha_vantage("CASH_FLOW", symbol)

    income_q = index_by_quarter(income_raw.get("quarterlyReports"))
    balance_q = index_by_quarter(balance_raw.get("quarterlyReports"))
    cashflow_q = index_by_quarter(cashflow_raw.get("quarterlyReports"))

    results = []

    for year in YEARS:
        for q in QUARTERS:
            key = f"{year}{q}"
            print(f"{symbol} {key}...", end=" ", flush=True)

            is_data = income_q.get(key)
            bs_data = balance_q.get(key)
            cf_data = cashflow_q.get(key)

            metrics = extract_metrics(is_data, bs_data, cf_data)

            if metrics is None:
                print("no data")
                continue

            print("saved")

            results.append({
                "symbol": symbol,
                "year": year,
                "quarter": q,
                **metrics
            })

    out_path = os.path.join(OUTPUT_DIR, f"{symbol}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"{symbol}: {len(results)} quarterly records written to {out_path}")


def main():
    if not API_KEY:
        print("ALPHAVANTAGE_KEY environment variable not set.")
        return

    for symbol in COMPANIES:
        download_company(symbol)


if __name__ == "__main__":
    main()
