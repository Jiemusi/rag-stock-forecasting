import os
import pandas as pd

PRICE_DIR = "data/prices"
FUND_DIR = "data/fundamentals_csv"
MACRO_CSV = "data/macro/macro.csv"

OUTPUT_DIR = "data/company_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATE_START = "2022-01-01"
DATE_END = "2024-12-31"


def load_price(symbol):
    path = os.path.join(PRICE_DIR, f"{symbol}.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df.loc[DATE_START:DATE_END]


def load_fund(symbol):
    path = os.path.join(FUND_DIR, f"{symbol}.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    
    # forward-fill quarterly values
    df = df.resample("D").ffill()
    return df.loc[DATE_START:DATE_END]


def load_macro():
    df = pd.read_csv(MACRO_CSV, parse_dates=["date"])
    df = df.set_index("date").sort_index()

    # macro 已经 daily 但我们再截取一次日期避免 mismatch
    return df.loc[DATE_START:DATE_END]


def clean_dataset(df):
    # 删除 eps 列
    if "eps" in df.columns:
        df = df.drop(columns=["eps"])

    # forward fill
    df = df.ffill()

    # backfill
    df = df.bfill()

    return df


def build_company_dataset(symbol, macro_df):
    price = load_price(symbol)
    fund = load_fund(symbol)

    # 合并：price + fundamentals + macro
    df = price.join(fund, how="left")
    df = df.join(macro_df, how="left")

    # 加入 symbol 列
    df["symbol"] = symbol

    # 保存
    outpath = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
    df = clean_dataset(df)
    df.to_csv(outpath)

    print(f"[Saved] {outpath}")


def main():
    symbols = [
        f.replace(".csv", "")
        for f in os.listdir(PRICE_DIR)
        if f.endswith(".csv")
    ]

    macro_df = load_macro()

    print("Building datasets for:", symbols)

    for symbol in symbols:
        build_company_dataset(symbol, macro_df)

    print("\n[Done] All company datasets built!")


if __name__ == "__main__":
    main()


