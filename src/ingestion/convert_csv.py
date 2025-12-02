import os
import json
import pandas as pd

MACRO_JSON = "data/macro/macro.json"
OUTPUT_CSV = "data/macro/macro.csv"

def load_macro(json_path):
    with open(json_path, "r") as f:
        macro = json.load(f)

    macro_frames = []

    for key, block in macro.items():
        # Skip invalid indicators
        if "data" not in block or not isinstance(block["data"], list):
            print(f"[SKIP] {key} has no data field.")
            continue

        records = block["data"]
        interval = block.get("interval", "daily").lower()

        # Parse date/value
        dates = [pd.to_datetime(r["date"]) for r in records]
        values = [float(r["value"]) for r in records]

        df = pd.DataFrame({"date": dates, key: values}).set_index("date")

        # Resample → daily
        df = df.resample("D").ffill()

        macro_frames.append(df)

    macro_df = pd.concat(macro_frames, axis=1).sort_index()

    # ---- Apply date constraint: keep only 2022-01-01 to 2024-12-31 ----
    macro_df = macro_df.loc["2022-01-01":"2024-12-31"]

    macro_df.to_csv(OUTPUT_CSV)

    print(f"[Saved] {OUTPUT_CSV}")
    return macro_df






if __name__ == "__main__":
     load_macro(MACRO_JSON)


# import os
# import json
# import pandas as pd
# from datetime import datetime

# FUND_DIR = "data/fundamentals"
# OUT_DIR = "data/fundamentals_csv"

# os.makedirs(OUT_DIR, exist_ok=True)

# # ---- Quarter → Date 映射 ----
# def quarter_to_date(year, quarter):
#     if quarter == "Q1":
#         return datetime(year, 3, 31)
#     if quarter == "Q2":
#         return datetime(year, 6, 30)
#     if quarter == "Q3":
#         return datetime(year, 9, 30)
#     if quarter == "Q4":
#         return datetime(year, 12, 31)
#     raise ValueError(f"Invalid quarter: {quarter}")

# # ---- 转换单个公司 JSON ----
# def convert_one(symbol):
#     infile = os.path.join(FUND_DIR, f"{symbol}.json")
#     outfile = os.path.join(OUT_DIR, f"{symbol}.csv")

#     if not os.path.exists(infile):
#         print(f"[WARN] missing file: {infile}")
#         return

#     with open(infile, "r") as f:
#         data = json.load(f)

#     rows = []
#     for row in data:
#         date = quarter_to_date(row["year"], row["quarter"])
#         new_row = {"date": date}

#         for k, v in row.items():
#             if k in ["year", "quarter"]:
#                 continue
#             new_row[k] = v

#         rows.append(new_row)

#     df = pd.DataFrame(rows)
#     df = df.sort_values("date")

#     df.to_csv(outfile, index=False)
#     print(f"[Saved] {outfile}")

# # ---- 主函数：批量转换 ----
# def main():
#     symbols = [
#         f.replace(".json", "")
#         for f in os.listdir(FUND_DIR)
#         if f.endswith(".json")
#     ]

#     print("Found companies:", symbols)

#     for symbol in symbols:
#         convert_one(symbol)

#     print("\n[Done] All fundamentals converted to CSV.")

# if __name__ == "__main__":
#     main()