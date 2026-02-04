import os
print("RUNNING FILE:", os.path.abspath(__file__))
import pandas as pd

OUT_PATH = "fund_positions_clean.csv"  
FUND_ID = "SGC_W2024"

POSITIONS = [
    {"symbol": "ALA.TO",  "entry_date": "2024-01-01", "entry_price": 28.31, "capital": None, "side": "long", "asset_type": "equity"},
    {"symbol": "DOL.TO",  "entry_date": "2024-01-01", "entry_price": 97.03, "capital": None, "side": "long", "asset_type": "equity"},
    {"symbol": "KILO.TO", "entry_date": "2024-01-01", "entry_price": 30.51, "capital": None, "side": "long", "asset_type": "equity"},
    {"symbol": "BABA",    "entry_date": "2024-01-01", "entry_price": 73.10, "capital": None, "side": "long", "asset_type": "equity"},
    {"symbol": "L.TO",    "entry_date": "2024-01-01", "entry_price": 32.35, "capital": None, "side": "long", "asset_type": "equity"},
    {"symbol": "WMB",     "entry_date": "2024-01-01", "entry_price": 54.13, "capital": None, "side": "long", "asset_type": "equity"},
]

def main():
    df = pd.DataFrame(POSITIONS)

    df.insert(0, "fund_id", FUND_ID)

cols = ["fund_id", "symbol", "side", "asset_type", "entry_date", "entry_price", "capital"]
df = df.reindex(columns=cols)


    tmp_path = OUT_PATH + ".tmp"
    df.to_csv(tmp_path, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(tmp_path, OUT_PATH)

    abs_path = os.path.abspath(OUT_PATH)
    print(f"Wrote {len(df)} rows to: {abs_path}")
    print("Columns written:", df.columns.tolist())

    check = pd.read_csv(OUT_PATH)
    print("Columns read back:", check.columns.tolist())

if __name__ == "__main__":
    main()
