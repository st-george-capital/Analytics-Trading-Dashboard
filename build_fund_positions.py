import pandas as pd

OUT_PATH = "fund_positions.csv"
FUND_ID = "SGC_W2024"

# Fill these in manually.
# change capital once known quantity
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

    cols = ["fund_id", "symbol", "side", "asset_type", "entry_date", "entry_price", "capital", "thesis"]
    df = df[cols]

    df.to_csv(OUT_PATH, index=False, encoding="utf-8", lineterminator="\n")
    print(f"Wrote {len(df)} rows to {OUT_PATH}")

if __name__ == "__main__":
    main()
