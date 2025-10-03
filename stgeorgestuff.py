import yfinance as yf
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional, Tuple


class CSVTradeLogger:
    def __init__(self, csv_path: str, tickers: List[str]):
        self.csv_path = csv_path
        self.tickers = tickers
        self._ensure_csv()

    def _ensure_csv(self):
        if not os.path.exists(self.csv_path):
            cols = ["timestamp","ticker","close","action","quantity","position_after","cash_after","note"]
            pd.DataFrame(columns=cols).to_csv(self.csv_path, index=False)

    def backfill_history(self, start: str, end: Optional[str] = None, interval: str = "1d",
                         note: str = "history backfill"):
        data = yf.download(self.tickers, start=start, end=end, interval=interval,
                           group_by='ticker', auto_adjust=False, progress=False)
        rows = []
        for t in self.tickers:
            df = data[t] if t in data else data
            if isinstance(df, pd.DataFrame) and "Close" in df.columns:
                for ts, close in df["Close"].dropna().items():
                    rows.append({
                        "timestamp": pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
                        "ticker": t,
                        "close": float(close),
                        "action": "NONE",
                        "quantity": 0,
                        "position_after": None,
                        "cash_after": None,
                        "note": note
                    })
        if rows:
            pd.DataFrame(rows).to_csv(self.csv_path, mode='a',
                                      header=not os.path.getsize(self.csv_path), index=False)
            self._dedup_csv()  # <-- auto-dedup after backfill

    def log_now(self, prices: Dict[str, float], action_map: Dict[str, Dict],
                positions_after: Dict[str, int], cash_after: float, note: str = ""):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = []
    
        for t in self.tickers:
            act = action_map.get(t, {"action":"NONE","quantity":0})
            rows.append({
                "timestamp": ts,
                "ticker": t,
                "close": float(prices.get(t, float("nan"))),
                "action": act["action"],
                "quantity": int(act["quantity"]),
                "position_after": int(positions_after.get(t, 0)),
                "cash_after": float(cash_after),
                "note": note
            })
        pd.DataFrame(rows).to_csv(self.csv_path, mode='a',
                                  header=not os.path.getsize(self.csv_path), index=False)
        self._dedup_csv()  # <-- auto-dedup after logging

    # ---- auto-backfill machinery ----
    def _last_logged_day(self) -> Optional[datetime]:
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            return None
        try:
            df = pd.read_csv(self.csv_path, parse_dates=["timestamp"])
            if df.empty:
                return None
            return df["timestamp"].dt.normalize().max().to_pydatetime()
        except Exception:
            return None

    def _dedup_csv(self):
        try:
            df = pd.read_csv(self.csv_path, parse_dates=["timestamp"])
            df = df.drop_duplicates(subset=["timestamp","ticker"], keep="last")
            df.to_csv(self.csv_path, index=False)
        except Exception:
            pass

    def autobackfill_on_start(self, default_lookback_days: int = 365):
        today = datetime.now().date()
        last_day = self._last_logged_day()
        start_dt = (datetime.now() - timedelta(days=default_lookback_days)) if last_day is None else (last_day + timedelta(days=1))
        if start_dt.date() > today:
            self._dedup_csv()
            return
        self.backfill_history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=(today + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            note="auto backfill on start"
        )
        self._dedup_csv()
        