import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class CSVTradeLogger:
    def __init__(self, csv_path: str, tickers: List[str]):
        self.csv_path = csv_path
        self.tickers = self._validate_tickers(tickers)
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
                    self._assert_price(close, t, "backfill")
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
        self._assert_cash(cash_after)
        rows = []
        for t in self.tickers:
            act = action_map.get(t, {"action":"NONE","quantity":0})
            action = str(act.get("action","NONE")).upper()
            qty = int(act.get("quantity",0))
            pos_after = int(positions_after.get(t, 0))
            price = prices.get(t, float("nan"))
            self._assert_action(action, t)
            self._assert_quantity(qty, t)
            self._assert_position(pos_after, t)
            self._assert_price(price, t, "log_now")
            rows.append({
                "timestamp": ts,
                "ticker": t,
                "close": float(price),
                "action": action,
                "quantity": qty,
                "position_after": pos_after,
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

    def _validate_tickers(self, tickers: List[str]) -> List[str]:
        if not isinstance(tickers, list) or not tickers:
            raise ValueError("tickers must be a non-empty list of symbols.")
        clean = []
        for t in tickers:
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"Invalid ticker: {t!r}")
            clean.append(t.strip().upper())
        seen = set()
        uniq = []
        for t in clean:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        return uniq

    def _assert_action(self, action: str, ticker: str) -> None:
        if action not in {"BUY","SELL","NONE"}:
            raise ValueError(f"[{ticker}] invalid action: {action}")

    def _assert_quantity(self, qty: int, ticker: str) -> None:
        if not isinstance(qty, int):
            raise ValueError(f"[{ticker}] quantity must be int")
        if qty < 0:
            raise ValueError(f"[{ticker}] quantity cannot be negative")

    def _assert_position(self, pos_after: int, ticker: str) -> None:
        if not isinstance(pos_after, int):
            raise ValueError(f"[{ticker}] position_after must be int")
        if pos_after < 0:
            raise ValueError(f"[{ticker}] position_after cannot be negative")

    def _assert_price(self, price: float, ticker: str, context: str) -> None:
        if pd.isna(price):
            raise ValueError(f"[{ticker}] price is NaN in {context}")
        try:
            p = float(price)
        except Exception:
            raise ValueError(f"[{ticker}] price must be numeric in {context}")
        if p < 0:
            raise ValueError(f"[{ticker}] price cannot be negative in {context}")

    def _assert_cash(self, cash_after: float) -> None:
        try:
            c = float(cash_after)
        except Exception:
            raise ValueError("cash_after must be numeric")
        if c < 0:
            raise ValueError("cash_after cannot be negative")