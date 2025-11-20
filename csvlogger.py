import yfinance as yf
import pandas as pd
import os, tempfile, hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

class CSVTradeLogger:
    BASE_COLS = ["timestamp","ticker","close","action","quantity",
                 "position_after","cash_after","note"]
    EXTRA_COLS = ["event_id","kind","price_source","cash_delta",
                  "position_before","out_of_order"]
    SCHEMA = BASE_COLS + EXTRA_COLS

    def __init__(self, csv_path: str, tickers: List[str]):
        self.csv_path = csv_path
        self.tickers = self._validate_tickers(tickers)
        self._ensure_csv()

        # for manual backfill
        self._last_backfill_ts = None       # prevents spam clicking
        self._cooldown_seconds = 60         # limit: 1 request per minute

    # ------------------------------------------------------------
    # CSV + Schema Handling
    # ------------------------------------------------------------

    def _ensure_csv(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=self.SCHEMA).to_csv(
                self.csv_path, index=False, encoding="utf-8", lineterminator="\n"
            )
        else:
            try:
                df = pd.read_csv(self.csv_path, dtype=str)
            except Exception:
                df = pd.DataFrame(columns=self.SCHEMA)

            for c in self.SCHEMA:
                if c not in df.columns:
                    df[c] = None

            df = df[self.SCHEMA]
            tmp = self.csv_path + ".tmp"
            df.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
            os.replace(tmp, self.csv_path)

    # ------------------------------------------------------------
    # Manual Backfill 
    # ------------------------------------------------------------

    def manual_backfill(self, default_lookback_days: int = 365,
                        interval: str = "1d",
                        note: str = "manual backfill") -> bool:
        """
        Triggered by dashboard button.
        Backfills ONLY missing dates since last_logged_day().
        Has cooldown to avoid yfinance rate limits.
        Returns True if new data was fetched, False otherwise.
        """

        now = datetime.now(timezone.utc)

        # Prevent spam-pressing
        if self._last_backfill_ts and \
           (now - self._last_backfill_ts).total_seconds() < self._cooldown_seconds:
            return False

        today = now.date()
        last_day = self._last_logged_day()

        # If no history exists, fetch up to 1 year
        if last_day is None:
            start_dt = today - timedelta(days=default_lookback_days)
        else:
            start_dt = (last_day + timedelta(days=1)).date()

        if start_dt > today:
            return False

        # Use existing backfill method
        self.backfill_history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=(today + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval=interval,
            note=note
        )

        self._last_backfill_ts = now
        return True

    # ------------------------------------------------------------
    # History Fetching
    # ------------------------------------------------------------

    def backfill_history(self, start: str, end: Optional[str] = None,
                         interval: str = "1d", note: str = "history backfill"):

        data = yf.download(
            self.tickers,
            start=start,
            end=end,
            interval=interval,
            group_by='ticker',
            auto_adjust=False,
            progress=False
        )

        rows = []
        for t in self.tickers:
            df = data[t] if t in getattr(data, "keys", lambda: [])() else data
            if isinstance(df, pd.DataFrame) and "Close" in df.columns:
                for ts, close in df["Close"].dropna().items():
                    self._assert_price(close, t, "backfill")

                    timestamp = (pd.Timestamp(ts)
                                 .tz_localize(None)
                                 .to_pydatetime()
                                 .replace(tzinfo=timezone.utc)
                                 .strftime("%Y-%m-%dT%H:%M:%SZ"))

                    row = {
                        "timestamp": timestamp,
                        "ticker": t,
                        "close": float(close),
                        "action": "NONE",
                        "quantity": 0,
                        "position_after": None,
                        "cash_after": None,
                        "note": note,
                        "kind": "HISTORY",
                        "price_source": "yfinance",
                        "cash_delta": 0.0,
                        "position_before": None,
                        "out_of_order": False
                    }
                    row["event_id"] = self._event_id(row)
                    rows.append(row)

        if rows:
            self._append_rows_atomic(rows)
            self._dedup_csv()

    # ------------------------------------------------------------
    # Realtime Logging
    # ------------------------------------------------------------

    def log_now(self, prices: Dict[str, float], action_map: Dict[str, Dict],
                positions_after: Dict[str, int], cash_after: float, note: str = ""):

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._assert_cash(cash_after)
        rows = []

        last_map = self._last_ts_per_ticker()

        for t in self.tickers:
            act = action_map.get(t, {"action": "NONE", "quantity": 0})
            action = str(act.get("action", "NONE")).upper()
            qty = int(act.get("quantity", 0))
            pos_after = int(positions_after.get(t, 0))
            price = prices.get(t, float("nan"))

            self._assert_action(action, t)
            self._assert_quantity(qty, t, action)
            self._assert_position(pos_after, t, action, qty)
            self._assert_price(price, t, "log_now")

            pos_before = (
                pos_after - qty if action == "BUY" else
                pos_after + qty if action == "SELL" else
                pos_after
            )

            cash_delta = (
                -qty * float(price) if action == "BUY"
                else qty * float(price) if action == "SELL"
                else 0.0
            )

            out_of_order = False
            try:
                last_ts = last_map.get(t)
                out_of_order = (last_ts is not None and ts < last_ts)
            except Exception:
                out_of_order = False

            row = {
                "timestamp": ts,
                "ticker": t,
                "close": float(price),
                "action": action,
                "quantity": qty,
                "position_after": pos_after,
                "cash_after": float(cash_after),
                "note": self._sanitize_note(note),
                "kind": "SNAPSHOT" if action=="NONE" else "TRADE",
                "price_source": "user",
                "cash_delta": float(cash_delta),
                "position_before": int(pos_before),
                "out_of_order": bool(out_of_order)
            }
            row["event_id"] = self._event_id(row)
            rows.append(row)

        self._append_rows_atomic(rows)
        self._dedup_csv()

    # ------------------------------------------------------------
    # Utilities (unchanged)
    # ------------------------------------------------------------

    def _last_logged_day(self) -> Optional[datetime]:
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            return None
        try:
            df = pd.read_csv(self.csv_path, parse_dates=["timestamp"])
            if df.empty:
                return None
            return (df["timestamp"]
                    .dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
                    .dt.normalize()
                    .max()
                    .to_pydatetime())
        except Exception:
            return None

    def _dedup_csv(self):
        try:
            df = pd.read_csv(self.csv_path, dtype={"timestamp": str})
            for c in self.SCHEMA:
                if c not in df.columns:
                    df[c] = None
            if "event_id" in df.columns:
                df = df.drop_duplicates(subset=["event_id"], keep="last")
            else:
                df = df.drop_duplicates(
                    subset=["timestamp","ticker","action","quantity","close"],
                    keep="last"
                )
            df = df[self.SCHEMA]
            tmp = self.csv_path + ".tmp"
            df.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
            os.replace(tmp, self.csv_path)
        except Exception:
            pass

    def _validate_tickers(self, tickers: List[str]) -> List[str]:
        if not isinstance(tickers, list) or not tickers:
            raise ValueError("tickers must be a non-empty list.")
        clean = []
        for t in tickers:
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"Invalid ticker: {t!r}")
            clean.append(t.strip().upper())
        uniq = []
        seen = set()
        for t in clean:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        return uniq

    def _assert_action(self, action: str, ticker: str) -> None:
        if action not in {"BUY","SELL","NONE"}:
            raise ValueError(f"[{ticker}] invalid action: {action}")

    def _assert_quantity(self, qty: int, ticker: str, action: str) -> None:
        if not isinstance(qty, int):
            raise ValueError(f"[{ticker}] qty must be int")
        if action=="NONE" and qty != 0:
            raise ValueError(f"[{ticker}] qty must be 0 for NONE")
        if action in {"BUY","SELL"} and qty <= 0:
            raise ValueError(f"[{ticker}] qty must be >0 for {action}")

    def _assert_position(self, pos_after: int, ticker: str, action: str, qty: int) -> None:
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
            raise ValueError(f"[{ticker}] invalid price in {context}")
        if p <= 0:
            raise ValueError(f"[{ticker}] price must be > 0 in {context}")

    def _assert_cash(self, cash_after: float) -> None:
        try:
            c = float(cash_after)
        except Exception:
            raise ValueError("cash_after must be numeric")
        if c < 0:
            raise ValueError("cash_after cannot be negative")

    def _sanitize_note(self, s: str) -> str:
        s = s or ""
        return "'" + s if s[:1] in ("=", "+", "-", "@") else s

    def _event_id(self, row: Dict) -> str:
        key = f"{row['timestamp']}|{row['ticker']}|{row['action']}|{row['quantity']}|{row['close']}|{row.get('note','')}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def _append_rows_atomic(self, rows: List[Dict]) -> None:
        if not rows:
            return
        df_new = pd.DataFrame(rows)
        for c in self.SCHEMA:
            if c not in df_new.columns:
                df_new[c] = None
        try:
            df_old = pd.read_csv(self.csv_path, dtype=str)
        except Exception:
            df_old = pd.DataFrame(columns=self.SCHEMA)
        for c in self.SCHEMA:
            if c not in df_old.columns:
                df_old[c] = None
        df_old = df_old[self.SCHEMA]
        df_all = pd.concat([df_old, df_new[self.SCHEMA]], ignore_index=True)
        if "event_id" in df_all.columns:
            df_all = df_all.drop_duplicates(subset=["event_id"], keep="last")
        else:
            df_all = df_all.drop_duplicates(
                subset=["timestamp","ticker","action","quantity","close"],
                keep="last"
            )
        tmp = self.csv_path + ".tmp"
        df_all.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
        os.replace(tmp, self.csv_path)

    def _last_ts_per_ticker(self) -> Dict[str, str]:
        try:
            df = pd.read_csv(self.csv_path, dtype=str, usecols=["ticker","timestamp"])
            df = df.dropna()
            df = df.sort_values(["ticker","timestamp"])
            return df.groupby("ticker")["timestamp"].max().to_dict()
        except Exception:
            return {}