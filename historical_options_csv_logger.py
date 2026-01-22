import os
import json
import time
from datetime import datetime, timezone, date
from typing import List, Optional, Dict

import pandas as pd
import requests


class OptionsChainLogger:
    def __init__(
        self,
        csv_path: str = "options_chain_history.csv",
        state_path: str = "options_chain_state.json",
        api_key: str = "",
        max_calls_per_day: int = 75,
        min_seconds_between_calls: float = 1.2,
    ):
        self.csv_path = csv_path
        self.state_path = state_path
        self.api_key = api_key  # set this explicitly OR use env var below
        self.max_calls_per_day = int(max_calls_per_day)
        self.min_seconds_between_calls = float(min_seconds_between_calls)

        self._last_call_ts: Optional[float] = None
        self._ensure_files()


    def update_today(self, symbols: List[str]) -> bool:
        """Fetch and append today's options chain for each symbol (weekdays only)."""
        d = datetime.now(timezone.utc).date()
        return self.update_for_date(symbols, d)

    def update_for_date(self, symbols: List[str], d: date) -> bool:
        """
        Fetch and append options chains for a given date (weekdays only).
        Returns True if anything new was written.
        """
        if not self._is_weekday(d):
            return False

        api_key = self.api_key or os.getenv("ALPHAVANTAGE_API_KEY", "") #implement either explicitly or locally
        if not api_key:
            raise ValueError("Missing Alpha Vantage API key. Set api_key=... or ALPHAVANTAGE_API_KEY env var.")

        state = self._load_state()
        day_key = d.isoformat()

        wrote_any = False
        for sym in symbols:
            if not self._can_call_today(state):
                return wrote_any

            already_done = state.get("done", {}).get(day_key, {}).get(sym, False)
            if already_done:
                continue

            df = self._fetch_historical_options(sym, day_key, api_key)
            if df is None or df.empty:
                self._mark_done(state, day_key, sym, wrote=False)
                self._save_state(state)
                continue

            df["asof_date"] = day_key
            df["underlying"] = sym

            self._append_and_dedup(df)

            self._mark_done(state, day_key, sym, wrote=True)
            self._save_state(state)
            wrote_any = True

            self._throttle()

        return wrote_any

#alpha vantage 
    def _fetch_historical_options(self, symbol: str, day_iso: str, api_key: str) -> pd.DataFrame:
        """
        Calls Alpha Vantage HISTORICAL_OPTIONS for symbol+date.
        Uses datatype=csv to get a flat table directly.
        """
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "HISTORICAL_OPTIONS",
            "symbol": symbol,
            "date": day_iso,
            "apikey": api_key,
            "datatype": "csv",
        }

        r = requests.get(url, params=params, timeout=60)
        self._count_call()
        r.raise_for_status()

        text = r.text.strip()
        if not text or text.lower().startswith("{"):
            return pd.DataFrame()

        from io import StringIO
        try:
            df = pd.read_csv(StringIO(text))
        except Exception:
            return pd.DataFrame()

        return df

#storing
    def _append_and_dedup(self, df_new: pd.DataFrame) -> None:
        if os.path.exists(self.csv_path):
            df_old = pd.read_csv(self.csv_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new

        # DEDUP KEY (adjust column names if AlphaVantage uses different headers)
        key_cols = [c for c in ["asof_date", "underlying", "expiration", "strike", "type"] if c in df_all.columns]
        if key_cols:
            df_all = df_all.drop_duplicates(subset=key_cols, keep="last")

        df_all.to_csv(self.csv_path, index=False, encoding="utf-8", lineterminator="\n")

#rate limiters
    def _ensure_files(self) -> None:
        if not os.path.exists(self.state_path):
            self._save_state({"calls": {}, "done": {}, "last_call_ts": None})
        if not os.path.exists(self.csv_path):
            pd.DataFrame().to_csv(self.csv_path, index=False)

    def _load_state(self) -> Dict:
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"calls": {}, "done": {}, "last_call_ts": None}

    def _save_state(self, state: Dict) -> None:
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def _mark_done(self, state: Dict, day_iso: str, symbol: str, wrote: bool) -> None:
        state.setdefault("done", {}).setdefault(day_iso, {})[symbol] = True
        state.setdefault("done_meta", {}).setdefault(day_iso, {})[symbol] = {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wrote_rows": bool(wrote),
        }

    def _can_call_today(self, state: Dict) -> bool:
        today = datetime.now(timezone.utc).date().isoformat()
        used = int(state.get("calls", {}).get(today, 0))
        return used < self.max_calls_per_day

    def _count_call(self) -> None:
        state = self._load_state()
        today = datetime.now(timezone.utc).date().isoformat()
        state.setdefault("calls", {})
        state["calls"][today] = int(state["calls"].get(today, 0)) + 1
        self._save_state(state)

    def _throttle(self) -> None:
        now = time.time()
        if self._last_call_ts is None:
            self._last_call_ts = now
            return
        dt = now - self._last_call_ts
        if dt < self.min_seconds_between_calls:
            time.sleep(self.min_seconds_between_calls - dt)
        self._last_call_ts = time.time()

    @staticmethod
    def _is_weekday(d: date) -> bool:
        return d.weekday() < 5


if __name__ == "__main__":
    logger = OptionsChainLogger(
        csv_path="options_chain_history.csv",
        state_path="options_chain_state.json",
        api_key="",  # <-- put key here OR set env var ALPHAVANTAGE_API_KEY
        max_calls_per_day=75,
        min_seconds_between_calls=1.2,
    )

    symbols = ["BABA", "DOL.TO"]
    ok = logger.update_today(symbols)
    print("Updated:", ok)
