import os
import json
import time
from io import StringIO
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import requests
from pandas.errors import EmptyDataError


class StockDailyPricesBuilder:
    CORE_COLS = [
        "date", "symbol",
        "open", "high", "low", "close", "adjusted_close",
        "volume", "dividend_amount", "split_coefficient",
    ]

    def __init__(
        self,
        csv_path: str = "stock_daily_prices.csv",
        state_path: str = "stock_daily_prices_state.json",
        api_key: str = "",
        outputsize: str = "compact",   # "compact" or "full"
        max_calls_per_day: int = 75,
        min_seconds_between_calls: float = 1.2,
        raw_dir: str = "av_raw_stock",
        verbose: bool = True,
    ):
        self.csv_path = csv_path
        self.state_path = state_path
        self.api_key = api_key
        self.outputsize = outputsize
        self.max_calls_per_day = int(max_calls_per_day)
        self.min_seconds_between_calls = float(min_seconds_between_calls)
        self.raw_dir = raw_dir
        self.verbose = bool(verbose)
        self._last_call_ts: Optional[float] = None
        self._ensure_files()

    def update(self, symbols: List[str], dry_run: bool = False) -> bool:
        api_key = self.api_key or os.getenv("ALPHAVANTAGE_API_KEY", "")
        if not api_key and not dry_run:
            raise ValueError("Missing Alpha Vantage API key (api_key=... or ALPHAVANTAGE_API_KEY).")

        state = self._load_state()
        wrote_any = False

        for sym in symbols:
            if dry_run:
                print(f"[DRY RUN] Would fetch DAILY_ADJUSTED for {sym}")
                continue

            if not self._can_call_today(state):
                print("[STOP] Daily call cap reached.")
                break

            df, meta = self._fetch_daily_adjusted(sym, api_key, state)

            if df is None or df.empty:
                self._note(state, sym, meta)
                self._save_state(state)
                if self.verbose:
                    print(f"[NO DATA] {sym}: {meta.get('reason','unknown')}")
                self._throttle()
                continue

            self._append_and_dedup(df)
            self._mark_done(state, sym, wrote_rows=len(df))
            self._save_state(state)

            wrote_any = True
            if self.verbose:
                print(f"[OK] {sym}: wrote {len(df)} rows")

            self._throttle()

        return wrote_any


    def _fetch_daily_adjusted(
        self,
        symbol: str,
        api_key: str,
        state: Dict[str, Any],
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": self.outputsize,
            "apikey": api_key,
            "datatype": "csv",
        }

        meta: Dict[str, Any] = {"symbol": symbol, "reason": "", "http_status": None}

        try:
            r = requests.get(url, params=params, timeout=60)
            meta["http_status"] = r.status_code
            self._count_call_in_state(state)

            text = (r.text or "").strip()

            if r.status_code != 200:
                meta["reason"] = f"http_{r.status_code}"
                self._save_raw(symbol, text, suffix=meta["reason"])
                return None, meta

            if not text:
                meta["reason"] = "empty_response"
                self._save_raw(symbol, text, suffix=meta["reason"])
                return None, meta

            if text.lstrip().startswith("{"):
                try:
                    j = json.loads(text)
                    meta["reason"] = j.get("Note") or j.get("Error Message") or j.get("Information") or "json_message"
                except Exception:
                    meta["reason"] = "json_unparseable"
                self._save_raw(symbol, text, suffix="json_message")
                return None, meta

            df = pd.read_csv(StringIO(text))
            if df.empty:
                meta["reason"] = "csv_empty"
                self._save_raw(symbol, text, suffix=meta["reason"])
                return None, meta

            if "timestamp" in df.columns:
                df = df.rename(columns={"timestamp": "date"})
            df["symbol"] = symbol

            for c in self.CORE_COLS:
                if c not in df.columns:
                    df[c] = pd.NA
            df = df[self.CORE_COLS]

            return df, meta

        except requests.exceptions.RequestException as e:
            meta["reason"] = f"request_exception:{type(e).__name__}"
            self._save_raw(symbol, str(e), suffix=meta["reason"])
            return None, meta

    # csv

    def _append_and_dedup(self, df_new: pd.DataFrame) -> None:
        df_old = self._safe_read_csv(self.csv_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True) if not df_old.empty else df_new.copy()

        df_all = df_all.drop_duplicates(subset=["symbol", "date"], keep="last")
        df_all.to_csv(self.csv_path, index=False, encoding="utf-8", lineterminator="\n")

    def _safe_read_csv(self, path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except (FileNotFoundError, EmptyDataError):
            return pd.DataFrame(columns=self.CORE_COLS)

    def _ensure_files(self) -> None:
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=self.CORE_COLS).to_csv(self.csv_path, index=False, encoding="utf-8", lineterminator="\n")
        if not os.path.exists(self.state_path):
            self._save_state({"calls": {}, "done_meta": {}, "notes": {}})
        os.makedirs(self.raw_dir, exist_ok=True)

    def _load_state(self) -> Dict[str, Any]:
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"calls": {}, "done_meta": {}, "notes": {}}

    def _save_state(self, state: Dict[str, Any]) -> None:
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def _mark_done(self, state: Dict[str, Any], symbol: str, wrote_rows: int) -> None:
        state.setdefault("done_meta", {})[symbol] = {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wrote_rows": int(wrote_rows),
            "outputsize": self.outputsize,
        }

    def _note(self, state: Dict[str, Any], symbol: str, meta: Dict[str, Any]) -> None:
        state.setdefault("notes", {})[symbol] = {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "meta": meta,
        }

    def _can_call_today(self, state: Dict[str, Any]) -> bool:
        today = datetime.now(timezone.utc).date().isoformat()
        used = int(state.get("calls", {}).get(today, 0))
        return used < self.max_calls_per_day

    def _count_call_in_state(self, state: Dict[str, Any]) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        state.setdefault("calls", {})
        state["calls"][today] = int(state["calls"].get(today, 0)) + 1

    # raw

    def _save_raw(self, symbol: str, content: str, suffix: str = "raw") -> None:
        safe_sym = symbol.replace("/", "_").replace(":", "_")
        path = os.path.join(self.raw_dir, f"{safe_sym}_{suffix}.txt")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content if content is not None else "")
        except Exception:
            pass

    def _throttle(self) -> None:
        now = time.time()
        if self._last_call_ts is None:
            self._last_call_ts = now
            return
        dt = now - self._last_call_ts
        if dt < self.min_seconds_between_calls:
            time.sleep(self.min_seconds_between_calls - dt)
        self._last_call_ts = time.time()


if __name__ == "__main__":
    symbols = ["ALA.TO", "DOL.TO", "KILO.TO", "BABA", "L.TO", "WMB"]

    builder = StockDailyPricesBuilder(
        csv_path="stock_daily_prices.csv",
        state_path="stock_daily_prices_state.json",
        api_key="",         # or set ALPHAVANTAGE_API_KEY
        outputsize="full",  # first time backfill
        verbose=True,
    )

    ok = builder.update(symbols, dry_run=False)
    print("Updated any:", ok)
