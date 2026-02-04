import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

#configs to change if csv files change

DEFAULT_POSITIONS_CSV_CANDIDATES = ["fund_positions_clean.csv", "fund_positions.csv"]
DEFAULT_PRICES_CSV = "stock_daily_prices.csv"

OUT_TIMESERIES_CSV = "fund_daily_timeseries.csv"
OUT_METRICS_JSON = "fund_metrics.json"

TRADING_DAYS_PER_YEAR = 252


#helpers
def _find_existing(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _coerce_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _pick_price_column(df: pd.DataFrame) -> str:
    """
    Try to find the best daily price column.
    Common candidates across data sources: adjusted_close, adj_close, close, price, last
    """
    candidates = [
        "adjusted_close",
        "adj_close",
        "adjClose",
        "adjclose",
        "close",
        "Close",
        "price",
        "last",
        "settle",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a usable price column in stock_daily_prices.csv. "
        f"Expected one of: {candidates}. Found: {list(df.columns)}"
    )


def _ensure_symbol_column(df: pd.DataFrame) -> str:
    for c in ["symbol", "ticker", "underlying"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a symbol column. Found: {list(df.columns)}")


def _ensure_date_column(df: pd.DataFrame) -> str:
    for c in ["date", "timestamp", "asof_date", "datetime"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a date column. Found: {list(df.columns)}")


def _max_drawdown(equity: pd.Series) -> Tuple[float, Optional[str], Optional[str]]:
    """
    equity: pandas Series of portfolio value indexed by date (ascending)
    returns: (max_dd, peak_date, trough_date)
    max_dd is negative (e.g., -0.23 for -23%)
    """
    if equity.empty:
        return 0.0, None, None

    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    trough_idx = dd.idxmin()
    max_dd = float(dd.loc[trough_idx])

    peak_idx = equity.loc[:trough_idx].idxmax()

    return max_dd, str(peak_idx), str(trough_idx)


def _annualized_vol(daily_returns: pd.Series) -> float:
    if daily_returns.dropna().empty:
        return 0.0
    return float(daily_returns.dropna().std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _annualized_sharpe(daily_returns: pd.Series, rf_annual: float = 0.0) -> float:
    """
    rf_annual: e.g. 0.03 for 3% annual
    Uses simple approximation: excess daily = r - rf_daily, sharpe = mean/std * sqrt(252)
    """
    r = daily_returns.dropna()
    if r.empty:
        return 0.0
    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    ex = r - rf_daily
    sd = ex.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(ex.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR))


def _cagr(equity: pd.Series) -> float:
    if equity.dropna().empty:
        return 0.0
    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    if start_val <= 0:
        return 0.0
    days = (pd.to_datetime(equity.index[-1]) - pd.to_datetime(equity.index[0])).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    return float((end_val / start_val) ** (1.0 / years) - 1.0)


def _total_return(equity: pd.Series) -> float:
    if equity.dropna().empty:
        return 0.0
    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    if start_val == 0:
        return 0.0
    return float(end_val / start_val - 1.0)


def _win_rate(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return 0.0
    return float((r > 0).mean())


@dataclass
class FundMetrics:
    start_date: str
    end_date: str
    n_days: int
    total_return: float
    cagr: float
    vol_annual: float
    sharpe_annual: float
    max_drawdown: float
    max_dd_peak_date: Optional[str]
    max_dd_trough_date: Optional[str]
    win_rate_daily: float

#note;;; only daily, no intraday so no 3:50-4:00

def load_positions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["symbol", "entry_date", "entry_price", "side", "asset_type"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{path} missing required column '{c}'. Found: {list(df.columns)}")

    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce").dt.date
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["side"] = df["side"].astype(str).str.strip().str.lower()
#check capital if there
    if "capital" in df.columns:
        df["capital"] = pd.to_numeric(df["capital"], errors="coerce")
    else:
        df["capital"] = np.nan

    df = df[(df["asset_type"].astype(str).str.lower() == "equity") & (df["side"] == "long")].copy()

    if df.empty:
        raise ValueError("No long equity positions found after filtering asset_type==equity and side==long.")

    return df


def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    sym_col = _ensure_symbol_column(df)
    date_col = _ensure_date_column(df)
    px_col = _pick_price_column(df)

    df[sym_col] = df[sym_col].astype(str).str.strip().str.upper()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    df[px_col] = pd.to_numeric(df[px_col], errors="coerce")

    df = df.dropna(subset=[sym_col, date_col, px_col]).copy()

    df = df.rename(columns={sym_col: "symbol", date_col: "date", px_col: "price"})
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    return df


def build_fund_timeseries(positions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a daily timeseries:
      - per-symbol returns (from daily prices)
      - portfolio daily return (capital-weighted if capital present, else equal-weight)
      - portfolio equity curve (starts at 1.0)
    Logic:
      - each position is considered "active" from entry_date onward
      - weights are fixed from entry based on initial capital allocation
        (if capital missing -> equal weight across names)
      - handles different trading calendars by aligning on dates and skipping missing returns for that symbol/day
    """

    pos_syms = positions["symbol"].unique().tolist()
    prices = prices[prices["symbol"].isin(pos_syms)].copy()

    missing_syms = sorted(set(pos_syms) - set(prices["symbol"].unique()))
    if missing_syms:
        print("[WARN] No price data for symbols:", missing_syms)

    prices["ret"] = prices.groupby("symbol")["price"].pct_change()

    all_dates = pd.DataFrame({"date": sorted(prices["date"].unique())})

    ret_wide = prices.pivot(index="date", columns="symbol", values="ret").sort_index()

    entry_dates = positions.set_index("symbol")["entry_date"].to_dict()

    active = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns, dtype=float)
    for sym in ret_wide.columns:
        ed = entry_dates.get(sym, None)
        if ed is None or pd.isna(ed):
            active[sym] = 1.0
        else:
            active[sym] = (active.index >= ed).astype(float)

    # determine weights:
    # using equal weights across all assets until leads tells the capitals
    cap = positions.set_index("symbol")["capital"]

    if cap.notna().any():
        cap_filled = cap.fillna(0.0)
        total_cap = float(cap_filled.sum())
        if total_cap <= 0:
            w = pd.Series(1.0, index=ret_wide.columns) / len(ret_wide.columns)
            weight_mode = "equal_weight_fallback"
        else:
            w = cap_filled.reindex(ret_wide.columns).fillna(0.0) / total_cap
            weight_mode = "capital_weighted"
    else:
        w = pd.Series(1.0, index=ret_wide.columns) / len(ret_wide.columns)
        weight_mode = "equal_weight"

    weighted = ret_wide.copy()

    avail = (active == 1.0) & (~ret_wide.isna())
    base_w = w.reindex(ret_wide.columns).fillna(0.0)

    # computation of daily returns
    port_ret = []
    for dt in ret_wide.index:
        mask = avail.loc[dt]
        if mask.sum() == 0:
            port_ret.append(np.nan)
            continue
        w_day = base_w[mask]
        w_sum = float(w_day.sum())
        if w_sum <= 0:
            # equal weight across available
            w_day = pd.Series(1.0, index=w_day.index) / len(w_day.index)
        else:
            w_day = w_day / w_sum
        r_day = float((ret_wide.loc[dt, mask.index] * w_day).sum())
        port_ret.append(r_day)

    ts = pd.DataFrame(index=ret_wide.index)
    ts.index.name = "date"
    ts["portfolio_return"] = port_ret

    ts["equity"] = (1.0 + ts["portfolio_return"].fillna(0.0)).cumprod()

    ts["weight_mode"] = weight_mode

    return ts.reset_index()


def compute_metrics(ts: pd.DataFrame, rf_annual: float = 0.0) -> FundMetrics:
    ts = ts.dropna(subset=["date"]).copy()
    ts["date"] = pd.to_datetime(ts["date"])
    ts = ts.sort_values("date")

    # daily returns series
    r = ts["portfolio_return"].astype(float)
    equity = ts.set_index("date")["equity"].astype(float)

    max_dd, peak, trough = _max_drawdown(equity)
    metrics = FundMetrics(
        start_date=str(ts["date"].iloc[0].date()),
        end_date=str(ts["date"].iloc[-1].date()),
        n_days=int(len(ts)),
        total_return=_total_return(equity),
        cagr=_cagr(equity),
        vol_annual=_annualized_vol(r),
        sharpe_annual=_annualized_sharpe(r, rf_annual=rf_annual),
        max_drawdown=max_dd,
        max_dd_peak_date=peak,
        max_dd_trough_date=trough,
        win_rate_daily=_win_rate(r),
    )
    return metrics


def main(
    positions_csv: Optional[str] = None,
    prices_csv: str = DEFAULT_PRICES_CSV,
    rf_annual: float = 0.0,
):
    pos_path = positions_csv or _find_existing(DEFAULT_POSITIONS_CSV_CANDIDATES)
    if not pos_path:
        raise FileNotFoundError(
            f"Could not find positions CSV. Looked for: {DEFAULT_POSITIONS_CSV_CANDIDATES}. "
            f"Or pass positions_csv=..."
        )
    if not os.path.exists(prices_csv):
        raise FileNotFoundError(f"Missing prices CSV: {prices_csv}")

    positions = load_positions(pos_path)
    prices = load_prices(prices_csv)

    ts = build_fund_timeseries(positions, prices)
    ts.to_csv(OUT_TIMESERIES_CSV, index=False, encoding="utf-8", lineterminator="\n")

    metrics = compute_metrics(ts, rf_annual=rf_annual)

    with open(OUT_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics.__dict__, f, indent=2)

    print("\n=== Fund Metrics ===")
    print(f"Positions file: {os.path.abspath(pos_path)}")
    print(f"Prices file:    {os.path.abspath(prices_csv)}")
    print(f"Date range:     {metrics.start_date} -> {metrics.end_date}  ({metrics.n_days} rows)")
    print(f"Weight mode:    {ts['weight_mode'].iloc[0] if not ts.empty else 'n/a'}")
    print(f"Total return:   {metrics.total_return:.4f}")
    print(f"CAGR:           {metrics.cagr:.4f}")
    print(f"Vol (ann):      {metrics.vol_annual:.4f}")
    print(f"Sharpe (ann):   {metrics.sharpe_annual:.4f}  (rf={rf_annual:.4f})")
    print(f"Max drawdown:   {metrics.max_drawdown:.4f}  (peak={metrics.max_dd_peak_date}, trough={metrics.max_dd_trough_date})")
    print(f"Win rate (d):   {metrics.win_rate_daily:.4f}")

    print("\nWrote:")
    print(f"  - {os.path.abspath(OUT_TIMESERIES_CSV)}")
    print(f"  - {os.path.abspath(OUT_METRICS_JSON)}")


if __name__ == "__main__":
    # edit risk free annual here
    main(rf_annual=0.0)