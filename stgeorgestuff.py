import yfinance as yf
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional, Tuple

# csv logging
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


class PortfolioManager:
    def __init__(self, starting_cash=100000, data_file="portfolio_data.json",
                 csv_path: str = "trading_log.csv"):
        self.stocks = ['NVDA', 'MSFT', 'AAPL', 'JPM', 'UNH']
        self.trade_probability = 0.1
        self.shares_per_trade = 100
        self.data_file = data_file
        self.risk_free_rate = 0.05

        self.logger = CSVTradeLogger(csv_path, self.stocks)
        self.logger.autobackfill_on_start(default_lookback_days=365)

        if os.path.exists(data_file):
            self.load_data()
        else:
            self.portfolio = {stock: 0 for stock in self.stocks}
            self.cash = starting_cash
            self.trades = []
            self.initial_value = starting_cash
            self.daily_values = []
            self.start_date = datetime.now().date()
            self.save_data()

    def save_data(self):
        data = {
            'portfolio': self.portfolio,
            'cash': self.cash,
            'trades': self.trades,
            'initial_value': self.initial_value,
            'daily_values': self.daily_values,
            'start_date': self.start_date.isoformat() if hasattr(self,'start_date') else datetime.now().date().isoformat()
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load_data(self):
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        self.portfolio = data['portfolio']
        self.cash = data['cash']
        self.trades = data['trades']
        self.initial_value = data['initial_value']
        self.daily_values = data.get('daily_values', [])
        self.start_date = datetime.fromisoformat(data.get('start_date', datetime.now().date().isoformat())).date()

    def get_current_prices(self) -> Dict[str, float]:
        try:
            prices = {}
            for stock in self.stocks:
                ticker = yf.Ticker(stock)
                hist = ticker.history(period='1d')
                if not hist.empty:
                    prices[stock] = hist['Close'].iloc[-1]
                else:
                    info = ticker.info
                    prices[stock] = info.get('regularMarketPrice', 100)
            return prices
        except Exception as e:
            print(f"Error fetching prices: {e}")
            return {stock: random.uniform(100, 500) for stock in self.stocks}

    def get_sp500_data(self, start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.now()
        try:
            spy = yf.Ticker('SPY')
            return spy.history(start=start_date, end=end_date)
        except Exception as e:
            print(f"Error fetching SP500 data: {e}")
            return pd.DataFrame()

    def snapshot_now(self, note="scheduled snapshot"):
        prices = self.get_current_prices()
        self.logger.log_now(
            prices=prices,
            action_map={},
            positions_after={t: self.portfolio[t] for t in self.stocks},
            cash_after=self.cash,
            note=note
        )

    def simulate_trade(self) -> Optional[Dict]:
        if random.random() < self.trade_probability:
            stock_to_buy = random.choice(self.stocks)
            prices = self.get_current_prices()
            price = prices[stock_to_buy]
            cost = price * self.shares_per_trade

            if self.cash >= cost:
                self.portfolio[stock_to_buy] += self.shares_per_trade
                self.cash -= cost
                trade = {
                    'timestamp': datetime.now().isoformat(),
                    'stock': stock_to_buy,
                    'shares': self.shares_per_trade,
                    'price': price,
                    'total_cost': cost
                }
                self.trades.append(trade)
                self.save_data()

                action_map = {t: {"action":"NONE","quantity":0} for t in self.stocks}
                action_map[stock_to_buy] = {"action":"BUY","quantity":self.shares_per_trade}
                self.logger.log_now(
                    prices=prices,
                    action_map=action_map,
                    positions_after={t: self.portfolio[t] for t in self.stocks},
                    cash_after=self.cash,
                    note="auto-sim trade"
                )
                print(f"Trade executed: {self.shares_per_trade} {stock_to_buy} @ ${price:.2f}")
                return trade
            else:
                print(f"Insufficient cash for {stock_to_buy}")
                prices = self.get_current_prices()
                self.logger.log_now(
                    prices=prices,
                    action_map={},
                    positions_after={t: self.portfolio[t] for t in self.stocks},
                    cash_after=self.cash,
                    note="insufficient-cash snapshot"
                )
        else:
            print("No trade this turn")
            prices = self.get_current_prices()
            self.logger.log_now(
                prices=prices,
                action_map={},
                positions_after={t: self.portfolio[t] for t in self.stocks},
                cash_after=self.cash,
                note="no-trade snapshot"
            )
        return None

    def record_daily_value(self):
        stats = self.calculate_portfolio_stats()
        today = datetime.now().date().isoformat()
        updated = False
        for record in self.daily_values:
            if record['date'] == today:
                record['value'] = stats['total_portfolio_value']
                updated = True
                break
        if not updated:
            self.daily_values.append({'date': today, 'value': stats['total_portfolio_value']})
        self.save_data()

    def calculate_portfolio_stats(self) -> Dict:
        prices = self.get_current_prices()
        stock_values = {}
        total_stock_value = 0
        for stock in self.stocks:
            shares = self.portfolio[stock]
            current_price = prices[stock]
            value = shares * current_price
            stock_values[stock] = {'shares': shares, 'price': current_price, 'value': value}
            total_stock_value += value
        total_portfolio_value = total_stock_value + self.cash
        total_pnl = total_portfolio_value - self.initial_value
        pnl_percent = (total_pnl / self.initial_value) * 100 if self.initial_value > 0 else 0
        return {
            'stock_values': stock_values,
            'prices': prices,
            'total_stock_value': total_stock_value,
            'cash': self.cash,
            'total_portfolio_value': total_portfolio_value,
            'total_pnl': total_pnl,
            'pnl_percent': pnl_percent
        }

    def calculate_returns(self) -> List[float]:
        if len(self.daily_values) < 2:
            return []
        returns = []
        for i in range(1, len(self.daily_values)):
            prev_val = self.daily_values[i - 1]['value']
            curr_val = self.daily_values[i]['value']
            if prev_val > 0:
                returns.append((curr_val - prev_val) / prev_val)
        return returns

    def calculate_sharpe_ratio(self, returns: List[float]) -> float:
        if len(returns) < 2:
            return 0.0
        returns_array = np.array(returns)
        excess_returns = returns_array - (self.risk_free_rate / 252)
        if np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def get_sp500_returns(self) -> Tuple[List[float], float]:
        start_date = datetime.combine(self.start_date, datetime.min.time())
        sp500_data = self.get_sp500_data(start_date)
        if sp500_data.empty:
            return [], 0.0
        sp500_data['daily_return'] = sp500_data['Close'].pct_change()
        sp500_returns = sp500_data['daily_return'].dropna().tolist()
        sp500_sharpe = self.calculate_sharpe_ratio(sp500_returns)
        return sp500_returns, sp500_sharpe

    def compare_to_sp500(self) -> Dict:
        self.record_daily_value()
        portfolio_returns = self.calculate_returns()
        portfolio_sharpe = self.calculate_sharpe_ratio(portfolio_returns)
        sp500_returns, sp500_sharpe = self.get_sp500_returns()
        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252) if portfolio_returns else 0
        sp500_volatility = np.std(sp500_returns) * np.sqrt(252) if sp500_returns else 0
        stats = self.calculate_portfolio_stats()
        portfolio_total_return = stats['pnl_percent'] / 100
        start_date = datetime.combine(self.start_date, datetime.min.time())
        sp500_data = self.get_sp500_data(start_date)
        sp500_total_return = 0
        if not sp500_data.empty:
            initial_sp500 = sp500_data['Close'].iloc[0]
            current_sp500 = sp500_data['Close'].iloc[-1]
            sp500_total_return = (current_sp500 - initial_sp500) / initial_sp500
        return {
            'portfolio': {
                'total_return': portfolio_total_return,
                'sharpe_ratio': portfolio_sharpe,
                'volatility': portfolio_volatility,
                'num_trades': len(self.trades)
            },
            'sp500': {
                'total_return': sp500_total_return,
                'sharpe_ratio': sp500_sharpe,
                'volatility': sp500_volatility
            }
        }

    def display_portfolio_summary(self):
        stats = self.calculate_portfolio_stats()
        print("\nPORTFOLIO SUMMARY")
        print("-" * 50)
        print(f"Total Value: ${stats['total_portfolio_value']:,.2f}")
        print(f"P&L: ${stats['total_pnl']:,.2f} ({stats['pnl_percent']:+.2f}%)")
        print(f"Cash: ${stats['cash']:,.2f}")
        print(f"Stock Value: ${stats['total_stock_value']:,.2f}")
        print(f"\nHOLDINGS:")
        for stock, data in stats['stock_values'].items():
            if data['shares'] > 0:
                weight = (data['value'] / stats['total_portfolio_value']) * 100
                print(f"{stock}: {data['shares']} shares, ${data['value']:,.2f} ({weight:.1f}%)")
        print(f"\nTotal Trades: {len(self.trades)}")

    def display_comparison(self):
        comparison = self.compare_to_sp500()
        print("\nPERFORMANCE COMPARISON")
        print("-" * 50)
        portfolio = comparison['portfolio']
        sp500 = comparison['sp500']
        print(f"{'Metric':<20} {'Portfolio':<15} {'SP500':<15}")
        print("-" * 50)
        print(f"{'Total Return':<20} {portfolio['total_return'] * 100:>+8.2f}%     {sp500['total_return'] * 100:>+8.2f}%")
        print(f"{'Sharpe Ratio':<20} {portfolio['sharpe_ratio']:>8.3f}       {sp500['sharpe_ratio']:>8.3f}")
        print(f"{'Volatility':<20} {portfolio['volatility'] * 100:>8.2f}%       {sp500['volatility'] * 100:>8.2f}%")
        print(f"{'Trades':<20} {len(self.trades):>8d}       {'N/A':>8}")

    def run_simulation_turns(self, num_turns=10):
        trades_executed = 0
        for turn in range(1, num_turns + 1):
            print(f"Turn {turn}")
            trade = self.simulate_trade()
            if trade:
                trades_executed += 1
        print(f"Simulation complete: {trades_executed}/{num_turns} trades executed")
        return trades_executed

    def reset_portfolio(self):
        self.portfolio = {stock: 0 for stock in self.stocks}
        self.cash = self.initial_value
        self.trades = []
        self.daily_values = []
        self.start_date = datetime.now().date()
        self.save_data()
        print("Portfolio reset")


def main():
    portfolio = PortfolioManager(csv_path="trading_log.csv")
    while True:
        print("\nPORTFOLIO MANAGER")
        print("1. View Portfolio")
        print("2. Simulate Trade")
        print("3. Run Multiple Turns")
        print("4. Compare to SP500")
        print("5. Reset Portfolio")
        print("6. Exit")
        print("7. Backfill History (1y daily)")
        print("8. Snapshot Now (no-trade)")
        choice = input("\nSelect (1-8): ").strip()
        if choice == '1':
            portfolio.display_portfolio_summary()
        elif choice == '2':
            portfolio.simulate_trade()
        elif choice == '3':
            turns = int(input("Number of turns (default 10): ") or "10")
            portfolio.run_simulation_turns(turns)
            portfolio.display_portfolio_summary()
        elif choice == '4':
            portfolio.display_comparison()
        elif choice == '5':
            confirm = input("Reset portfolio? (y/N): ").lower()
            if confirm == 'y':
                portfolio.reset_portfolio()
        elif choice == '6':
            break
        elif choice == '7':
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            portfolio.logger.backfill_history(start=start, end=None, interval="1d",
                                              note="history backfill (menu)")
            print("Backfill complete.")
        elif choice == '8':
            portfolio.snapshot_now(note="manual snapshot")
            print("Snapshot logged.")
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()