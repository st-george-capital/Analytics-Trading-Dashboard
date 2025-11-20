import dash
from dash import dcc, html, Input, Output, callback, dash_table, ctx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

from stgeo_v1 import PortfolioManager


class PortfolioDashboard:
    def __init__(self, portfolio_manager):
        self.pm = portfolio_manager
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def load_trading_log(self):
        """Load trading log CSV data"""
        try:
            df = pd.read_csv(self.pm.logger.csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except FileNotFoundError:
            return pd.DataFrame()

    def calculate_performance_metrics(self, df):
        """Calculate advanced performance metrics"""
        if df.empty:
            return {}

        # Get portfolio values over time
        portfolio_values = []
        for timestamp in df['timestamp'].dt.floor('h').unique():
            timestamp_data = df[df['timestamp'].dt.floor('h') == timestamp]
            if not timestamp_data['cash_after'].isna().all():
                latest_cash = timestamp_data['cash_after'].dropna().iloc[-1]
                stock_value = 0
                for stock in self.pm.stocks:
                    stock_rows = timestamp_data[timestamp_data['ticker'] == stock]
                    if not stock_rows.empty and not stock_rows['position_after'].isna().all():
                        position = stock_rows['position_after'].dropna().iloc[-1]
                        price = stock_rows['close'].dropna().iloc[-1]
                        if not pd.isna(position) and not pd.isna(price):
                            stock_value += position * price
                portfolio_values.append(stock_value + latest_cash)

        if len(portfolio_values) < 2:
            return {}

        # Calculate metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        metrics = {
            'total_return': ((portfolio_values[-1] - self.pm.initial_value) / self.pm.initial_value) * 100,
            'volatility': np.std(returns) * np.sqrt(252) * 100 if len(returns) > 0 else 0,
            'max_drawdown': self.calculate_max_drawdown(portfolio_values),
            'win_rate': self.calculate_win_rate(df),
            'avg_trade_return': self.calculate_avg_trade_return(df),
            'best_stock': self.get_best_performing_stock(df),
            'worst_stock': self.get_worst_performing_stock(df)
        }

        return metrics

    def calculate_max_drawdown(self, values):
        """Calculate maximum drawdown"""
        if len(values) < 2:
            return 0
        peak = values[0]
        max_dd = 0
        for value in values:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def calculate_win_rate(self, df):
        """Calculate win rate from trades"""
        trades = df[df['action'] == 'BUY'].copy()
        if trades.empty:
            return 0

        winning_trades = 0
        for _, trade in trades.iterrows():
            # Check if stock price increased after purchase 
            future_prices = df[(df['ticker'] == trade['ticker']) &
                               (df['timestamp'] > trade['timestamp'])]['close']
            if not future_prices.empty and future_prices.iloc[-1] > trade['close']:
                winning_trades += 1

        return (winning_trades / len(trades)) * 100 if len(trades) > 0 else 0

    def calculate_avg_trade_return(self, df):
        """Calculate average return per trade"""
        trades = df[df['action'] == 'BUY'].copy()
        if trades.empty:
            return 0

        returns = []
        for _, trade in trades.iterrows():
            future_prices = df[(df['ticker'] == trade['ticker']) &
                               (df['timestamp'] > trade['timestamp'])]['close']
            if not future_prices.empty:
                ret = ((future_prices.iloc[-1] - trade['close']) / trade['close']) * 100
                returns.append(ret)

        return np.mean(returns) if returns else 0

    def get_best_performing_stock(self, df):
        """Get best performing stock"""
        if df.empty:
            return "N/A"

        best_stock = None
        best_return = -float('inf')

        for stock in self.pm.stocks:
            stock_data = df[df['ticker'] == stock]
            if not stock_data.empty and len(stock_data) > 1:
                first_price = stock_data['close'].iloc[0]
                last_price = stock_data['close'].iloc[-1]
                ret = ((last_price - first_price) / first_price) * 100
                if ret > best_return:
                    best_return = ret
                    best_stock = f"{stock} (+{ret:.1f}%)"

        return best_stock if best_stock else "N/A"

    def get_worst_performing_stock(self, df):
        """Get worst performing stock"""
        if df.empty:
            return "N/A"

        worst_stock = None
        worst_return = float('inf')

        for stock in self.pm.stocks:
            stock_data = df[df['ticker'] == stock]
            if not stock_data.empty and len(stock_data) > 1:
                first_price = stock_data['close'].iloc[0]
                last_price = stock_data['close'].iloc[-1]
                ret = ((last_price - first_price) / first_price) * 100
                if ret < worst_return:
                    worst_return = ret
                    worst_stock = f"{stock} ({ret:+.1f}%)"

        return worst_stock if worst_stock else "N/A"

    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("📊 Portfolio Dashboard", style={'textAlign': 'center', 'marginBottom': 30, 'color': '#2c3e50'}),

            # Control buttons with improved styling
            html.Div([
                html.Button('Simulate Trade', id='simulate-btn', n_clicks=0,
                            style={'marginRight': 10, 'padding': '12px 24px', 'backgroundColor': '#3498db',
                                   'color': 'white', 'border': 'none', 'borderRadius': '8px',
                                   'fontSize': '16px', 'cursor': 'pointer', 'fontWeight': 'bold',
                                   'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}),
                html.Button('Take Snapshot', id='snapshot-btn', n_clicks=0,
                            style={'marginRight': 10, 'padding': '12px 24px', 'backgroundColor': '#27ae60',
                                   'color': 'white', 'border': 'none', 'borderRadius': '8px',
                                   'fontSize': '16px', 'cursor': 'pointer', 'fontWeight': 'bold',
                                   'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}),
                html.Button('Backfill CSV', id='backfill-btn', n_clicks=0,
                            style={'marginRight': 10, 'padding': '12px 24px', 'backgroundColor': '#8e44ad',
                                   'color': 'white', 'border': 'none', 'borderRadius': '8px',
                                   'fontSize': '16px', 'cursor': 'pointer', 'fontWeight': 'bold',
                                   'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}),
                html.Button('Reset Portfolio', id='reset-btn', n_clicks=0,
                            style={'padding': '12px 24px', 'backgroundColor': '#e74c3c', 'color': 'white',
                                   'border': 'none', 'borderRadius': '8px', 'fontSize': '16px',
                                   'cursor': 'pointer', 'fontWeight': 'bold',
                                   'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}),
            ], style={'textAlign': 'center', 'marginBottom': 30}),

            # Auto-refresh interval
            dcc.Interval(id='interval-component', interval=30 * 1000, n_intervals=0),

            # Status message
            html.Div(id='status-message',
                     style={'textAlign': 'center', 'marginBottom': 20, 'fontSize': 18, 'fontWeight': 'bold',
                            'padding': '10px', 'borderRadius': '8px', 'backgroundColor': '#ecf0f1'}),

            # Key Performance Metrics Row
            html.Div([
                html.Div(id='kpi-cards', style={'display': 'flex', 'justifyContent': 'space-around',
                                                'flexWrap': 'wrap', 'marginBottom': 30})
            ]),

            # Main content in two columns
            html.Div([
                # Left column - Portfolio overview
                html.Div([
                    html.H3("💼 Portfolio Overview", style={'textAlign': 'center', 'color': '#34495e'}),
                    html.Div(id='portfolio-stats'),
                    html.Br(),
                    dcc.Graph(id='portfolio-pie-chart')
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

                # Right column - Holdings and trades
                html.Div([
                    html.H3("📈 Current Holdings", style={'textAlign': 'center', 'color': '#34495e'}),
                    html.Div(id='holdings-table'),
                    html.Br(),
                    html.H3("🔔 Recent Activity", style={'textAlign': 'center', 'color': '#34495e'}),
                    html.Div(id='recent-trades')
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'padding': '10px'})
            ]),

            html.Hr(style={'margin': '30px 0'}),

            # Charts row - 3 charts
            html.Div([
                html.Div([
                    html.H3("Portfolio Value Over Time", style={'textAlign': 'center', 'color': '#34495e'}),
                    dcc.Graph(id='portfolio-timeline')
                ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

                html.Div([
                    html.H3("Stock Prices", style={'textAlign': 'center', 'color': '#34495e'}),
                    dcc.Graph(id='stock-prices')
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'padding': '10px'})
            ]),

            # New row for additional charts
            html.Div([
                html.Div([
                    html.H3("Returns Distribution", style={'textAlign': 'center', 'color': '#34495e'}),
                    dcc.Graph(id='returns-histogram')
                ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

                html.Div([
                    html.H3("Trade Performance", style={'textAlign': 'center', 'color': '#34495e'}),
                    dcc.Graph(id='trade-performance')
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'padding': '10px'})
            ])
        ], style={'padding': 20, 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f8f9fa'})

    def setup_callbacks(self):
        @callback(
            [Output('portfolio-stats', 'children'),
             Output('holdings-table', 'children'),
             Output('recent-trades', 'children'),
             Output('portfolio-pie-chart', 'figure'),
             Output('portfolio-timeline', 'figure'),
             Output('stock-prices', 'figure'),
             Output('status-message', 'children'),
             Output('kpi-cards', 'children'),
             Output('returns-histogram', 'figure'),
             Output('trade-performance', 'figure')],
            [Input('interval-component', 'n_intervals'),
             Input('simulate-btn', 'n_clicks'),
             Input('snapshot-btn', 'n_clicks'),
             Input('backfill-btn', 'n_clicks'),
             Input('reset-btn', 'n_clicks')]
        )
        def update_dashboard(n_intervals, sim_clicks, snap_clicks, backfill_clicks, reset_clicks):
            status_msg = ""

            # Handle button clicks
            if ctx.triggered:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                if button_id == 'simulate-btn' and sim_clicks > 0:
                    trade = self.pm.simulate_trade()
                    if trade:
                        status_msg = f"Trade executed: {trade['shares']} {trade['stock']} @ ${trade['price']:.2f}"
                    else:
                        status_msg = "ℹNo trade executed this turn"
                elif button_id == 'snapshot-btn' and snap_clicks > 0:
                    self.pm.snapshot_now("dashboard snapshot")
                    status_msg = "📸 Portfolio snapshot saved"
                elif button_id == 'backfill-btn' and backfill_clicks > 0:
                    ok = self.pm.logger.manual_backfill()
                    if ok:
                        status_msg = "CSV backfilled with latest history"
                    else:
                        status_msg = " No new history to backfill (or cooldown active)"
                elif button_id == 'reset-btn' and reset_clicks > 0:
                    self.pm.reset_portfolio()
                    status_msg = "🔄 Portfolio reset to initial state"

            # Get current portfolio stats
            stats = self.pm.calculate_portfolio_stats()
            df = self.load_trading_log()
            metrics = self.calculate_performance_metrics(df)

            # KPI Cards
            kpi_cards = self.create_kpi_cards(stats, metrics)

            # Portfolio stats display (simplified since we have KPI cards)
            stats_style = {'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '12px',
                           'margin': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}
            pnl_color = '#27ae60' if stats['total_pnl'] >= 0 else '#e74c3c'

            stats_display = html.Div([
                html.Div(f"${stats['total_portfolio_value']:,.2f}",
                         style={'fontSize': 36, 'fontWeight': 'bold', 'textAlign': 'center', 'color': '#2980b9'}),
                html.P("Total Portfolio Value",
                       style={'textAlign': 'center', 'margin': '0 0 20px 0', 'color': '#7f8c8d', 'fontSize': '16px'}),

                html.Div([
                    html.Div([
                        html.Div(f"${stats['cash']:,.2f}", style={'fontSize': 18, 'fontWeight': 'bold'}),
                        html.P(" Cash", style={'margin': 0, 'fontSize': 14, 'color': '#7f8c8d'})
                    ], style={'textAlign': 'center', 'width': '50%', 'display': 'inline-block'}),

                    html.Div([
                        html.Div(f"${stats['total_stock_value']:,.2f}", style={'fontSize': 18, 'fontWeight': 'bold'}),
                        html.P(" Stock Value", style={'margin': 0, 'fontSize': 14, 'color': '#7f8c8d'})
                    ], style={'textAlign': 'center', 'width': '50%', 'display': 'inline-block'})
                ])
            ], style=stats_style)

            # Holdings table with improved styling
            holdings_data = []
            for stock, data in stats['stock_values'].items():
                if data['shares'] > 0:
                    weight = (data['value'] / stats['total_portfolio_value']) * 100
                    holdings_data.append(
                        [stock, data['shares'], f"${data['price']:.2f}", f"${data['value']:,.2f}", f"{weight:.1f}%"])

            if holdings_data:
                table_style = {'width': '100%', 'textAlign': 'center', 'fontSize': '14px',
                               'backgroundColor': 'white', 'borderRadius': '12px', 'overflow': 'hidden',
                               'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}
                header_style = {'backgroundColor': '#3498db', 'color': 'white', 'padding': '12px', 'fontWeight': 'bold'}
                cell_style = {'padding': '10px', 'borderBottom': '1px solid #ecf0f1'}

                holdings_table = html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Stock", style=header_style),
                            html.Th("Shares", style=header_style),
                            html.Th("Price", style=header_style),
                            html.Th("Value", style=header_style),
                            html.Th("Weight", style=header_style)
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td(row[0], style={**cell_style, 'fontWeight': 'bold'}),
                            html.Td(row[1], style=cell_style),
                            html.Td(row[2], style=cell_style),
                            html.Td(row[3], style=cell_style),
                            html.Td(row[4], style=cell_style)
                        ]) for row in holdings_data
                    ])
                ], style=table_style)
            else:
                holdings_table = html.Div("No current holdings",
                                          style={'textAlign': 'center', 'padding': '30px', 'color': '#95a5a6',
                                                 'backgroundColor': 'white', 'borderRadius': '12px'})

            # Recent trades with enhanced styling
            recent_trades_display = html.Div("No recent trades",
                                             style={'textAlign': 'center', 'padding': '30px', 'color': '#95a5a6',
                                                    'backgroundColor': 'white', 'borderRadius': '12px'})
            if self.pm.trades:
                last_5_trades = self.pm.trades[-5:]
                trades_list = []
                for i, trade in enumerate(reversed(last_5_trades)):
                    trade_time = datetime.fromisoformat(trade['timestamp']).strftime('%m/%d %H:%M')
                    bg_color = 'white' if i % 2 == 0 else '#f8f9fa'
                    trades_list.append(
                        html.Div([
                            html.Div([
                                html.Span(trade_time, style={'fontWeight': 'bold', 'color': '#3498db'}),
                                html.Span(f" • {trade['stock']}",
                                          style={'color': '#2c3e50', 'fontWeight': 'bold', 'marginLeft': '10px'})
                            ]),
                            html.Div([
                                html.Span(f"{trade['shares']} shares @ ${trade['price']:.2f}",
                                          style={'color': '#7f8c8d'}),
                                html.Span(f" = ${trade['total_cost']:,.2f}",
                                          style={'color': '#27ae60', 'fontWeight': 'bold', 'marginLeft': '10px'})
                            ])
                        ], style={'padding': '12px', 'backgroundColor': bg_color, 'margin': '4px 0',
                                  'borderRadius': '8px', 'borderLeft': '4px solid #3498db'})
                    )
                recent_trades_display = html.Div(trades_list, style={'backgroundColor': 'white',
                                                                     'borderRadius': '12px', 'padding': '10px'})

            # Portfolio pie chart
            pie_data = []
            pie_labels = []
            pie_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']

            for i, (stock, data) in enumerate(stats['stock_values'].items()):
                if data['value'] > 0:
                    pie_data.append(data['value'])
                    pie_labels.append(stock)

            if stats['cash'] > 0:
                pie_data.append(stats['cash'])
                pie_labels.append('Cash')

            pie_fig = go.Figure(data=[go.Pie(
                labels=pie_labels,
                values=pie_data,
                marker_colors=pie_colors[:len(pie_data)],
                textinfo='label+percent',
                textfont_size=13,
                hole=0.4,
                hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
            )])
            pie_fig.update_layout(
                title="Portfolio Allocation",
                height=450,
                font=dict(size=12),
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            # Portfolio timeline
            timeline_fig = go.Figure()
            if not df.empty and 'cash_after' in df.columns:
                portfolio_data = []
                for timestamp in df['timestamp'].dt.floor('h').unique():
                    timestamp_data = df[df['timestamp'].dt.floor('h') == timestamp]
                    if not timestamp_data['cash_after'].isna().all():
                        latest_cash = timestamp_data['cash_after'].dropna().iloc[-1]
                        stock_value = 0
                        for stock in self.pm.stocks:
                            stock_rows = timestamp_data[timestamp_data['ticker'] == stock]
                            if not stock_rows.empty and not stock_rows['position_after'].isna().all():
                                position = stock_rows['position_after'].dropna().iloc[-1]
                                price = stock_rows['close'].dropna().iloc[-1]
                                if not pd.isna(position) and not pd.isna(price):
                                    stock_value += position * price
                        total_value = stock_value + latest_cash
                        portfolio_data.append({'timestamp': timestamp, 'value': total_value})

                if portfolio_data:
                    portfolio_df = pd.DataFrame(portfolio_data)
                    colors = ['#27ae60' if v >= self.pm.initial_value else '#e74c3c'
                              for v in portfolio_df['value']]

                    timeline_fig.add_trace(go.Scatter(
                        x=portfolio_df['timestamp'],
                        y=portfolio_df['value'],
                        mode='lines+markers',
                        name='Portfolio Value',
                        line=dict(color='#3498db', width=3),
                        marker=dict(size=8, color=colors, line=dict(color='white', width=2)),
                        fill='tozeroy',
                        fillcolor='rgba(52, 152, 219, 0.1)',
                        hovertemplate='<b>%{x}</b><br>Value: $%{y:,.2f}<extra></extra>'
                    ))

                    timeline_fig.add_hline(
                        y=self.pm.initial_value,
                        line_dash="dash",
                        line_color="#e74c3c",
                        line_width=2,
                        annotation_text="Initial Value",
                        annotation_position="right"
                    )

            timeline_fig.update_layout(
                height=450,
                xaxis_title="Time",
                yaxis_title="Portfolio Value ($)",
                hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='#ecf0f1'),
                yaxis=dict(gridcolor='#ecf0f1')
            )

            # Stock prices chart
            prices_fig = go.Figure()
            if not df.empty:
                colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
                for i, stock in enumerate(self.pm.stocks):
                    stock_data = df[df['ticker'] == stock].copy()
                    if not stock_data.empty:
                        prices_fig.add_trace(go.Scatter(
                            x=stock_data['timestamp'],
                            y=stock_data['close'],
                            mode='lines',
                            name=stock,
                            line=dict(color=colors[i % len(colors)], width=2.5),
                            hovertemplate='<b>%{fullData.name}</b><br>Price: $%{y:.2f}<extra></extra>'
                        ))

            prices_fig.update_layout(
                height=450,
                xaxis_title="Time",
                yaxis_title="Stock Price ($)",
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='#ecf0f1'),
                yaxis=dict(gridcolor='#ecf0f1')
            )

            # Returns histogram
            returns_fig = self.create_returns_histogram(df)

            # Trade performance chart
            trade_perf_fig = self.create_trade_performance_chart(df)

            return (stats_display, holdings_table, recent_trades_display,
                    pie_fig, timeline_fig, prices_fig, status_msg, kpi_cards,
                    returns_fig, trade_perf_fig)

    def create_kpi_cards(self, stats, metrics):
        """Create KPI cards for key metrics"""
        card_style = {
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '12px',
            'textAlign': 'center',
            'minWidth': '180px',
            'margin': '10px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
        }

        pnl_color = '#27ae60' if stats['total_pnl'] >= 0 else '#e74c3c'

        cards = [
            html.Div([
                html.Div("💰", style={'fontSize': '32px', 'marginBottom': '10px'}),
                html.Div(f"${stats['total_pnl']:,.2f}",
                         style={'fontSize': '24px', 'fontWeight': 'bold', 'color': pnl_color}),
                html.Div("Total P&L", style={'fontSize': '12px', 'color': '#7f8c8d', 'marginTop': '5px'})
            ], style=card_style),

            html.Div([
                html.Div("📈", style={'fontSize': '32px', 'marginBottom': '10px'}),
                html.Div(f"{stats['pnl_percent']:+.2f}%",
                         style={'fontSize': '24px', 'fontWeight': 'bold', 'color': pnl_color}),
                html.Div("Return %", style={'fontSize': '12px', 'color': '#7f8c8d', 'marginTop': '5px'})
            ], style=card_style),

            html.Div([
                html.Div("🎯", style={'fontSize': '32px', 'marginBottom': '10px'}),
                html.Div(f"{metrics.get('win_rate', 0):.1f}%",
                         style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#3498db'}),
                html.Div("Win Rate", style={'fontSize': '12px', 'color': '#7f8c8d', 'marginTop': '5px'})
            ], style=card_style),

            html.Div([
                html.Div("📊", style={'fontSize': '32px', 'marginBottom': '10px'}),
                html.Div(f"{metrics.get('volatility', 0):.1f}%",
                         style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#9b59b6'}),
                html.Div("Volatility", style={'fontSize': '12px', 'color': '#7f8c8d', 'marginTop': '5px'})
            ], style=card_style),

            html.Div([
                html.Div("📉", style={'fontSize': '32px', 'marginBottom': '10px'}),
                html.Div(f"{metrics.get('max_drawdown', 0):.1f}%",
                         style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#e67e22'}),
                html.Div("Max Drawdown", style={'fontSize': '12px', 'color': '#7f8c8d', 'marginTop': '5px'})
            ], style=card_style),

            html.Div([
                html.Div("🔢", style={'fontSize': '32px', 'marginBottom': '10px'}),
                html.Div(f"{len(self.pm.trades)}",
                         style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#1abc9c'}),
                html.Div("Total Trades", style={'fontSize': '12px', 'color': '#7f8c8d', 'marginTop': '5px'})
            ], style=card_style)
        ]

        return cards

    def create_returns_histogram(self, df):
        """Create histogram of portfolio returns"""
        fig = go.Figure()

        if not df.empty and 'cash_after' in df.columns:
            portfolio_values = []
            for timestamp in df['timestamp'].dt.floor('h').unique():
                timestamp_data = df[df['timestamp'].dt.floor('h') == timestamp]
                if not timestamp_data['cash_after'].isna().all():
                    latest_cash = timestamp_data['cash_after'].dropna().iloc[-1]
                    stock_value = 0
                    for stock in self.pm.stocks:
                        stock_rows = timestamp_data[timestamp_data['ticker'] == stock]
                        if not stock_rows.empty and not stock_rows['position_after'].isna().all():
                            position = stock_rows['position_after'].dropna().iloc[-1]
                            price = stock_rows['close'].dropna().iloc[-1]
                            if not pd.isna(position) and not pd.isna(price):
                                stock_value += position * price
                    portfolio_values.append(stock_value + latest_cash)

            if len(portfolio_values) > 1:
                returns = np.diff(portfolio_values) / portfolio_values[:-1] * 100

                fig.add_trace(go.Histogram(
                    x=returns,
                    nbinsx=30,
                    marker=dict(
                        color=returns,
                        colorscale=[[0, '#e74c3c'], [0.5, '#f39c12'], [1, '#27ae60']],
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
                ))

                # Add mean line
                if len(returns) > 0:
                    mean_return = np.mean(returns)
                    fig.add_vline(
                        x=mean_return,
                        line_dash="dash",
                        line_color="#3498db",
                        line_width=2,
                        annotation_text=f"Mean: {mean_return:.2f}%",
                        annotation_position="top"
                    )

        fig.update_layout(
            title="Distribution of Portfolio Returns",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='#ecf0f1'),
            yaxis=dict(gridcolor='#ecf0f1'),
            showlegend=False
        )

        return fig

    def create_trade_performance_chart(self, df):
        """Create chart showing performance by stock"""
        fig = go.Figure()

        if not df.empty:
            stock_performance = {}

            for stock in self.pm.stocks:
                stock_data = df[df['ticker'] == stock].copy()
                if not stock_data.empty and len(stock_data) > 1:
                    first_price = stock_data['close'].iloc[0]
                    last_price = stock_data['close'].iloc[-1]
                    ret = ((last_price - first_price) / first_price) * 100

                    # Count trades for this stock
                    trades_count = len(df[(df['ticker'] == stock) & (df['action'] == 'BUY')])

                    stock_performance[stock] = {
                        'return': ret,
                        'trades': trades_count
                    }

            if stock_performance:
                stocks = list(stock_performance.keys())
                returns = [stock_performance[s]['return'] for s in stocks]
                trades = [stock_performance[s]['trades'] for s in stocks]

                colors = ['#27ae60' if r >= 0 else '#e74c3c' for r in returns]

                fig.add_trace(go.Bar(
                    x=stocks,
                    y=returns,
                    marker=dict(
                        color=colors,
                        line=dict(color='white', width=2)
                    ),
                    text=[f"{r:+.1f}%<br>{t} trades" for r, t in zip(returns, trades)],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>'
                ))

                fig.add_hline(
                    y=0,
                    line_dash="solid",
                    line_color="#95a5a6",
                    line_width=2
                )

        fig.update_layout(
            title="Stock Performance & Trade Count",
            xaxis_title="Stock",
            yaxis_title="Return (%)",
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='#ecf0f1'),
            yaxis=dict(gridcolor='#ecf0f1'),
            showlegend=False
        )

        return fig

    def run(self, debug=True, port=8050):
        self.app.run(debug=debug, port=port)


if __name__ == "__main__":
    # Initialize portfolio manager
    portfolio = PortfolioManager(csv_path="trading_log.csv")

    # Create and run dashboard
    dashboard = PortfolioDashboard(portfolio)
    print("Starting enhanced dashboard at http://127.0.0.1:8050")
    dashboard.run()