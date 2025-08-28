# model/backtest_model.py

import pandas as pd
import numpy as np
import joblib
import logging
import warnings
from datetime import datetime
from pathlib import Path
import os
from config import CONFIDENCE_THRESHOLD, MODEL_FILE
from time import sleep

warnings.filterwarnings('ignore')

# ---- Logging Setup ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Constants ----
MODEL_FILE = Path(MODEL_FILE)
HISTORICAL_DATA_FILE = "data/historical_data.parquet"

# ---- Backtester Class ----
class SwingTradingBacktester:
    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.trades = []
        self.cash = initial_capital
        self.total_portfolio_value = initial_capital

    def load_model(self, model_path):
        try:
            self.model_pipeline = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def prepare_features(self, df):
        try:
            full_feature_list = self.model_pipeline['all_features']
            X = df[full_feature_list].copy()
            X = X.dropna()
            if X.empty:
                return None
            df = df.loc[X.index]
            self.features_for_prediction = X
            return df
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None

    def generate_signals(self, df, X=None):
        try:
            pipeline = self.model_pipeline['pipeline']

            if X is None:
                if not hasattr(self, 'features_for_prediction'):
                    logger.error("Features not prepared.")
                    return None
                X = self.features_for_prediction

            if X.shape[0] != df.shape[0]:
                logger.error("Mismatch between features and target DataFrame rows")
                return None

            df = df.copy()
            df['signal'] = pipeline.predict(X)
            df['confidence'] = pipeline.predict_proba(X)[:, 1]
            return df

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return None


    def run_backtest_with_daily_signals(self, df, holding_period=21, stop_loss_pct=0.05, take_profit_pct=0.15):
        logger.info("Starting backtest...")
        df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
        unique_dates = df['date'].drop_duplicates().sort_values().tolist()

        open_trades = []
        closed_trades = []

        for i in range(1, len(unique_dates)):
            current_date = unique_dates[i]
            prev_date = unique_dates[i - 1]

            # Prepare training data till current day (simulate real-time)
            past_data = df[df['date'] <= current_date].copy()

            # 2. Feature engineering on all past data
            past_data = self.prepare_features(past_data)
            
            if past_data is None or past_data.empty:
                continue
        
            # 3. Select only today’s row after feature engineering
            today_df = past_data[past_data['date'] == current_date].copy()
            if i == 40:
                today_df.to_csv('today_df_with_feature.csv')
            if today_df.empty:
                continue

            # 4. Prepare features just for today’s row
            X_today = today_df[self.model_pipeline['all_features']].copy()

            # 5. Predict only for today
            signal_df = self.generate_signals(today_df, X_today)
            if signal_df is None or signal_df.empty:
                continue

            # ---- New Buys ----
            for _, row in signal_df.iterrows():
                if row['signal'] == 1 and row['confidence'] >= CONFIDENCE_THRESHOLD:
                    already_open = any(trade['symbol'] == row['symbol'] for trade in open_trades)
                    if not already_open:
                        open_trades.append({
                            'symbol': row['symbol'],
                            'buy_price': row['close'],
                            'buy_date': prev_date,
                            'holding_days_left': holding_period,
                            'stop_loss': row['close'] * (1 - 0.05),
                            'target': row['close'] * (1 + 0.15),
                            'max_holding_period': holding_period
                        })

            # ---- Monitor Existing Trades ----
            day_prices = df[df['date'] == current_date]
            updated_trades = []
            for trade in open_trades:
                symbol_data = day_prices[day_prices['symbol'] == trade['symbol']]
                if symbol_data.empty:
                    trade['holding_days_left'] -= 1
                    updated_trades.append(trade)
                    continue

                row = symbol_data.iloc[0]
                high, low, close = row['high'], row['low'], row['close']
                exit_reason, exit_price = None, None

                if close <= trade['stop_loss']:
                    exit_reason, exit_price = 'Stop Loss', trade['stop_loss']
                elif high >= trade['target']:
                    exit_reason, exit_price = 'Target Hit', trade['target']
                elif trade['holding_days_left'] <= 0:
                    exit_reason, exit_price = 'Max Hold', close

                if exit_reason:
                    closed_trades.append({
                        'Symbol': trade['symbol'],
                        'Buy': round(trade['buy_price'], 2),
                        'Buy Date': trade['buy_date'],
                        'Sell': round(exit_price, 2),
                        'Sell Date': current_date,
                        'P&L': round(((exit_price - trade['buy_price'])*100/trade['buy_price']), 2),
                        'Exit Reason': exit_reason,
                        'Holding Days': trade['max_holding_period'] - trade['holding_days_left']
                    })
                else:
                    trade['holding_days_left'] -= 1
                    updated_trades.append(trade)
            open_trades = updated_trades

        results_df = pd.DataFrame(closed_trades)
        if results_df.empty:
            logger.warning("No trades executed.")
            return results_df, 0, 0, 0

        total_trades = len(results_df)
        success_rate = 100 * (results_df['P&L'] > 0).sum() / total_trades
        avg_pnl = results_df['P&L'].mean()
        total_pnl = results_df['P&L'].sum()

        logger.info(f"Backtest completed: {total_trades} trades | Success: {success_rate:.2f}% | Total P&L: {total_pnl:.2f}")
        results_df.to_csv("daily_backtest_trades.csv", index=False)
        return results_df, success_rate, total_pnl, avg_pnl


# ---- Utility Runner ----
def run_backtest(df, model_path=None, **kwargs):
    backtester = SwingTradingBacktester(
        initial_capital=kwargs.get('initial_capital', 100000),
        commission=kwargs.get('commission', 0.001),
        slippage=kwargs.get('slippage', 0.0005)
    )

    if not backtester.load_model(model_path or MODEL_FILE):
        return None

    return backtester.run_backtest_with_daily_signals(
        df,
        holding_period=kwargs.get('holding_period', 21),
        stop_loss_pct=kwargs.get('stop_loss_pct', 0.05),
        take_profit_pct=kwargs.get('take_profit_pct', 0.15)
    )


# ---- CLI Execution ----
if __name__ == "__main__":
    df = pd.read_parquet(HISTORICAL_DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    results, success_rate, total_pnl, avg_pnl = run_backtest(df)
    print("\nBacktest Results:")
    print(results)
