# refactored_fyers_swing/utils/helper.py

import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
from config import ATR_Period

def get_today_date_str():
    """Returns today's date in yyyy-mm-dd format."""
    return datetime.now().strftime("%Y-%m-%d")

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average (SMA)."""
    return series.rolling(window=period, min_periods=period).mean()

def calculate_rsi(series: pd.Series, period) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df: pd.DataFrame, ATR_period) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=ATR_period).mean()
    return atr


def delete_csv(folder_path, keep_file):
    
    for ext in ("*.csv", "*.xlsx"):
        for file_path in glob.glob(os.path.join(folder_path, ext)):
            if os.path.basename(file_path) != keep_file:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    print("Cleanup complete.")
