# features/engineer_features.py

import numpy as np
import pandas as pd
from numba import jit
import warnings
warnings.filterwarnings('ignore')

from utils.helper import calculate_ema, calculate_rsi, calculate_atr, calculate_sma
from features.swing_utils import (
    calculate_bb_position,
    add_candle_features,
    add_nearest_sr,
    generate_swing_labels
)
from config import EMA_PERIODS, RSI_PERIOD, Vol_Avg_Period, Fib_Pivot_Window, AVG_VOL, AVG_PRICE
from config import ATR_Period, Strong_Low_Close, Strong_High_Close
from utils.logger import get_logger


logger = get_logger(__name__)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive technical indicators with optimized performance.
    """
    # Pre-allocate list for better memory management
    all_dfs = []
    # Group by symbol for vectorized operations
    grouped = df.groupby('symbol')
    for symbol, symbol_df in grouped:
        symbol_df = symbol_df.copy()
        symbol_df.sort_values("date", inplace=True)
        # Reset index for consistent indexing
        symbol_df.reset_index(drop=True, inplace=True)
        # Extract OHLCV arrays once for reuse
        high = symbol_df['high'].values
        low = symbol_df['low'].values
        close = symbol_df['close'].values
        open_price = symbol_df['open'].values
        volume = symbol_df['volume'].values if 'volume' in symbol_df.columns else None
        
        # ➕ Basic Technical Indicators (vectorized)
        symbol_df = add_basic_indicators_vectorized(symbol_df, close, high, low, open_price,volume)
       
        # ➕ Candlestick Pattern Features
        symbol_df = add_candle_features(symbol_df)
        
        # ➕ Bollinger Band position
        symbol_df["bb_position"] = calculate_bb_position(symbol_df["close"], 20)
        
        # ➕ Support & Resistance        
        symbol_df = add_nearest_sr(symbol_df)
    
        # ➕ Swing Labels
        symbol_df = generate_swing_labels(symbol_df)
        # symbol_df = symbol_df.drop(columns=['max_return', 'min_return', 'risk_reward_ratio'])
        
        

        # symbol_df['strong_rejection'] = (
        #     (symbol_df['close_compared_to_previous'] < Strong_Low_Close) |
        #     (symbol_df['close_compared_to_previous'] > Strong_High_Close) |
        #     (symbol_df['ema50_ema200'] >1.02)&
        #     (symbol_df['rsi']< 45)&
        #     (symbol_df['ema20_price'] > 0.98)
        # ).astype(int)

        symbol_df['strong_rejection'] = (
            (symbol_df['close_compared_to_previous'] < Strong_Low_Close) |
            (symbol_df['close_compared_to_previous'] > Strong_High_Close) |
            (symbol_df['ema50_ema200'] > 1.02)
        ).astype(int)

        all_dfs.append(symbol_df)
        # logger.info(f" Features added for {symbol}")
        # pd.concat(all_dfs).to_csv('features.csv', index=False)
    return pd.concat(all_dfs, ignore_index=True)

def calculate_rolling_fib_pivots(df, window):
    """Calculate rolling Fibonacci pivot levels using a rolling window."""
    
    # Calculate rolling high, low, close over window periods
    rolling_high = df['high'].rolling(window=window).max()
    rolling_low = df['low'].rolling(window=window).min()
    rolling_close = df['close'].rolling(window=window).mean()
    
    # Calculate pivot point
    pivot = (rolling_high + rolling_low + rolling_close) / 3
    
    # Fibonacci levels
    fib_range = rolling_high - rolling_low
    fib_r1 = pivot + 0.382 * fib_range
    fib_r2 = pivot + 0.618 * fib_range
    fib_s1 = pivot - 0.382 * fib_range
    fib_s2 = pivot - 0.618 * fib_range
    
    # Calculate distances as percentages
    df['pivot'] = pivot
    df['fib_r1'] = fib_r1
    df['fib_r2'] = fib_r2
    df['fib_s1'] = fib_s1
    df['fib_s2'] = fib_s2
    df['fib_pivot_distance_pct'] = (df['close'] - pivot) / pivot * 100
    df['fib_r1_distance_pct'] = (df['close'] - fib_r1) / fib_r1 * 100
    df['fib_r2_distance_pct'] = (df['close'] - fib_r2) / fib_r2 * 100
    df['fib_s1_distance_pct'] = (df['close'] - fib_s1) / fib_s1 * 100
    df['fib_s2_distance_pct'] = (df['close'] - fib_s2) / fib_s2 * 100
    
    return df

def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    fast_length = 12
    slow_length = 26
    signal_length = 9

    def compute_macd(group):
        close = group['close']
        ema_fast = close.ewm(span=fast_length, adjust=False).mean()
        ema_slow = close.ewm(span=slow_length, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal_length, adjust=False).mean()
        macd_hist = macd - macd_signal
        macd_crossover = macd > macd_signal
        macd_cross_signal = macd_crossover.ne(macd_crossover.shift(1))
        macd_signal_type = macd_crossover & macd_cross_signal
        macd_signal_type = macd_signal_type.map({True: 'buy'}).fillna(
            macd_cross_signal.map({True: 'sell'}).fillna(''))

        macd_ls_signal =  (macd_crossover).astype(int)
        group['macd_ls_signal'] = macd_ls_signal
        
        return group
    return df.groupby('symbol', group_keys=False).apply(compute_macd)

def add_basic_indicators_vectorized(df: pd.DataFrame, close: np.ndarray, high: np.ndarray, low: np.ndarray, open_price: np.ndarray, volume:np.ndarray) -> pd.DataFrame:
    """Add basic indicators using vectorized operations."""
    # EMA calculations (vectorized)
    for period in EMA_PERIODS:
        df[f"ema{period}"] = calculate_ema(df["close"], period)

    for period in EMA_PERIODS:
        df[f"sma{period}"] = calculate_sma(df["close"], period)
    
    df['ema20_ema50'] = df['ema20']/df['ema50']
    df['sma20_sma50'] = df['sma20']/df['sma50']
    df['ema50_ema200'] = df['ema50']/df['ema200']
    df['sma50_sma200'] = df['sma50']/df['sma200']
    df['ema20_price'] =df['ema20']/df['close']
    df['ema50_price'] =df['ema50']/df['close']
    df['ema200_price'] =df['ema200']/df['close']
    df['sma20_price'] =df['sma20']/df['close']
    df['sma50_price'] =df['sma50']/df['close']
    df['sma200_price'] =df['sma200']/df['close']
    # RSI and ATR
    df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
    df['atr'] = calculate_atr(df, ATR_Period)  # keep atr column for evaluation; we'll add atr_pct
    df['atr_pct'] = df['atr'] / df['close']
    if 'volume' in df.columns:
        vol = df['volume'].fillna(0)
        df['vol_by_avg_vol'] = vol / vol.rolling(Vol_Avg_Period, min_periods=1).mean()
        df['obv'] = calculate_obv(df['close'], vol)
        df['vol_change_5d'] = vol.pct_change(5)
    else:
        df['vol_by_avg_vol'] = 0
        df['obv'] = 0
        df['vol_change_5d'] = 0

    # Short-term returns (momentum context)
    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_3d'] = df['close'].pct_change(3)
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_10d'] = df['close'].pct_change(10)

    # Volatility (rolling std of returns)
    df['vol_5d'] = df['close'].pct_change().rolling(5).std()
    df['vol_10d'] = df['close'].pct_change().rolling(10).std()

    # RSI slope (momentum increasing/decreasing)
    df['rsi_3_slope'] = rolling_slope(df['rsi'], 3)
    df['rsi_5_slope'] = rolling_slope(df['rsi'], 5)

    # EMA crossover signals and fresh cross flags
    df['ema20_above_ema50'] = (df['ema20'] > df['ema50']).astype(int)
    df['ema20_50_cross_up'] = ((df['ema20'] > df['ema50']) & (df['ema20'].shift(1) <= df['ema50'].shift(1))).astype(int)
    df['sma20_50_cross_up'] = ((df['sma20'] > df['sma50']) & (df['sma20'].shift(1) <= df['sma50'].shift(1))).astype(int)
    df['ema20_50_cross_down'] = ((df['ema20'] < df['ema50']) & (df['ema20'].shift(1) >= df['ema50'].shift(1))).astype(int)

    range_hl = df['high'] - df['low']
    df['close_position_in_range'] = np.where(range_hl != 0, (df['close'] - df['low']) / range_hl, 0.5)

    df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

    df = calculate_rolling_fib_pivots(df,window=Fib_Pivot_Window)
    df = add_macd(df)
    return df

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    # On-balance volume (vectorized)
    direction = np.sign(close.diff().fillna(0))
    obv = (direction * volume.fillna(0)).cumsum()
    return obv

def rolling_slope(series: pd.Series, period: int) -> pd.Series:
    # Simple slope approximation: (value_now - value_n_steps_ago) / (period-1)
    return (series - series.shift(period-1)) / (period-1)
