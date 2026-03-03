"""
技术指标计算模块
"""

import numpy as np
import pandas as pd
from typing import Tuple


class TechnicalIndicators:
    """技术指标计算"""
    
    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """
        简单移动平均 (Simple Moving Average)
        
        SMA = (P1 + P2 + ... + Pn) / n
        """
        return series.rolling(window=window).mean()
    
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        """
        指数移动平均 (Exponential Moving Average)
        
        EMA_t = α * P_t + (1-α) * EMA_{t-1}
        其中 α = 2 / (span + 1)
        """
        return series.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def macd(series: pd.Series, 
             fast_period: int = 12, 
             slow_period: int = 26,
             signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence)
        
        MACD = EMA_fast - EMA_slow
        Signal = EMA(MACD, 9)
        Histogram = MACD - Signal
        """
        ema_fast = TechnicalIndicators.ema(series, fast_period)
        ema_slow = TechnicalIndicators.ema(series, slow_period)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI (Relative Strength Index)
        
        RSI = 100 - 100 / (1 + RS)
        RS = 平均涨幅 / 平均跌幅
        """
        delta = series.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def bollinger_bands(series: pd.Series, 
                        window: int = 20, 
                        num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        布林带 (Bollinger Bands)
        
        Middle = SMA(window)
        Upper = Middle + num_std * STD
        Lower = Middle - num_std * STD
        """
        middle = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        
        upper = middle + num_std * std
        lower = middle - num_std * std
        
        return upper, middle, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, 
            close: pd.Series, period: int = 14) -> pd.Series:
        """
        ATR (Average True Range)
        
        True Range = max(H-L, |H-PC|, |L-PC|)
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, 
                   close: pd.Series, period: int = 14,
                   smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        随机指标 (Stochastic Oscillator)
        
        %K = 100 * (C - L14) / (H14 - L14)
        %D = SMA(%K, 3)
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        
        k_smoothed = k.rolling(window=smooth_k).mean()
        d = k_smoothed.rolling(window=smooth_d).mean()
        
        return k_smoothed, d
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        OBV (On-Balance Volume)
        
        如果今天收盘 > 昨天收盘, OBV = OBV_昨天 + Volume
        如果今天收盘 < 昨天收盘, OBV = OBV_昨天 - Volume
        """
        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()
        
        return obv


class TradingStrategy:
    """交易策略基类"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: 包含OHLCV的DataFrame
        """
        self.data = data.copy()
        self.signals = pd.Series(0, index=data.index)
    
    def generate_signals(self) -> pd.Series:
        """生成交易信号"""
        raise NotImplementedError


class MovingAverageCrossover(TradingStrategy):
    """均线交叉策略"""
    
    def __init__(self, data: pd.Series, short_window: int = 20, 
                 long_window: int = 50):
        super().__init__(data.to_frame(name='close'))
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self) -> pd.Series:
        """生成信号: 1=买入, -1=卖出, 0=持有"""
        close = self.data['close']
        
        sma_short = TechnicalIndicators.sma(close, self.short_window)
        sma_long = TechnicalIndicators.sma(close, self.long_window)
        
        # 金叉买入, 死叉卖出
        self.signals = pd.Series(0, index=close.index)
        self.signals[sma_short > sma_long] = 1
        self.signals[sma_short < sma_long] = -1
        
        return self.signals


class MomentumStrategy(TradingStrategy):
    """动量策略"""
    
    def __init__(self, data: pd.Series, lookback: int = 20, 
                 threshold: float = 0.02):
        super().__init__(data.to_frame(name='close'))
        self.lookback = lookback
        self.threshold = threshold
    
    def generate_signals(self) -> pd.Series:
        """动量策略: 过去收益为正则买入"""
        close = self.data['close']
        
        returns = close.pct_change(self.lookback)
        
        self.signals = pd.Series(0, index=close.index)
        self.signals[returns > self.threshold] = 1
        self.signals[returns < -self.threshold] = -1
        
        return self.signals


class RSIStrategy(TradingStrategy):
    """RSI策略"""
    
    def __init__(self, data: pd.Series, period: int = 14,
                 oversold: float = 30, overbought: float = 70):
        super().__init__(data.to_frame(name='close'))
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self) -> pd.Series:
        """RSI策略: RSI<30买入, RSI>70卖出"""
        close = self.data['close']
        
        rsi = TechnicalIndicators.rsi(close, self.period)
        
        self.signals = pd.Series(0, index=close.index)
        self.signals[rsi < self.oversold] = 1
        self.signals[rsi > self.overbought] = -1
        
        return self.signals
