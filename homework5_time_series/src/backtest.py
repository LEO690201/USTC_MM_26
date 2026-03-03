"""
简单的回测框架
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


class Backtest:
    """
    策略回测系统
    """
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000.0,
                 commission: float = 0.001):
        """
        Args:
            data: 包含 'close' 列的DataFrame
            initial_capital: 初始资金
            commission: 交易佣金比例
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        
        # 结果存储
        self.portfolio_value = []
        self.returns = []
        self.positions = []
        self.cash = initial_capital
    
    def run(self, signals: pd.Series) -> pd.DataFrame:
        """
        运行回测
        
        Args:
            signals: 交易信号序列 (1=买入/持有, -1=卖出/空仓, 0=不变)
        """
        self.data['signal'] = signals
        position = 0  # 当前持仓数量
        
        for i, (idx, row) in enumerate(self.data.iterrows()):
            price = row['close']
            signal = row['signal']
            
            # 简单的全仓买卖策略
            if signal == 1 and position == 0:
                # 买入
                shares = self.cash // (price * (1 + self.commission))
                cost = shares * price * (1 + self.commission)
                self.cash -= cost
                position = shares
                
            elif signal == -1 and position > 0:
                # 卖出
                revenue = position * price * (1 - self.commission)
                self.cash += revenue
                position = 0
            
            # 记录每日市值
            current_value = self.cash + position * price
            self.portfolio_value.append(current_value)
            self.positions.append(position)
        
        self.data['portfolio_value'] = self.portfolio_value
        self.data['daily_return'] = self.data['portfolio_value'].pct_change().fillna(0)
        
        return self.data
    
    def plot_equity_curve(self):
        """绘制资金曲线"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['portfolio_value'], label='Strategy')
        
        # 对比基准 (买入持有)
        benchmark_ret = (self.data['close'] / self.data['close'].iloc[0])
        plt.plot(self.data.index, benchmark_ret * self.initial_capital, 
                 label='Benchmark (Buy & Hold)', alpha=0.6)
        
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def calculate_metrics(self) -> Dict[str, float]:
        """计算策略指标"""
        total_return = (self.data['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        annual_return = total_return * (252 / len(self.data))
        
        daily_returns = self.data['daily_return']
        volatility = daily_returns.std() * np.sqrt(252)
        
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
        
        # 最大回撤
        cum_max = self.data['portfolio_value'].cummax()
        drawdown = (self.data['portfolio_value'] - cum_max) / cum_max
        max_drawdown = drawdown.min()
        
        return {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }
