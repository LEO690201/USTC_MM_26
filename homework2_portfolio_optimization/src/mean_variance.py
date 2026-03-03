"""
均值-方差投资组合优化
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.optimize import minimize


class MeanVariancePortfolio:
    """
    Markowitz均值-方差模型
    """
    
    def __init__(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                 risk_free_rate: float = 0.0):
        """
        Args:
            expected_returns: 预期收益向量 (n,)
            cov_matrix: 协方差矩阵 (n, n)
            risk_free_rate: 无风险利率
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
        
        self._validate_inputs()
    
    def _validate_inputs(self):
        """验证输入有效性"""
        if self.cov_matrix.shape[0] != self.cov_matrix.shape[1]:
            raise ValueError("协方差矩阵必须是方阵")
        if len(self.expected_returns) != self.cov_matrix.shape[0]:
            raise ValueError("收益向量维度与协方差矩阵不匹配")
    
    def portfolio_return(self, weights: np.ndarray) -> float:
        """计算组合收益"""
        return np.dot(weights, self.expected_returns)
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """计算组合波动率 (标准差)"""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def portfolio_sharpe(self, weights: np.ndarray) -> float:
        """计算夏普比率"""
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        return (ret - self.risk_free_rate) / vol
    
    def minimize_volatility(self) -> np.ndarray:
        """
        最小方差组合
        """
        # TODO: 实现最小方差组合优化
        # 目标函数: minimize w'Σw
        # 约束: sum(w) = 1, w >= 0
        
        def objective(w):
            return self.portfolio_volatility(w) ** 2
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # 初始权重: 等权
        w0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x
    
    def maximize_sharpe(self) -> np.ndarray:
        """
        最大夏普比率组合
        """
        # TODO: 实现
        def objective(w):
            return -self.portfolio_sharpe(w)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        w0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        return result.x
    
    def efficient_return(self, target_return: float) -> np.ndarray:
        """
        给定目标收益下的最小方差组合
        """
        # TODO: 实现
        # 约束: portfolio_return(w) = target_return
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: self.portfolio_return(w) - target_return}
        ]
        
        def objective(w):
            return self.portfolio_volatility(w) ** 2
        
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        w0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        return result.x
    
    def efficient_frontier(self, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算有效前沿
        """
        # 计算收益范围
        min_ret = self.expected_returns.min()
        max_ret = self.expected_returns.max()
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        volatilities = []
        returns = []
        
        for target in target_returns:
            try:
                w = self.efficient_return(target)
                volatilities.append(self.portfolio_volatility(w))
                returns.append(target)
            except:
                continue
        
        return np.array(volatilities), np.array(returns)
    
    def maximum_return(self, max_weight: float = 1.0) -> np.ndarray:
        """
        最大收益组合 (可能有上限约束)
        """
        # TODO: 实现
        def objective(w):
            return -self.portfolio_return(w)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, max_weight) for _ in range(self.n_assets))
        w0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        return result.x


def calc_beta(asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
    """计算Beta"""
    # TODO: 使用协方差公式计算beta
    # β = Cov(r_i, r_m) / Var(r_m)
    pass


def calc_alpha(asset_returns: np.ndarray, market_returns: np.ndarray, 
               risk_free_rate: float) -> float:
    """计算Jensen's Alpha"""
    # TODO: α = E(r_i) - [r_f + β(E(r_m) - r_f)]
    pass
