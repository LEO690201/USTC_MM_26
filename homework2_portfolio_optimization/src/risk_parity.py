"""
风险平价投资组合
"""

import numpy as np
from typing import Tuple
from scipy.optimize import minimize


class RiskParityPortfolio:
    """
    风险平价组合 - 各资产对组合风险贡献相等
    """
    
    def __init__(self, cov_matrix: np.ndarray):
        """
        Args:
            cov_matrix: 协方差矩阵
        """
        self.cov_matrix = cov_matrix
        self.n_assets = cov_matrix.shape[0]
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """组合波动率"""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def risk_contribution(self, weights: np.ndarray) -> np.ndarray:
        """
        计算各资产的风险贡献
        
        RC_i = w_i * (Σw)_i / σ_p
        
        Returns:
            风险贡献向量 (n,)
        """
        # TODO: 实现
        # 步骤1: 计算组合波动率
        vol = self.portfolio_volatility(weights)
        
        # 步骤2: 计算边际风险贡献 (Σw)
        marginal_risk = np.dot(self.cov_matrix, weights)
        
        # 步骤3: 计算风险贡献
        # RC = w * marginal_risk / vol
        risk_contrib = np.zeros(self.n_assets)
        
        return risk_contrib
    
    def risk_parity_error(self, weights: np.ndarray) -> float:
        """
        风险平价目标函数
        
        目标: 最小化各资产风险贡献与平均风险贡献的偏离
        """
        # TODO: 实现
        # 目标: minimize Σ (RC_i - σ_p/n)²
        # 或使用 RC_i / RC_j = 1 的约束
        
        rc = self.risk_contribution(weights)
        vol = self.portfolio_volatility(weights)
        n = self.n_assets
        
        target_rc = vol / n  # 目标风险贡献
        
        # 计算误差
        error = 0.0
        
        return error
    
    def optimize(self, method: str = 'closed_form') -> np.ndarray:
        """
        求解风险平价组合
        
        Args:
            method: 'closed_form' 或 'iterative' 或 'optimization'
        
        Returns:
            权重向量
        """
        if method == 'optimization':
            return self._optimization_solve()
        elif method == 'iterative':
            return self._iterative_solve()
        elif method == 'closed_form':
            return self._closed_form_solve()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _optimization_solve(self) -> np.ndarray:
        """使用优化求解"""
        # TODO: 实现
        # minimize risk_parity_error
        # subject to: sum(w) = 1, w > 0
        
        def objective(w):
            return self.risk_parity_error(w)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((1e-8, 1) for _ in range(self.n_assets))
        
        # 使用等权作为初始值
        w0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000})
        
        return result.x
    
    def _iterative_solve(self, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
        """
        迭代求解 (梯度下降法)
        
        公式: w_i ∝ 1 / (Σw)_i
        """
        # TODO: 实现
        # 迭代更新: w_i = c / (Σw)_i, 然后归一化
        
        w = np.ones(self.n_assets) / self.n_assets
        
        for _ in range(max_iter):
            # 保存旧权重
            w_old = w.copy()
            
            # 计算风险边际
            marginal_risk = np.dot(self.cov_matrix, w)
            
            # 更新权重 (逆风散)
            w = 1.0 / marginal_risk
            w = w / w.sum()  # 归一化
            
            # 检查收敛
            if np.abs(w - w_old).max() < tol:
                break
        
        return w
    
    def _closed_form_solve(self) -> np.ndarray:
        """
        闭式解 (需使用数值方法求解非线性方程)
        """
        # TODO: 实现
        # 对于简单情况可近似求解
        # 更一般使用优化方法
        return self._optimization_solve()
    
    def get_risk_contribution(self, weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        获取风险贡献及百分比
        """
        rc = self.risk_contribution(weights)
        total_risk = self.portfolio_volatility(weights)
        rc_percent = rc / total_risk if total_risk > 0 else rc
        return rc, rc_percent


class RiskBudgetPortfolio:
    """
    风险预算组合 - 指定各资产风险贡献比例
    """
    
    def __init__(self, cov_matrix: np.ndarray, risk_budget: np.ndarray):
        """
        Args:
            risk_budget: 目标风险贡献比例 (n,), sum = 1
        """
        self.cov_matrix = cov_matrix
        self.risk_budget = risk_budget
        self.n_assets = len(risk_budget)
        
        if not np.isclose(risk_budget.sum(), 1.0):
            raise ValueError("风险预算总和必须为1")
    
    def optimize(self) -> np.ndarray:
        """
        求解风险预算组合
        """
        rp = RiskParityPortfolio(self.cov_matrix)
        
        def objective(w):
            rc = rp.risk_contribution(w)
            vol = rp.portfolio_volatility(w)
            
            if vol < 1e-10:
                return 0.0
            
            rc_percent = rc / vol
            
            # 误差: sum((rc_percent_i - budget_i)²)
            error = np.sum((rc_percent - self.risk_budget) ** 2)
            return error
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((1e-8, 1) for _ in range(self.n_assets))
        
        w0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x
