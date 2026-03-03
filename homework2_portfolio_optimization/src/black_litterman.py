"""
Black-Litterman模型实现
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from mean_variance import MeanVariancePortfolio


class BlackLittermanModel:
    """
    Black-Litterman全球资产配置模型
    
    后验收益公式:
    mu_BL = [(tau * Sigma)^-1 + P^T * Omega^-1 * P]^-1 * [(tau * Sigma)^-1 * Pi + P^T * Omega^-1 * Q]
    """
    
    def __init__(self, cov_matrix: np.ndarray, market_weights: np.ndarray,
                 risk_aversion: float = 2.5, risk_free_rate: float = 0.0):
        """
        Args:
            cov_matrix: 协方差矩阵 (Sigma)
            market_weights: 市值加权组合权重 (w_mkt)
            risk_aversion: 风险厌恶系数 (lambda)
            risk_free_rate: 无风险利率
        """
        self.cov_matrix = cov_matrix
        self.market_weights = market_weights
        self.risk_aversion = risk_aversion
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(market_weights)
        
        # 计算市场均衡收益 Pi
        self.pi = self._calc_equilibrium_returns()
    
    def _calc_equilibrium_returns(self) -> np.ndarray:
        """
        计算市场均衡收益 (Implied Equilibrium Returns)
        
        Pi = lambda * Sigma * w_mkt
        """
        # TODO: 实现 Pi 的计算
        # Pi = self.risk_aversion * self.cov_matrix @ self.market_weights
        
        return np.zeros(self.n_assets)
    
    def blend_views(self, P: np.ndarray, Q: np.ndarray, 
                    tau: float = 0.05, 
                    Omega: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        合并观点 (Blend Views)
        
        Args:
            P: 观点矩阵 (K x N)
            Q: 观点收益向量 (K x 1)
            tau: 均衡收益不确定性系数
            Omega: 观点不确定性矩阵 (K x K). 如果为 None，建议估算: diag(P * (tau*Sigma) * P^T)
        
        Returns:
            (posterior_mu, posterior_sigma)
        """
        # TODO: 1. 如果 Omega 为空，自动计算
        # Omega = diag(P @ (tau * self.cov_matrix) @ P.T)
        
        # TODO: 2. 计算后验期望收益 mu_BL
        # 使用 np.linalg.solve 或 np.linalg.inv
        # term1 = inv(tau * Sigma) + P.T @ inv(Omega) @ P
        # term2 = inv(tau * Sigma) @ Pi + P.T @ inv(Omega) @ Q
        # mu_BL = inv(term1) @ term2
        
        posterior_mu = np.zeros(self.n_assets)
        
        # (选做) TODO: 3. 计算后验协方差矩阵 Sigma_BL
        # Sigma_BL = inv(term1) + Sigma
        posterior_sigma = self.cov_matrix
        
        return posterior_mu, posterior_sigma
    
    def get_optimized_portfolio(self, P: np.ndarray, Q: np.ndarray, 
                                tau: float = 0.05,
                                Omega: Optional[np.ndarray] = None) -> np.ndarray:
        """基于BL后验参数优化组合"""
        mu_bl, sigma_bl = self.blend_views(P, Q, tau, Omega)
        
        # 使用后验参数进行均值方差优化
        mv = MeanVariancePortfolio(mu_bl, sigma_bl, self.risk_free_rate)
        return mv.maximize_sharpe()
