"""
SEIR传染病模型实现
"""

import numpy as np
from typing import Dict, Tuple
from sir_model import SIRModel


class SEIRModel:
    """
    SEIR模型 (考虑潜伏期)
    
    S -> E (暴露) -> I (感染) -> R (康复)
    
    Attributes:
        beta: 传染率
        sigma: 潜伏期转化率 (1/sigma 为平均潜伏期)
        gamma: 康复率
        N: 总人口
    """
    
    def __init__(self, beta: float, sigma: float, gamma: float, N: float):
        """
        初始化SEIR模型
        """
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.N = N
        
        if beta <= 0 or sigma <= 0 or gamma <= 0 or N <= 0:
            raise ValueError("参数必须为正数")
    
    @property
    def R0(self) -> float:
        """有效再生数"""
        # TODO: 计算SEIR模型的R0
        # 提示: R0 = beta / gamma * 某些因子
        return 0.0
    
    def deriv(self, y: np.ndarray, t: np.ndarray) -> Tuple[float, float, float, float]:
        """
        SEIR模型导数
        
        Args:
            y: [S, E, I, R] 状态向量
        
        Returns:
            [dS/dt, dE/dt, dI/dt, dR/dt]
        """
        S, E, I, R = y
        N = self.N
        
        # TODO: 实现SEIR微分方程
        # 提示: 
        # dS/dt = -beta * S * I / N
        # dE/dt = beta * S * I / N - sigma * E
        # dI/dt = sigma * E - gamma * I
        # dR/dt = gamma * I
        
        return 0.0, 0.0, 0.0, 0.0
    
    def solve(self, S0: float, E0: float, I0: float, R0: float,
              t_max: float, n_points: int = 1000) -> Dict[str, np.ndarray]:
        """
        求解SEIR模型
        """
        from scipy.integrate import odeint
        
        t = np.linspace(0, t_max, n_points)
        y0 = [S0, E0, I0, R0]
        
        # TODO: 使用odeint求解
        # solution = odeint(self.deriv, y0, t)
        
        return {
            't': t,
            'S': np.zeros(n_points),
            'E': np.zeros(n_points),
            'I': np.zeros(n_points),
            'R': np.zeros(n_points)
        }
    
    def compare_with_sir(self, S0: float, I0: float, R0: float,
                         t_max: float) -> Dict[str, np.ndarray]:
        """
        对比SIR和SEIR模型结果
        
        Args:
            S0, I0, R0: 初始条件 (E0=0)
            t_max: 模拟时间
        """
        # TODO: 运行SEIR模型
        # 创建等效SIR模型并运行
        # 返回对比结果
        pass


class SEIRModelWithDemographics(SEIRModel):
    """
    带人口动态的SEIR模型
    """
    
    def __init__(self, beta: float, sigma: float, gamma: float, N: float,
                 birth_rate: float, death_rate: float):
        super().__init__(beta, sigma, gamma, N)
        self.birth_rate = birth_rate
        self.death_rate = death_rate
    
    def deriv(self, y: np.ndarray, t: np.ndarray) -> Tuple:
        """带人口动态的SEIR"""
        S, E, I, R = y
        N = self.N
        mu = self.death_rate
        Lambda = self.birth_rate
        
        # TODO: 添加出生和死亡项
        # 注意: 考虑因死亡导致的各类人口变化
        
        return 0.0, 0.0, 0.0, 0.0
