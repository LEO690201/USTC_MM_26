"""
SIR传染病模型实现
"""

import numpy as np
from typing import Tuple, Dict, Optional, List


class SIRModel:
    """
    基础SIR模型 (Basic SIR Model)
    
    微分方程 (Differential Equations):
    dS/dt = -beta * S * I / N
    dI/dt = beta * S * I / N - gamma * I
    dR/dt = gamma * I
    """
    
    def __init__(self, beta: float, gamma: float, N: float):
        """
        Args:
            beta: 传染率 (Transmission Rate)
            gamma: 康复率 (Recovery Rate, 1/gamma = Infectious Period)
            N: 总人口 (Total Population)
        """
        self.beta = beta
        self.gamma = gamma
        self.N = N
        self._check_parameters()
    
    def _check_parameters(self):
        if self.beta < 0 or self.gamma <= 0 or self.N <= 0:
            raise ValueError("参数必须为非负数，人口必须为正数，gamma必须为正数")
    
    @property
    def R0(self) -> float:
        """基本再生数 (Basic Reproduction Number) R0 = beta / gamma"""
        return self.beta / self.gamma
    
    def deriv(self, y: List[float], t: float) -> List[float]:
        """
        计算SIR模型的导数 (用于 odeint)
        
        Args:
            y: [S, I, R] 当前状态
            t: 当前时间
        
        Returns:
            [dS/dt, dI/dt, dR/dt]
        """
        S, I, R = y
        N = self.N
        beta = self.beta
        gamma = self.gamma
        
        # TODO: 实现标准SIR微分方程
        # dSdt = -beta * S * I / N
        # dIdt = beta * S * I / N - gamma * I
        # dRdt = gamma * I
        
        dSdt = 0.0
        dIdt = 0.0
        dRdt = 0.0
        
        return [dSdt, dIdt, dRdt]
    
    def solve(self, S0: float, I0: float, R0: float, 
              t_max: float, n_points: int = 1000) -> Dict[str, np.ndarray]:
        """
        求解模型 (Solve ODE)
        """
        from scipy.integrate import odeint
        
        t = np.linspace(0, t_max, n_points)
        y0 = [S0, I0, R0]
        
        # TODO: 调用 odeint 求解
        # solution = odeint(self.deriv, y0, t)
        
        # S = solution[:, 0]
        # I = solution[:, 1]
        # R = solution[:, 2]
        
        return {
            't': t,
            'S': np.zeros_like(t),
            'I': np.zeros_like(t),
            'R': np.zeros_like(t)
        }
    
    def get_peak_infection(self, solution: Dict) -> Tuple[float, float]:
        """
        获取感染峰值 (Peak Infection)
        
        Returns:
            (峰值时间, 峰值感染人数)
        """
        # TODO: 从 solution['I'] 中找到最大值及其对应的时间
        pass
    
    def get_final_size(self, solution: Dict) -> float:
        """
        计算最终规模 (Final Size) = R(end) / N
        """
        pass


class SIRModelWithDemographics(SIRModel):
    """
    带人口动态的SIR模型 (SIR with Vital Dynamics)
    
    方程:
    dS/dt = Lambda - mu * S - beta * S * I / N
    dI/dt = beta * S * I / N - (gamma + mu) * I
    dR/dt = gamma * I - mu * R
    """
    
    def __init__(self, beta: float, gamma: float, N: float, 
                 birth_rate: float, death_rate: float):
        """
        Args:
            birth_rate: 出生率 (Lambda)
            death_rate: 自然死亡率 (mu)
        """
        super().__init__(beta, gamma, N)
        self.birth_rate = birth_rate # Lambda
        self.death_rate = death_rate # mu
    
    def deriv(self, y: List[float], t: float) -> List[float]:
        S, I, R = y
        N = self.N # 假设 N 近似常数，或使用 S+I+R 动态计算
        
        # TODO: 实现带人口动态的微分方程
        # Lambda = self.birth_rate
        # mu = self.death_rate
        
        return [0.0, 0.0, 0.0]
    
    def get_equilibrium(self) -> Dict[str, float]:
        """
        计算平衡点 (Equilibrium Points)
        
        1. 无病平衡点 (DFE): I=0
        2. 地方病平衡点 (Endemic Equilibrium): R0 > 1 时存在
        """
        pass
