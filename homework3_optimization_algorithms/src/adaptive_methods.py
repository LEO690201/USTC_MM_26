"""
自适应学习率优化算法
"""

import numpy as np
from typing import Callable, List
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    """优化结果"""
    weights: np.ndarray
    loss_history: List[float]
    grad_norm_history: List[float]
    n_iterations: int

class AdaGrad:
    """
    AdaGrad: 自适应学习率
    
    r_t = r_{t-1} + g_t^2
    w_{t+1} = w_t - lr / (sqrt(r_t) + epsilon) * g_t
    """
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8,
                 max_iter: int = 1000):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.max_iter = max_iter
    
    def optimize(self, loss_fn: Callable, grad_fn: Callable,
                 w0: np.ndarray) -> OptimizationResult:
        w = w0.copy()
        grad_squared_sum = np.zeros_like(w)
        loss_history = []
        grad_norm_history = []
        
        for i in range(self.max_iter):
            grad = grad_fn(w)
            loss = loss_fn(w)
            
            loss_history.append(loss)
            grad_norm_history.append(np.linalg.norm(grad))
            
            # TODO: AdaGrad更新
            # grad_squared_sum += grad**2
            # w -= self.lr * grad / (np.sqrt(grad_squared_sum) + self.epsilon)
            
            if np.linalg.norm(grad) < 1e-6:
                break
        
        return OptimizationResult(w, loss_history, grad_norm_history, i + 1)


class RMSProp:
    """
    RMSProp: 指数移动平均
    
    r_t = rho * r_{t-1} + (1-rho) * g_t^2
    w_{t+1} = w_t - lr / (sqrt(r_t) + epsilon) * g_t
    """
    
    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9,
                 epsilon: float = 1e-8, max_iter: int = 1000):
        self.lr = learning_rate
        self.rho = rho  # 衰减率 (alpha in some texts)
        self.epsilon = epsilon
        self.max_iter = max_iter
    
    def optimize(self, loss_fn: Callable, grad_fn: Callable,
                 w0: np.ndarray) -> OptimizationResult:
        w = w0.copy()
        avg_sq_grad = np.zeros_like(w)
        loss_history = []
        grad_norm_history = []
        
        for i in range(self.max_iter):
            grad = grad_fn(w)
            loss = loss_fn(w)
            
            loss_history.append(loss)
            grad_norm_history.append(np.linalg.norm(grad))
            
            # TODO: RMSProp更新
            # avg_sq_grad = self.rho * avg_sq_grad + (1 - self.rho) * grad**2
            # w -= self.lr * grad / (np.sqrt(avg_sq_grad) + self.epsilon)
            
            if np.linalg.norm(grad) < 1e-6:
                break
        
        return OptimizationResult(w, loss_history, grad_norm_history, i + 1)


class Adam:
    """
    Adam: 自适应矩估计
    
    m_t = beta1 * m_{t-1} + (1-beta1) * g_t
    v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    w_{t+1} = w_t - lr * m_hat / (sqrt(v_hat) + epsilon)
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 max_iter: int = 1000):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_iter = max_iter
    
    def optimize(self, loss_fn: Callable, grad_fn: Callable,
                 w0: np.ndarray) -> OptimizationResult:
        w = w0.copy()
        m = np.zeros_like(w)
        v = np.zeros_like(w)
        loss_history = []
        grad_norm_history = []
        
        for t in range(1, self.max_iter + 1):
            grad = grad_fn(w)
            loss = loss_fn(w)
            
            loss_history.append(loss)
            grad_norm_history.append(np.linalg.norm(grad))
            
            # TODO: Adam更新
            # m = self.beta1 * m + (1 - self.beta1) * grad
            # v = self.beta2 * v + (1 - self.beta2) * grad**2
            
            # 偏差修正
            # m_hat = m / (1 - self.beta1**t)
            # v_hat = v / (1 - self.beta2**t)
            
            # 更新参数
            # w -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            if np.linalg.norm(grad) < 1e-6:
                break
        
        return OptimizationResult(w, loss_history, grad_norm_history, t)
