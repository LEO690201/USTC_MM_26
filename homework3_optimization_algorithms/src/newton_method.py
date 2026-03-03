"""
牛顿法与拟牛顿法实现
"""

import numpy as np
from typing import Callable, Tuple, List
from gradient_descent import OptimizationResult

class NewtonMethod:
    """
    牛顿法 (Newton's Method)
    
    x_{k+1} = x_k - H^{-1} * g_k
    """
    
    def __init__(self, learning_rate: float = 1.0, max_iter: int = 100, tol: float = 1e-6):
        """
        Args:
            learning_rate: 步长 (牛顿法通常为1.0)
        """
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
    
    def optimize(self, func: Callable, grad_fn: Callable, hess_fn: Callable,
                 x0: np.ndarray) -> OptimizationResult:
        x = x0.copy()
        loss_history = []
        grad_norm_history = []
        
        for i in range(self.max_iter):
            loss = func(x)
            grad = grad_fn(x)
            hess = hess_fn(x)
            
            loss_history.append(loss)
            grad_norm_history.append(np.linalg.norm(grad))
            
            if np.linalg.norm(grad) < self.tol:
                break
            
            # TODO: 牛顿法更新
            # dx = -inv(H) * g
            # x = x + lr * dx
            # 建议使用 np.linalg.solve(hess, -grad) 而不是求逆
            
            try:
                dx = np.linalg.solve(hess, -grad)
                x = x + self.lr * dx
            except np.linalg.LinAlgError:
                print("Hessian矩阵奇异，无法求解")
                break
                
        return OptimizationResult(x, loss_history, grad_norm_history, i + 1)


class BFGS:
    """
    BFGS算法 (拟牛顿法)
    
    迭代更新Hessian的逆近似矩阵 Bk
    """
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol
    
    def optimize(self, func: Callable, grad_fn: Callable,
                 x0: np.ndarray) -> OptimizationResult:
        x = x0.copy()
        n = len(x)
        I = np.eye(n)
        Bk = I  # 初始Hessian逆近似
        
        loss_history = []
        grad_norm_history = []
        
        grad = grad_fn(x)
        
        for i in range(self.max_iter):
            loss = func(x)
            loss_history.append(loss)
            grad_norm_history.append(np.linalg.norm(grad))
            
            if np.linalg.norm(grad) < self.tol:
                break
            
            # 1. 计算搜索方向 p = -Bk * g
            pk = -Bk @ grad
            
            # 2. 线搜索 (这里简化为固定步长，实际应使用Wolfe条件)
            alpha = 1.0
            x_new = x + alpha * pk
            grad_new = grad_fn(x_new)
            
            # 3. 计算 s_k 和 y_k
            sk = x_new - x
            yk = grad_new - grad
            
            # 4. 更新 Bk
            # rho = 1 / (y_k^T * s_k)
            # Bk = (I - rho * s_k * y_k^T) * Bk * (I - rho * y_k * s_k^T) + rho * s_k * s_k^T
            
            if np.dot(yk, sk) > 1e-10:  # 避免除零
                rho = 1.0 / np.dot(yk, sk)
                term1 = I - rho * np.outer(sk, yk)
                term2 = I - rho * np.outer(yk, sk)
                Bk = term1 @ Bk @ term2 + rho * np.outer(sk, sk)
            
            x = x_new
            grad = grad_new
            
        return OptimizationResult(x, loss_history, grad_norm_history, i + 1)
