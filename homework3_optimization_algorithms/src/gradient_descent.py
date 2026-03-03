"""
梯度下降法及其变体实现
"""

import numpy as np
from typing import Callable, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """优化结果"""
    weights: np.ndarray
    loss_history: List[float]
    grad_norm_history: List[float]
    n_iterations: int


class GradientDescent:
    """标准梯度下降"""
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, 
                 tol: float = 1e-6):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
    
    def optimize(self, loss_fn: Callable, grad_fn: Callable, 
                 w0: np.ndarray) -> OptimizationResult:
        """
        优化
        
        Args:
            loss_fn: 损失函数
            grad_fn: 梯度函数
            w0: 初始权重
        """
        w = w0.copy()
        loss_history = [loss_fn(w)]
        grad_norm_history = [np.linalg.norm(grad_fn(w))]
        
        for i in range(self.max_iter):
            # TODO: 梯度下降更新
            # w = w - lr * grad
            grad = grad_fn(w)
            w = w - self.lr * grad
            
            loss = loss_fn(w)
            loss_history.append(loss)
            grad_norm_history.append(np.linalg.norm(grad))
            
            if grad_norm_history[-1] < self.tol:
                break
        
        return OptimizationResult(w, loss_history, grad_norm_history, i + 1)


class StochasticGradientDescent:
    """随机梯度下降"""
    
    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 100,
                 tol: float = 1e-6):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.tol = tol
    
    def optimize(self, X: np.ndarray, y: np.ndarray, 
                 loss_fn: Callable, grad_fn: Callable,
                 w0: np.ndarray) -> OptimizationResult:
        """
        SGD优化
        
        Args:
            X: 特征 (n_samples, n_features)
            y: 标签 (n_samples,)
        """
        w = w0.copy()
        n_samples = X.shape[0]
        
        loss_history = [loss_fn(w, X, y)]
        
        for epoch in range(self.max_epochs):
            # 打乱数据顺序
            indices = np.random.permutation(n_samples)
            X_shuf = X[indices]
            y_shuf = y[indices]
            
            for i in range(n_samples):
                # TODO: 计算单个样本的梯度并更新
                # xi = X_shuf[i:i+1], yi = y_shuf[i:i+1]
                # grad = grad_fn(w, xi, yi)
                # w = w - lr * grad
                pass
            
            loss = loss_fn(w, X, y)
            loss_history.append(loss)
            
            if len(loss_history) > 2 and abs(loss_history[-1] - loss_history[-2]) < self.tol:
                break
        
        grad_norm = np.linalg.norm(grad_fn(w, X, y))
        
        return OptimizationResult(w, loss_history, [grad_norm], epoch + 1)


class MiniBatchGradientDescent:
    """小批量梯度下降"""
    
    def __init__(self, learning_rate: float = 0.01, batch_size: int = 32,
                 max_epochs: int = 100, tol: float = 1e-6):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tol = tol
    
    def optimize(self, X: np.ndarray, y: np.ndarray,
                 loss_fn: Callable, grad_fn: Callable,
                 w0: np.ndarray) -> OptimizationResult:
        """Mini-batch GD"""
        w = w0.copy()
        n_samples = X.shape[0]
        loss_history = []
        
        for epoch in range(self.max_epochs):
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]
                
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # TODO: 小批量梯度更新
                grad = grad_fn(w, X_batch, y_batch)
                w = w - self.lr * grad
            
            loss = loss_fn(w, X, y)
            loss_history.append(loss)
            
            if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < self.tol:
                break
        
        return OptimizationResult(w, loss_history, [], epoch + 1)


class MomentumSGD:
    """带动量的SGD"""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9,
                 max_epochs: int = 100, tol: float = 1e-6):
        self.lr = learning_rate
        self.momentum = momentum
        self.max_epochs = max_epochs
        self.tol = tol
    
    def optimize(self, X: np.ndarray, y: np.ndarray,
                 loss_fn: Callable, grad_fn: Callable,
                 w0: np.ndarray) -> OptimizationResult:
        """带动量的SGD"""
        w = w0.copy()
        v = np.zeros_like(w)  # 速度
        
        n_samples = X.shape[0]
        loss_history = []
        
        for epoch in range(self.max_epochs):
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, self.batch_size):
                # TODO: 动量更新
                # v = momentum * v - lr * grad
                # w = w + v
                pass
            
            loss = loss_fn(w, X, y)
            loss_history.append(loss)
        
        return OptimizationResult(w, loss_history, [], epoch + 1)


class LogisticRegression:
    """逻辑回归模型 - 用于测试优化算法"""
    
    def __init__(self, reg_lambda: float = 0.01):
        self.reg_lambda = reg_lambda
        self.weights = None
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid函数"""
        # TODO: 实现
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
    def loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """逻辑回归损失"""
        n = X.shape[0]
        z = X @ w
        h = self.sigmoid(z)
        
        # 交叉熵损失 + L2正则化
        eps = 1e-15
        loss = -1/n * np.sum(y * np.log(h + eps) + (1-y) * np.log(1-h + eps))
        loss += self.reg_lambda * np.sum(w ** 2)
        
        return loss
    
    def gradient(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """损失函数梯度"""
        n = X.shape[0]
        z = X @ w
        h = self.sigmoid(z)
        
        # 梯度 + L2正则化项
        grad = 1/n * X.T @ (h - y) + 2 * self.reg_lambda * w
        
        return grad
    
    def fit(self, X: np.ndarray, y: np.ndarray, optimizer: str = 'gd',
            **optimizer_kwargs) -> 'LogisticRegression':
        """训练模型"""
        n_features = X.shape[1]
        w0 = np.zeros(n_features)
        
        if optimizer == 'gd':
            opt = GradientDescent(**optimizer_kwargs)
            result = opt.optimize(
                lambda w: self.loss(w, X, y),
                lambda w: self.gradient(w, X, y),
                w0
            )
        elif optimizer == 'sgd':
            opt = StochasticGradientDescent(**optimizer_kwargs)
            result = opt.optimize(X, y, self.loss, self.gradient, w0)
        elif optimizer == 'mini_batch':
            opt = MiniBatchGradientDescent(**optimizer_kwargs)
            result = opt.optimize(X, y, self.loss, self.gradient, w0)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        self.weights = result.weights
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        return self.sigmoid(X @ self.weights)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """预测类别"""
        return (self.predict_proba(X) >= threshold).astype(int)
