"""
卡尔曼滤波实现
"""

import numpy as np
from typing import Tuple, List, Optional

class KalmanFilter:
    """
    卡尔曼滤波器
    
    状态方程: x_t = F * x_{t-1} + w_t, w_t ~ N(0, Q)
    观测方程: z_t = H * x_t + v_t, v_t ~ N(0, R)
    """
    
    def __init__(self, F: np.ndarray, H: np.ndarray, 
                 Q: np.ndarray, R: np.ndarray,
                 x0: np.ndarray, P0: np.ndarray):
        """
        Args:
            F: 状态转移矩阵 (n, n)
            H: 观测矩阵 (m, n)
            Q: 过程噪声协方差 (n, n)
            R: 观测噪声协方差 (m, m)
            x0: 初始状态均值 (n,)
            P0: 初始状态协方差 (n, n)
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        
        self.n = F.shape[0]
        self.m = H.shape[0]
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测步骤 (Time Update)
        
        x_{t|t-1} = F * x_{t-1|t-1}
        P_{t|t-1} = F * P_{t-1|t-1} * F^T + Q
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P
    
    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新步骤 (Measurement Update)
        
        K = P * H^T * (H * P * H^T + R)^-1
        x = x + K * (z - H * x)
        P = (I - K * H) * P
        """
        # 创新 (Innovation)
        y = z - self.H @ self.x
        
        # 创新协方差
        S = self.H @ self.P @ self.H.T + self.R
        
        # 卡尔曼增益
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((self.n, self.m))
        
        # 更新状态
        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ self.H) @ self.P
        
        return self.x, self.P
    
    def batch_filter(self, measurements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量滤波
        
        Args:
            measurements: (T, m) 观测序列
            
        Returns:
            (estimates, covariances)
        """
        T = measurements.shape[0]
        estimates = np.zeros((T, self.n))
        covariances = np.zeros((T, self.n, self.n))
        
        for t in range(T):
            self.predict()
            self.update(measurements[t])
            estimates[t] = self.x
            covariances[t] = self.P
            
        return estimates, covariances
    
    def rts_smoother(self, estimates: np.ndarray, covariances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        RTS平滑 (Rauch-Tung-Striebel Smoother)
        
        反向平滑过程，利用所有数据优化估计
        """
        T = estimates.shape[0]
        smoothed_x = estimates.copy()
        smoothed_P = covariances.copy()
        
        for t in range(T - 2, -1, -1):
            # 预测下一步
            x_pred = self.F @ estimates[t]
            P_pred = self.F @ covariances[t] @ self.F.T + self.Q
            
            # 平滑增益 C = P_t * F^T * P_{t+1|t}^-1
            try:
                C = covariances[t] @ self.F.T @ np.linalg.inv(P_pred)
            except np.linalg.LinAlgError:
                C = np.zeros_like(covariances[t])
            
            # 更新平滑估计
            smoothed_x[t] = estimates[t] + C @ (smoothed_x[t+1] - x_pred)
            smoothed_P[t] = covariances[t] + C @ (smoothed_P[t+1] - P_pred) @ C.T
            
        return smoothed_x, smoothed_P
