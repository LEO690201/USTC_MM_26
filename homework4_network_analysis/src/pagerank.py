"""
PageRank算法实现
"""

import numpy as np
from typing import Dict, Tuple, List
from scipy import sparse


class PageRank:
    """
    PageRank算法
    
    公式: PR = (1-d)/N * 1 + d * P * PR (其中P是转移矩阵)
    """
    
    def __init__(self, damping_factor: float = 0.85, max_iter: int = 100,
                 tol: float = 1e-6):
        """
        Args:
            damping_factor: 阻尼因子 d
            max_iter: 最大迭代次数
            tol: 收敛阈值
        """
        self.d = damping_factor
        self.max_iter = max_iter
        self.tol = tol
    
    def fit_power_iteration(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        幂迭代法计算PageRank
        
        Args:
            adj_matrix: 邻接矩阵 (n, n), A[i,j]=1表示从i指向j
        
        Returns:
            PageRank分数向量
        """
        n = adj_matrix.shape[0]
        
        # 处理悬挂节点 (出度为0的节点)
        # 如果节点i没有出边，通常假设它链接到所有其他节点，或者其PR值均匀分配
        out_degree = adj_matrix.sum(axis=1)
        dangling_nodes = np.where(out_degree == 0)[0]
        
        # 转移概率矩阵 (列归一化)
        # P[j, i] = 1/out_degree[i] if i->j
        # 注意: 这里adj_matrix[i,j]是从i到j，所以需要转置
        # A.T 的第i列是所有指向i的节点
        
        # 防止除零
        out_degree[out_degree == 0] = 1
        P = adj_matrix.T / out_degree
        
        # 初始PageRank值 (均匀分布)
        pr = np.ones(n) / n
        
        for i in range(self.max_iter):
            pr_old = pr.copy()
            
            # PageRank迭代公式
            # term1: 随机跳转 (1-d)/N
            # term2: 链接跳转 d * P * PR
            # term3: 悬挂节点贡献 (d * sum(PR[dangling]) / N)
            
            dangling_sum = np.sum(pr[dangling_nodes])
            
            # 核心公式
            # PR_new = (1-d)/N + d * P @ PR + d/N * sum(PR[dangling])
            # 注意: P @ PR 计算的是已有链接的传递
            
            pr = (1 - self.d) / n + self.d * (P @ pr) + self.d * dangling_sum / n
            
            # 检查收敛
            if np.linalg.norm(pr - pr_old, 1) < self.tol:
                print(f"收敛于第 {i+1} 次迭代")
                break
        
        return pr
    
    def fit_matrix_method(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        矩阵方法计算PageRank (直接求解线性方程组)
        
        (I - d*M) * PR = (1-d)/N * 1
        其中 M 是修正后的转移矩阵 (包含悬挂节点处理)
        """
        n = adj_matrix.shape[0]
        out_degree = adj_matrix.sum(axis=1)
        
        # 处理悬挂节点
        # 构建Google矩阵的组件
        # M = P + D (D是悬挂节点补偿矩阵)
        
        # 计算 P
        # 防止除零
        norm_factor = np.where(out_degree==0, 1, out_degree)
        P = adj_matrix.T / norm_factor
        
        # 构建方程 (I - d*P) * PR = (1-d)/N + d/N * sum(PR[dangling])
        # 这种形式比较难直接用 linalg.solve 求解，通常用幂迭代
        # 但如果我们将悬挂节点视为链接到所有节点 (1/N)
        # 则 M[:, j] = 1/N if j is dangling else P[:, j]
        
        M = P.copy()
        dangling_nodes = np.where(out_degree == 0)[0]
        for j in dangling_nodes:
            M[:, j] = 1.0 / n
            
        # 方程: PR = d * M @ PR + (1-d)/N * 1
        # => (I - d*M) @ PR = (1-d)/N * 1
        
        A = np.eye(n) - self.d * M
        b = np.ones(n) * (1 - self.d) / n
        
        # PR = A^-1 * b
        try:
            pr = np.linalg.solve(A, b)
            # 归一化
            pr = pr / pr.sum()
            return pr
        except np.linalg.LinAlgError:
            print("矩阵奇异，改用幂迭代")
            return self.fit_power_iteration(adj_matrix)
    
    def get_top_k(self, pr: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        indices = np.argsort(pr)[::-1][:k]
        return [(i, pr[i]) for i in indices]
