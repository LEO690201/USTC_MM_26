"""
网络扩散模型实现
"""

import numpy as np
from typing import List, Set, Dict, Tuple
from collections import defaultdict
import random


class IndependentCascadeModel:
    """
     (Independent Cascade Model独立级联模型)
    
    特点: 激活尝试只进行一次
    """
    
    def __init__(self, adj_dict: Dict[int, List[int]], 
                 activation_prob: float = 0.1, random_seed: int = 42):
        """
        Args:
            adj_dict: 邻接表 {node: [neighbors]}
            activation_prob: 边上的激活概率 (简化版，所有边相同)
        """
        self.adj = adj_dict
        self.p = activation_prob
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def simulate(self, seed_set: Set[int], 
                 max_steps: int = 100) -> Tuple[Set[int], List[Set[int]]]:
        """
        模拟扩散过程
        
        Args:
            seed_set: 初始激活的种子节点
            max_steps: 最大扩散步数
        
        Returns:
            (最终激活节点集合, 每步激活节点列表)
        """
        active = seed_set.copy()
        newly_active = seed_set.copy()
        activated_history = [seed_set.copy()]
        
        for step in range(max_steps):
            if not newly_active:
                break
            
            current_newly_active = set()
            
            # 遍历新激活的节点
            for node in newly_active:
                # 尝试激活其邻居
                for neighbor in self.adj.get(node, []):
                    if neighbor not in active:
                        # 按概率激活
                        if random.random() < self.p:
                            current_newly_active.add(neighbor)
            
            # 更新状态
            active.update(current_newly_active)
            newly_active = current_newly_active
            activated_history.append(current_newly_active.copy())
        
        return active, activated_history
    
    def simulate_monte_carlo(self, seed_set: Set[int_simulations: int], 
                             n = 100) -> Dict[str, float]:
        """
        蒙特卡洛模拟获取统计结果
        
        Returns:
            平均激活节点数, 标准差等
        """
        final_sizes = []
        
        for _ in range(n_simulations):
            final_active, _ = self.simulate(seed_set)
            final_sizes.append(len(final_active))
        
        return {
            'mean': np.mean(final_sizes),
            'std': np.std(final_sizes),
            'min': np.min(final_sizes),
            'max': np.max(final_sizes)
        }


class LinearThresholdModel:
    """
    线性阈值模型 (Linear Threshold Model)
    
    特点: 节点被激活需要邻居的影响之和超过阈值
    """
    
    def __init__(self, adj_dict: Dict[int, List[int]],
                 edge_weights: Dict[Tuple[int, int], float] = None,
                 random_seed: int = 42):
        """
        Args:
            adj_dict: 邻接表
            edge_weights: 边权重 (可选)
        """
        self.adj = adj_dict
        self.edge_weights = edge_weights or {}
        random.seed(random_seed)
        
        # 为每条边分配随机权重
        if not self.edge_weights:
            self._assign_random_weights()
    
    def _assign_random_weights(self):
        """为每条边分配随机权重"""
        for node, neighbors in self.adj.items():
            if neighbors:
                weights = np.random.random(len(neighbors))
                weights = weights / weights.sum()
                for neighbor, w in zip(neighbors, weights):
                    self.edge_weights[(node, neighbor)] = w
    
    def _get_threshold(self, node: int) -> float:
        """获取节点阈值"""
        return np.random.random()
    
    def simulate(self, seed_set: Set[int], 
                 max_steps: int = 100) -> Tuple[Set[int], List[Set[int]]]:
        """
        模拟线性阈值模型的扩散
        """
        active = seed_set.copy()
        newly_active = seed_set.copy()
        activated_history = [seed_set.copy()]
        
        # 计算每个节点的阈值
        thresholds = {node: self._get_threshold(node) for node in self.adj.keys()}
        
        # 记录每个节点的累计影响
        influence = defaultdict(float)
        
        for step in range(max_steps):
            if not newly_active:
                break
            
            current_newly_active = set()
            
            for node in newly_active:
                for neighbor in self.adj.get(node, []):
                    if neighbor not in active:
                        # 添加边的影响
                        edge_weight = self.edge_weights.get((node, neighbor), 0)
                        influence[neighbor] += edge_weight
                        
                        # 检查是否超过阈值
                        if influence[neighbor] >= thresholds[neighbor]:
                            current_newly_active.add(neighbor)
            
            active.update(current_newly_active)
            newly_active = current_newly_active
            activated_history.append(current_newly_active.copy())
        
        return active, activated_history


class InfluenceMaximize:
    """
    影响最大化问题求解
    
    使用贪心算法近似求解
    """
    
    def __init__(self, diffusion_model, n_simulations: int = 100):
        """
        Args:
            diffusion_model: 扩散模型实例
            n_simulations: 每次评估的模拟次数
        """
        self.model = diffusion_model
        self.n_simulations = n_simulations
    
    def greedy_select(self, k: int, candidates: List[int]) -> List[int]:
        """
        贪心选择k个种子节点
        
        策略: 每轮选择使边际收益最大的节点
        """
        seed_set = []
        current_candidates = candidates.copy()
        
        for _ in range(k):
            best_node = None
            best_spread = -1
            
            for node in current_candidates:
                # 计算加入该节点后的边际收益
                test_seeds = seed_set + [node]
                result = self.model.simulate_monte_carlo(
                    set(test_seeds), self.n_simulations
                )
                spread = result['mean']
                
                if spread > best_spread:
                    best_spread = spread
                    best_node = node
            
            seed_set.append(best_node)
            current_candidates.remove(best_node)
            print(f"选择节点 {best_node}, 当前影响范围: {best_spread:.2f}")
        
        return seed_set
    
    def celf_select(self, k: int, candidates: List[int]) -> List[int]:
        """
        CELF算法 (利用子模性加速贪心)
        
        利用影响力函数的子模性进行剪枝
        """
        # TODO: 实现CELF算法
        # 核心思想: 如果一个节点在之前的评估中边际收益不是最大,
        # 那么在后续迭代中它也不太可能成为最大
        pass
