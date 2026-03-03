# 作业4: 图论与网络分析 (Graph Theory & Network Analysis)

## 1. 实验背景与目的

### 1.1 背景介绍
图（Graph）是描述复杂系统的通用语言。从微观的蛋白质相互作用网络，到宏观的全球航空运输网络，再到虚拟的互联网链接结构，万物皆可为网。

本实验聚焦于网络科学中的三个核心问题：
1.  **节点重要性 (Centrality)**: 谁是网络中的关键人物？（例如 PageRank 用于网页排名）
2.  **社团检测 (Community Detection)**: 网络中是否存在紧密连接的群体？（例如社交圈子发现）
3.  **信息传播 (Information Diffusion)**: 谣言或病毒如何在网络中扩散？（例如舆情监控、疫情防控）

### 1.2 实验目的
本实验旨在让学生：
1.  **掌握** 图的基本表示方法（邻接矩阵、邻接表）。
2.  **理解** PageRank 算法的随机游走模型及其收敛性。
3.  **实现** 经典的社团检测算法（Louvain, Label Propagation）。
4.  **模拟** 网络上的动态过程（独立级联模型 IC, 线性阈值模型 LT）。

---

## 2. 数学模型详解

### 2.1 图的基本定义
一个图 $G=(V, E)$ 由节点集合 $V$ 和边集合 $E$ 组成。
*   $N = |V|$: 节点数。
*   $M = |E|$: 边数。
*   $\boldsymbol{A}$: $N \times N$ 的邻接矩阵（Adjacency Matrix）。
    *   若 $(i, j) \in E$，则 $A_{ij} = 1$（无权图）或 $w_{ij}$（加权图）；否则 $A_{ij} = 0$。
*   $k_i = \sum_j A_{ij}$: 节点 $i$ 的度（Degree）。对于有向图分入度 $k_i^{in}$ 和出度 $k_i^{out}$。

---

### 2.2 PageRank 算法

PageRank 是 Google 搜索引擎的核心算法，用于评估网页的重要性。

#### 随机游走模型 (Random Surfer Model)
想象一个随机上网者，他在当前页面 $i$ 有两种行为选择：
1.  **点击链接**: 以概率 $d$（阻尼因子，Damping Factor），从当前页面的外链中随机选择一个跳转。
2.  **随机跳转**: 以概率 $1-d$，在浏览器地址栏随机输入一个网址跳转（跳转到任意页面）。

#### 数学公式
节点 $i$ 的 PageRank 值 $PR(i)$ 定义为：

$$
PR(i) = \frac{1-d}{N} + d \sum_{j \in M(i)} \frac{PR(j)}{L(j)}
$$

*   $d$: 阻尼因子，通常取 $0.85$。
*   $N$: 总页面数。
*   $M(i)$: 所有指向节点 $i$ 的节点集合（入链集合）。
*   $L(j)$: 节点 $j$ 的出度（出链数）。

#### 矩阵形式与幂法 (Power Method)
定义转移矩阵 $\boldsymbol{M}$，其中 $M_{ij} = \begin{cases} 1/L(j), & \text{若 } j \to i \\ 0, & \text{否则} \end{cases}$。
PageRank 向量 $\boldsymbol{v}$ 是以下方程的解（特征值为 1 的特征向量）：

$$
\boldsymbol{v} = d \boldsymbol{M} \boldsymbol{v} + \frac{1-d}{N} \boldsymbol{1}
$$

迭代公式：
$$
\boldsymbol{v}_{t+1} = d \boldsymbol{M} \boldsymbol{v}_t + \frac{1-d}{N} \boldsymbol{1}
$$

---

### 2.3 社团检测 (Community Detection)

社团（Community）是指网络中内部连接紧密、外部连接稀疏的节点子集。

#### 模块度 (Modularity) $Q$
模块度是衡量社团划分质量的标准。其物理含义是：**社团内部边的实际数量与随机网络中期望数量的差值**。

$$
Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)
$$

*   $m = \frac{1}{2} \sum_{ij} A_{ij}$: 总边数（权重和）。
*   $A_{ij}$: 节点 $i$ 和 $j$ 之间的实际边权重。
*   $\frac{k_i k_j}{2m}$: 在保持节点度数不变的随机网络（Configuration Model）中，节点 $i$ 和 $j$ 之间期望的边权重。
*   $\delta(c_i, c_j)$: 克罗内克函数，当 $i, j$ 属于同一社团时为 1，否则为 0。

目标是寻找一种划分 $\mathcal{C} = \{c_1, \dots, c_N\}$，使得 $Q$ 最大化。

#### Louvain 算法
一种贪心优化 $Q$ 的层次聚类算法：
1.  **初始化**: 每个节点自成一个社团。
2.  **局部移动**: 对每个节点 $i$，尝试将其移入邻居 $j$ 的社团，计算 $\Delta Q$。若 $\Delta Q > 0$，则移动。重复直到收敛。
3.  **网络聚合**: 将每个社团合并为一个超节点（Super-node），社团间的边权重设为原社团间所有边的权重和。
4.  **迭代**: 对新网络重复步骤 2-3，直到 $Q$ 不再增加。

---

### 2.4 信息扩散模型 (Information Diffusion)

模拟信息在网络上的传播过程。

#### 独立级联模型 (Independent Cascade, IC)
*   **状态**: 节点分为激活（Active）和未激活（Inactive）。
*   **传播规则**:
    *   当节点 $u$ 在 $t$ 时刻被激活，它有且仅有一次机会尝试激活其未激活邻居 $v$。
    *   激活概率为 $p_{uv}$（独立事件）。
    *   若成功，$v$ 在 $t+1$ 时刻变为激活状态；若失败，$u$ 无法再次尝试激活 $v$。

#### 线性阈值模型 (Linear Threshold, LT)
*   **状态**: 同上。
*   **传播规则**:
    *   每个节点 $v$ 有一个激活阈值 $\theta_v \sim U[0, 1]$（随机生成）。
    *   每条边 $(u, v)$ 有权重 $b_{uv}$，且 $\sum_{u \in Neighbor(v)} b_{uv} \le 1$。
    *   当 $\sum_{u \in Active\_Neighbors(v)} b_{uv} \ge \theta_v$ 时，节点 $v$ 被激活。
*   **直观理解**: “三人成虎”，邻居的影响累积超过心理防线时，个体就会被同化。

---

## 3. 实验任务详解

### 任务 1: PageRank 实现与分析 (难度: ⭐⭐)
1.  **数据**: 使用 `networkx.karate_club_graph()` 或生成随机图。
2.  **实现**:
    *   **方法 A**: 使用 `networkx.pagerank`（作为基准）。
    *   **方法 B**: 手写幂法迭代（Power Iteration）。
        *   初始化 $\boldsymbol{v}_0 = [1/N, \dots, 1/N]^T$。
        *   设置 $d=0.85, \epsilon=10^{-6}$。
3.  **分析**:
    *   对比方法 A 和 B 的结果是否一致。
    *   调节阻尼因子 $d \in \{0.5, 0.85, 0.99\}$，观察排名变化。
    *   **思考**: 当 $d \to 1$ 时会发生什么？（提示：考虑非连通图或陷阱节点 Spider Trap）。

### 任务 2: 社团检测 (难度: ⭐⭐⭐)
1.  **算法**: 实现 Louvain 算法（可以使用 `python-louvain` 库，建议尝试手写核心逻辑）。
2.  **可视化**:
    *   对空手道俱乐部网络进行社团划分。
    *   使用 Force-directed Layout（力导向布局）绘图。
    *   不同社团的节点染不同颜色。
3.  **评估**: 计算最终的模块度 $Q$。

### 任务 3: 影响力最大化 (Influence Maximization) (难度: ⭐⭐⭐⭐)
**目标**: 寻找 $k$ 个种子节点，使得最终激活节点总数（影响力）最大。
这是一个 NP-hard 问题，通常使用贪心算法近似。

1.  **模型**: 独立级联模型 (IC)，设统一传播概率 $p=0.1$。
2.  **贪心策略**:
    *   初始化种子集 $S = \emptyset$。
    *   进行 $k$ 轮循环（例如 $k=5$）：
        *   对每个非种子节点 $v$，估算 $S \cup \{v\}$ 的影响力（需蒙特卡洛模拟，例如重复 100 次取平均）。
        *   选择边际收益最大的节点 $v^*$ 加入 $S$。
3.  **对比**:
    *   贪心算法选出的种子集 vs 度中心性（Degree Centrality）最高的节点集。
    *   对比两组种子集的最终影响力。

---

## 4. 提交物要求
1.  **代码**: 包含 PageRank, Louvain, IC 模型的实现。
2.  **报告**:
    *   PageRank 排名结果列表（Top 10）。
    *   社团划分的可视化图（带颜色）。
    *   贪心算法选出的 $k$ 个种子节点及其影响力评分。
    *   对不同 $d$ 值影响的讨论。

## 5. 参考资料
1.  Page, L., et al. (1999). The PageRank Citation Ranking: Bringing Order to the Web.
2.  Blondel, V. D., et al. (2008). Fast unfolding of communities in large networks.
3.  Kempe, D., Kleinberg, J., & Tardos, É. (2003). Maximizing the spread of influence through a social network.
