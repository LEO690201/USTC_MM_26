# 作业3: 机器学习中的优化算法 (Optimization Algorithms)

## 1. 实验背景与目的

### 1.1 背景介绍
优化算法是机器学习模型训练的核心驱动力。无论是简单的线性回归，还是复杂的深度神经网络，其本质都是在寻找一组参数 $\boldsymbol{\theta}$，使得损失函数 $J(\boldsymbol{\theta})$ 最小化。

梯度下降法（Gradient Descent）及其变体是目前最主流的优化方法。理解这些算法的数学原理、收敛性质以及超参数的影响，对于调优模型至关重要。

### 1.2 实验目的
本实验旨在让学生：
1.  **掌握** 一阶（梯度下降）和二阶（牛顿法）优化算法的数学推导。
2.  **实现** 多种优化器（SGD, Momentum, Adam, RMSProp, BFGS）。
3.  **分析** 学习率、动量系数等超参数对收敛速度和稳定性的影响。
4.  **理解** 凸优化与非凸优化的本质区别（如局部最优、鞍点）。

---

## 2. 数学模型详解

### 2.1 目标函数：逻辑回归 (Logistic Regression)

我们使用逻辑回归作为测试基准，因为它是一个典型的凸优化问题（Convex Optimization）。

#### 损失函数 (Cross-Entropy Loss)
给定数据集 $\{( \boldsymbol{x}^{(i)}, y^{(i)} )\}_{i=1}^N$，其中 $\boldsymbol{x} \in \mathbb{R}^d, y \in \{0, 1\}$。
逻辑回归模型预测概率为 $\hat{y} = \sigma(\boldsymbol{w}^T \boldsymbol{x} + b)$，其中 $\sigma(z) = \frac{1}{1+e^{-z}}$。

带 $L_2$ 正则化的损失函数为：
$$
J(\boldsymbol{w}) = - \frac{1}{N} \sum_{i=1}^N \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)}) \right] + \frac{\lambda}{2} \|\boldsymbol{w}\|^2
$$

#### 梯度计算
$$
\nabla_{\boldsymbol{w}} J(\boldsymbol{w}) = \frac{1}{N} \sum_{i=1}^N (\hat{y}^{(i)} - y^{(i)}) \boldsymbol{x}^{(i)} + \lambda \boldsymbol{w}
$$
注意：这里假设偏置 $b$ 已并入 $\boldsymbol{w}$（即 $\boldsymbol{x}$ 增加了一维常数 1）。

---

### 2.2 一阶优化算法 (First-Order Methods)

利用梯度信息 $\boldsymbol{g}_t = \nabla J(\boldsymbol{w}_t)$ 进行参数更新。

#### 1. 随机梯度下降 (SGD)
$$
\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta \boldsymbol{g}_t
$$
*   **$\eta$ (Learning Rate)**: 学习率，控制步长。
*   **Batch GD**: 使用所有样本计算梯度（准确但慢）。
*   **Mini-batch SGD**: 每次随机抽取 $m$ 个样本计算梯度（折中方案，最常用）。

#### 2. 动量法 (Momentum)
引入“速度”变量 $\boldsymbol{v}_t$，模拟物理中的惯性，抑制震荡，加速通过平坦区域。
$$
\begin{aligned}
\boldsymbol{v}_{t+1} &= \gamma \boldsymbol{v}_t + \eta \boldsymbol{g}_t \\
\boldsymbol{w}_{t+1} &= \boldsymbol{w}_t - \boldsymbol{v}_{t+1}
\end{aligned}
$$
*   **$\gamma$ (Momentum Coefficient)**: 动量系数，通常取 0.9。

#### 3. 自适应学习率：RMSProp
解决 SGD 学习率难以调整的问题。根据历史梯度的平方和调整每个参数的学习率。
$$
\begin{aligned}
\boldsymbol{s}_t &= \beta \boldsymbol{s}_{t-1} + (1-\beta) (\boldsymbol{g}_t \odot \boldsymbol{g}_t) \\
\boldsymbol{w}_{t+1} &= \boldsymbol{w}_t - \frac{\eta}{\sqrt{\boldsymbol{s}_t + \epsilon}} \odot \boldsymbol{g}_t
\end{aligned}
$$
*   **$\odot$**: 逐元素乘法。
*   **$\beta$**: 衰减率，通常取 0.9。
*   **$\epsilon$**: 防止分母为 0 的小常数，通常取 $10^{-8}$。

#### 4. 自适应矩估计：Adam (Adaptive Moment Estimation)
结合了 Momentum（一阶矩）和 RMSProp（二阶矩）。
$$
\begin{aligned}
\boldsymbol{m}_t &= \beta_1 \boldsymbol{m}_{t-1} + (1-\beta_1) \boldsymbol{g}_t \quad &(\text{一阶矩估计}) \\
\boldsymbol{v}_t &= \beta_2 \boldsymbol{v}_{t-1} + (1-\beta_2) (\boldsymbol{g}_t \odot \boldsymbol{g}_t) \quad &(\text{二阶矩估计}) \\
\hat{\boldsymbol{m}}_t &= \frac{\boldsymbol{m}_t}{1-\beta_1^t}, \quad \hat{\boldsymbol{v}}_t = \frac{\boldsymbol{v}_t}{1-\beta_2^t} \quad &(\text{偏差修正}) \\
\boldsymbol{w}_{t+1} &= \boldsymbol{w}_t - \frac{\eta}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon} \odot \hat{\boldsymbol{m}}_t
\end{aligned}
$$
*   **$\beta_1, \beta_2$**: 通常取 0.9, 0.999。

---

### 2.3 二阶优化算法 (Second-Order Methods)

利用 Hessian 矩阵 $\boldsymbol{H} = \nabla^2 J(\boldsymbol{w})$（二阶导数）信息，不仅知道梯度方向，还知道曲率（Curvature）。

#### 1. 牛顿法 (Newton's Method)
利用二次泰勒展开近似目标函数：
$$
J(\boldsymbol{w} + \Delta \boldsymbol{w}) \approx J(\boldsymbol{w}) + \nabla J(\boldsymbol{w})^T \Delta \boldsymbol{w} + \frac{1}{2} \Delta \boldsymbol{w}^T \boldsymbol{H} \Delta \boldsymbol{w}
$$
求极值得到更新公式：
$$
\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \boldsymbol{H}^{-1} \boldsymbol{g}_t
$$
*   **优点**: 收敛速度极快（二次收敛）。
*   **缺点**: 计算 $\boldsymbol{H}^{-1}$ 的复杂度为 $O(d^3)$，在高维空间不可行。

#### 2. 拟牛顿法 (BFGS)
通过迭代更新 $\boldsymbol{H}^{-1}$ 的近似矩阵 $\boldsymbol{B}_t$，避免直接求逆。
$$
\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta \boldsymbol{B}_t \boldsymbol{g}_t
$$
$\boldsymbol{B}_{t+1}$ 的更新公式较为复杂（见 `scipy.optimize.minimize(method='BFGS')`）。

---

## 3. 实验任务详解

### 任务 1: 数据生成与逻辑回归 (难度: ⭐⭐)
1.  **数据生成**:
    *   使用 `sklearn.datasets.make_classification` 生成二分类数据。
    *   样本数 $N=1000$，特征数 $d=20$，噪声较大。
    *   划分训练集 (80%) 和测试集 (20%)。
2.  **模型实现**:
    *   实现 `LogisticRegression` 类。
    *   包含 `sigmoid`, `loss`, `gradient` 方法。
    *   **验证**: 使用数值微分（Numerical Differentiation）检查解析梯度计算是否正确。
        $$ \frac{\partial J}{\partial w_i} \approx \frac{J(\boldsymbol{w} + \epsilon \boldsymbol{e}_i) - J(\boldsymbol{w} - \epsilon \boldsymbol{e}_i)}{2\epsilon} $$

### 任务 2: 优化算法对比 (难度: ⭐⭐⭐)
1.  **实现优化器**:
    *   **SGD**: 实现 mini-batch 更新。
    *   **Momentum**: 增加动量项。
    *   **Adam**: 完整实现 Adam 更新规则。
2.  **实验设置**:
    *   固定 Batch Size = 32。
    *   初始学习率 $\eta$ 需分别调优（例如 SGD 可能需要 0.1，Adam 需要 0.001）。
    *   记录每个 Epoch 的训练集 Loss 和测试集 Accuracy。
3.  **绘图**:
    *   绘制 Loss vs Epoch 曲线。
    *   绘制 Accuracy vs Time 曲线（对比收敛时间）。

### 任务 3: 牛顿法与 Rosenbrock 函数 (难度: ⭐⭐⭐⭐)
1.  **Rosenbrock 函数**:
    $$ f(x, y) = (a-x)^2 + b(y-x^2)^2 $$
    *   这是一个著名的非凸函数，全局最小值在 $(a, a^2)$ 处，但在狭长的山谷中很难收敛。
    *   设 $a=1, b=100$。
2.  **求解**:
    *   推导 $f(x,y)$ 的梯度向量 $\boldsymbol{g}$ 和 Hessian 矩阵 $\boldsymbol{H}$。
    *   分别使用 **梯度下降** 和 **牛顿法** 求解最小值。
    *   初始点设为 $(-1, -1)$ 或 $(0, 0)$。
3.  **可视化**:
    *   绘制 $f(x,y)$ 的等高线图。
    *   在图上画出两种算法的迭代轨迹。
    *   **观察**: 牛顿法是否直接跳到了最优点？梯度下降是否在震荡？

### 任务 4: 非凸优化与鞍点 (难度: ⭐⭐⭐⭐⭐)
1.  **构造函数**: $f(x, y) = x^2 - y^2$（马鞍面）。
    *   $(0,0)$ 是鞍点（Saddle Point），不是极小值。
    *   Hessian 矩阵 $\begin{bmatrix} 2 & 0 \\ 0 & -2 \end{bmatrix}$ 有正有负特征值。
2.  **实验**:
    *   从鞍点附近（如 $(0.01, 0)$）启动优化。
    *   对比 SGD（带噪声）和标准 GD 在逃离鞍点时的表现。
    *   **讨论**: 为什么深度学习中鞍点比局部最小值更可怕？

---

## 4. 提交物要求
1.  **代码**: 包含所有优化器的实现类。
2.  **报告**:
    *   逻辑回归梯度推导过程。
    *   不同优化器的收敛曲线对比图。
    *   Rosenbrock 函数的等高线轨迹图。
    *   对非凸优化实验结果的分析。

## 5. 参考资料
1.  Ruder, S. (2016). An overview of gradient descent optimization algorithms.
2.  Boyd, S., & Vandenberghe, L. (2004). Convex Optimization.
3.  Goodfellow, I., et al. (2016). Deep Learning (Chapter 8).
