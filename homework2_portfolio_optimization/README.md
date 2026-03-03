# 作业2: 投资组合优化 (Portfolio Optimization)

## 1. 实验背景与目的

### 1.1 背景介绍
在金融市场中，投资者面临的核心问题是：如何在**风险**与**收益**之间进行权衡？
*   **收益**：我们希望投资组合的回报最大化。
*   **风险**：我们希望投资组合的波动性（不确定性）最小化。

1952年，Harry Markowitz 提出了**均值-方差模型 (Mean-Variance Model)**，奠定了现代投资组合理论（MPT）的基础。该理论认为，通过分散投资（Diversification），可以在不降低预期收益的情况下降低风险。

随后，Black-Litterman 模型结合了贝叶斯统计思想，解决了 MPT 对输入参数敏感的问题；风险平价（Risk Parity）策略则关注风险贡献的均衡分配。

### 1.2 实验目的
本实验旨在让学生：
1.  **理解** 收益率、波动率、协方差矩阵等金融统计概念。
2.  **掌握** 二次规划（Quadratic Programming, QP）问题的求解方法。
3.  **构建** 有效前沿（Efficient Frontier），理解资产配置的核心思想。
4.  **探索** 高级模型（Black-Litterman, Risk Parity）在实际应用中的优势。

---

## 2. 数学模型详解

### 2.1 基础概念与符号定义

假设有 $N$ 个风险资产（如股票、债券等）。

*   **收益率向量 $\boldsymbol{r}$**: $N \times 1$ 的随机向量，表示各资产的未来收益率。
*   **预期收益向量 $\boldsymbol{\mu}$**: $N \times 1$ 向量，$\boldsymbol{\mu} = \mathbb{E}[\boldsymbol{r}]$。
    *   $\mu_i$: 第 $i$ 个资产的预期收益率。
*   **协方差矩阵 $\boldsymbol{\Sigma}$**: $N \times N$ 对称正定矩阵，$\boldsymbol{\Sigma} = \text{Cov}(\boldsymbol{r})$。
    *   $\Sigma_{ij}$: 第 $i$ 个资产与第 $j$ 个资产收益率的协方差。
    *   $\Sigma_{ii} = \sigma_i^2$: 第 $i$ 个资产收益率的方差。
*   **权重向量 $\boldsymbol{w}$**: $N \times 1$ 向量，表示各资产在投资组合中的比例。
    *   通常约束 $\sum_{i=1}^N w_i = 1$（全额投资）。
    *   若不允许卖空（Short Selling），则约束 $w_i \ge 0$。

#### 投资组合的特征
*   **组合预期收益率**:
    $$ \mu_p = \boldsymbol{w}^T \boldsymbol{\mu} = \sum_{i=1}^N w_i \mu_i $$
*   **组合方差（风险）**:
    $$ \sigma_p^2 = \boldsymbol{w}^T \boldsymbol{\Sigma} \boldsymbol{w} = \sum_{i=1}^N \sum_{j=1}^N w_i w_j \Sigma_{ij} $$

---

### 2.2 Markowitz 均值-方差模型 (MVO)

#### 优化目标
投资者希望在给定风险厌恶系数 $\lambda$ 下，最大化效用函数（收益 - 风险惩罚）：

$$
\max_{\boldsymbol{w}} \quad \boldsymbol{w}^T \boldsymbol{\mu} - \frac{\lambda}{2} \boldsymbol{w}^T \boldsymbol{\Sigma} \boldsymbol{w}
$$

或者等价地，在给定目标收益 $\mu_{target}$ 下，最小化组合方差：

$$
\begin{aligned}
\min_{\boldsymbol{w}} \quad & \frac{1}{2} \boldsymbol{w}^T \boldsymbol{\Sigma} \boldsymbol{w} \\
\text{s.t.} \quad & \boldsymbol{w}^T \boldsymbol{\mu} = \mu_{target} \\
& \boldsymbol{w}^T \boldsymbol{1} = 1 \\
& w_i \ge 0 \quad (\text{可选: 不允许卖空})
\end{aligned}
$$

#### 求解方法
这是一个典型的**凸二次规划 (Convex Quadratic Programming)** 问题。
*   如果只有等式约束（允许卖空），可以使用拉格朗日乘子法求得解析解。
*   如果有不等式约束（$w_i \ge 0$），需使用数值优化算法（如 `scipy.optimize.minimize` 或 `cvxpy`）。

#### 有效前沿 (Efficient Frontier)
有效前沿是指在通过改变 $\mu_{target}$ 或 $\lambda$ 得到的一系列最优组合 $(\sigma_p, \mu_p)$ 在风险-收益平面上构成的曲线。

---

### 2.3 Black-Litterman 模型 (BL)

Markowitz 模型的一个重大缺陷是：**输入敏感性**。$\boldsymbol{\mu}$ 的微小估计误差会导致 $\boldsymbol{w}$ 的剧烈波动（GIGO: Garbage In, Garbage Out）。

BL 模型引入贝叶斯思想：
$$ \text{Posterior} \propto \text{Prior} \times \text{Likelihood} $$

*   **先验 (Prior)**: 市场均衡状态隐含的预期收益 $\boldsymbol{\Pi}$。
    *   假设市场组合 $\boldsymbol{w}_{mkt}$ 是有效的。
    *   反推隐含收益: $\boldsymbol{\Pi} = \lambda \boldsymbol{\Sigma} \boldsymbol{w}_{mkt}$。
*   **观点 (Views)**: 投资者对某些资产的主观预测 $\boldsymbol{P}, \boldsymbol{Q}, \boldsymbol{\Omega}$。
    *   $\boldsymbol{P}$: $K \times N$ 矩阵，表示 $K$ 个观点涉及的资产组合。
    *   $\boldsymbol{Q}$: $K \times 1$ 向量，表示观点的预期收益值。
    *   $\boldsymbol{\Omega}$: $K \times K$ 对角矩阵，表示观点的不确定性（方差）。
    *   例子: "我确信资产 A 比资产 B 的收益高 2%" $\Rightarrow P_{1,A}=1, P_{1,B}=-1, Q_1=0.02$。

#### 后验预期收益公式
$$
\boldsymbol{\mu}_{BL} = [(\tau \boldsymbol{\Sigma})^{-1} + \boldsymbol{P}^T \boldsymbol{\Omega}^{-1} \boldsymbol{P}]^{-1} [(\tau \boldsymbol{\Sigma})^{-1} \boldsymbol{\Pi} + \boldsymbol{P}^T \boldsymbol{\Omega}^{-1} \boldsymbol{Q}]
$$
*   $\tau$: 标量，表示对先验（均衡收益）的不确定性程度。通常取较小值（如 0.05）。

---

### 2.4 风险平价 (Risk Parity)

MVO 模型往往导致风险集中在少数高波动资产上。风险平价策略的目标是：**让每个资产对组合总风险的贡献相等**。

#### 风险贡献 (Risk Contribution, RC)
利用欧拉定理，组合波动率 $\sigma_p$ 可以分解为各资产贡献之和：
$$
\sigma_p = \sum_{i=1}^N \text{RC}_i
$$
其中第 $i$ 个资产的风险贡献定义为：
$$
\text{RC}_i = w_i \cdot \frac{\partial \sigma_p}{\partial w_i} = w_i \cdot \frac{(\boldsymbol{\Sigma} \boldsymbol{w})_i}{\sigma_p}
$$

#### 优化目标
$$
\min_{\boldsymbol{w}} \sum_{i=1}^N \sum_{j=1}^N (\text{RC}_i - \text{RC}_j)^2
$$
约束条件: $\boldsymbol{w}^T \boldsymbol{1} = 1, w_i > 0$。

---

## 3. 实验任务详解

### 任务 1: 数据获取与处理 (难度: ⭐⭐)
1.  **数据来源**: 使用 `yfinance` 或 `tushare` 下载。
    *   标的: 选取 5-10 只不同行业的股票（如 AAPL, MSFT, GOOG, JPM, JNJ 等）。
    *   时间: 过去 3-5 年的日收盘价（Adjusted Close）。
2.  **预处理**:
    *   计算日收益率: $r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$。
    *   计算年化预期收益率向量 $\boldsymbol{\mu}$: 均值 $\times 252$。
    *   计算年化协方差矩阵 $\boldsymbol{\Sigma}$: 协方差 $\times 252$。

### 任务 2: 有效前沿构建 (难度: ⭐⭐⭐)
1.  **求解**:
    *   设定一系列目标收益率 $\mu_{target}$（从最小收益到最大收益之间取 50 个点）。
    *   对每个 $\mu_{target}$，求解 MVO 问题，得到最优权重 $\boldsymbol{w}^*$ 和对应的组合波动率 $\sigma_p^*$。
2.  **绘图**:
    *   以 $\sigma_p$ 为横轴，$\mu_p$ 为纵轴，绘制有效前沿曲线。
    *   在图上标出所有单只资产的位置。
    *   标出**全局最小方差组合 (GMV)** 的位置。
    *   标出**切点组合 (Tangency Portfolio)**: 最大夏普比率点（假设无风险利率 $r_f=0.02$）。

### 任务 3: Black-Litterman 模型应用 (难度: ⭐⭐⭐⭐)
1.  **参数设定**:
    *   $\boldsymbol{w}_{mkt}$: 使用市值加权作为先验权重。
    *   $\lambda$: 设为 2.5。
    *   $\tau$: 设为 0.05。
2.  **观点输入**:
    *   构造 1-2 个主观观点（例如：看好科技股，看空能源股）。
    *   设定观点矩阵 $\boldsymbol{P}$ 和收益向量 $\boldsymbol{Q}$。
    *   设定观点置信度矩阵 $\boldsymbol{\Omega}$（可以使用 He & Litterman 的方法构造）。
3.  **结果对比**:
    *   计算 BL 模型的后验收益 $\boldsymbol{\mu}_{BL}$。
    *   使用 $\boldsymbol{\mu}_{BL}$ 重新进行 MVO 优化。
    *   对比 MVO 和 BL 模型的权重分布 $\boldsymbol{w}_{MVO}$ vs $\boldsymbol{w}_{BL}$。

### 任务 4: 风险平价策略 (难度: ⭐⭐⭐⭐⭐)
1.  **求解**:
    *   使用数值优化算法求解风险平价权重 $\boldsymbol{w}_{RP}$。
2.  **验证**:
    *   计算该权重下各资产的风险贡献 $\text{RC}_i$。
    *   验证 $\text{RC}_i$ 是否大致相等。
3.  **对比**:
    *   对比 GMV, MVO (最大夏普), Risk Parity 三种策略的资产配置差异。

---

## 4. 提交物要求
1.  **代码**: 清晰的 Python 代码，建议封装成函数或类。
2.  **报告**:
    *   展示收益率分布、相关性热力图。
    *   展示有效前沿图，并在图上标记关键点。
    *   表格列出不同模型下的最优权重。
    *   分析不同模型结果的合理性。

## 5. 参考资料
1.  Markowitz, H. (1952). Portfolio Selection.
2.  Black, F., & Litterman, R. (1992). Global Portfolio Optimization.
3.  Maillard, S., Roncalli, T., & Teïletche, J. (2010). The Properties of Equally Weighted Risk Contribution Portfolios.
