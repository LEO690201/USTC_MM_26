# 作业5: 时间序列分析与预测 (Time Series Analysis)

## 1. 实验背景与目的

### 1.1 背景介绍
时间序列（Time Series）是按时间顺序排列的数据点集合。从宏观经济指标（GDP、CPI）到金融市场数据（股价、汇率），再到物联网传感器读数，时间序列无处不在。

预测未来的走势是时间序列分析的核心任务。本实验将涵盖从经典的统计学模型（ARIMA, Kalman Filter）到现代的深度学习模型（LSTM），并结合量化金融场景进行实战。

### 1.2 实验目的
本实验旨在让学生：
1.  **理解** 时间序列的平稳性（Stationarity）及其检验方法（ADF Test）。
2.  **掌握** ARIMA 模型的定阶与参数估计（ACF/PACF）。
3.  **应用** 状态空间模型（Kalman Filter）处理带噪声的动态系统。
4.  **构建** 循环神经网络（LSTM）处理长序列依赖问题。
5.  **设计** 简单的量化交易策略并进行回测。

---

## 2. 数学模型详解

### 2.1 ARIMA 模型

ARIMA (AutoRegressive Integrated Moving Average) 是处理单变量非平稳时间序列的经典方法。

#### 1. AR(p): 自回归模型
当前值 $y_t$ 线性依赖于过去 $p$ 个时刻的值：
$$
y_t = c + \phi_1 y_{t-1} + \dots + \phi_p y_{t-p} + \epsilon_t
$$
*   $\epsilon_t \sim N(0, \sigma^2)$: 白噪声。

#### 2. I(d): 差分 (Integrated)
为了使非平稳序列变得平稳，通常需要进行 $d$ 阶差分：
$$
\Delta y_t = y_t - y_{t-1} \quad (d=1)
$$
$$
\Delta^2 y_t = \Delta y_t - \Delta y_{t-1} \quad (d=2)
$$

#### 3. MA(q): 移动平均模型
当前值 $y_t$ 线性依赖于过去 $q$ 个时刻的预测误差：
$$
y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + \dots + \theta_q \epsilon_{t-q}
$$

#### ARIMA(p,d,q) 完整形式
结合上述三部分，对 $d$ 阶差分后的序列 $y'_t$ 建立 ARMA(p,q) 模型：
$$
y'_t = c + \sum_{i=1}^p \phi_i y'_{t-i} + \epsilon_t + \sum_{j=1}^q \theta_j \epsilon_{t-j}
$$

#### 参数选择 (Order Selection)
*   **ACF (自相关函数)**: 用于确定 MA 的阶数 $q$（截尾性）。
*   **PACF (偏自相关函数)**: 用于确定 AR 的阶数 $p$（截尾性）。
*   **信息准则 (AIC/BIC)**: 选择使 AIC/BIC 最小的模型。

---

### 2.2 状态空间模型与卡尔曼滤波 (Kalman Filter)

卡尔曼滤波是一种递归估计算法，用于从含噪声的观测数据中估计动态系统的内部状态。

#### 状态方程 (State Equation)
描述隐藏状态 $\boldsymbol{x}_t$ 如何随时间演变：
$$
\boldsymbol{x}_t = \boldsymbol{F}_t \boldsymbol{x}_{t-1} + \boldsymbol{B}_t \boldsymbol{u}_t + \boldsymbol{w}_t
$$
*   $\boldsymbol{x}_t$: $n \times 1$ 状态向量。
*   $\boldsymbol{F}_t$: $n \times n$ 状态转移矩阵。
*   $\boldsymbol{w}_t \sim N(0, \boldsymbol{Q}_t)$: 过程噪声。

#### 观测方程 (Measurement Equation)
描述观测值 $\boldsymbol{z}_t$ 与状态 $\boldsymbol{x}_t$ 的关系：
$$
\boldsymbol{z}_t = \boldsymbol{H}_t \boldsymbol{x}_t + \boldsymbol{v}_t
$$
*   $\boldsymbol{z}_t$: $m \times 1$ 观测向量。
*   $\boldsymbol{H}_t$: $m \times n$ 观测矩阵。
*   $\boldsymbol{v}_t \sim N(0, \boldsymbol{R}_t)$: 观测噪声。

#### 滤波过程
1.  **预测 (Predict)**:
    $$
    \hat{\boldsymbol{x}}_{t|t-1} = \boldsymbol{F}_t \hat{\boldsymbol{x}}_{t-1|t-1}
    $$
    $$
    \boldsymbol{P}_{t|t-1} = \boldsymbol{F}_t \boldsymbol{P}_{t-1|t-1} \boldsymbol{F}_t^T + \boldsymbol{Q}_t
    $$
2.  **更新 (Update)**:
    $$
    \boldsymbol{K}_t = \boldsymbol{P}_{t|t-1} \boldsymbol{H}_t^T (\boldsymbol{H}_t \boldsymbol{P}_{t|t-1} \boldsymbol{H}_t^T + \boldsymbol{R}_t)^{-1} \quad (\text{卡尔曼增益})
    $$
    $$
    \hat{\boldsymbol{x}}_{t|t} = \hat{\boldsymbol{x}}_{t|t-1} + \boldsymbol{K}_t (\boldsymbol{z}_t - \boldsymbol{H}_t \hat{\boldsymbol{x}}_{t|t-1})
    $$
    $$
    \boldsymbol{P}_{t|t} = (\boldsymbol{I} - \boldsymbol{K}_t \boldsymbol{H}_t) \boldsymbol{P}_{t|t-1}
    $$

---

### 2.3 长短期记忆网络 (LSTM)

RNN 的变体，解决了长序列训练中的梯度消失问题。

#### 核心组件：门 (Gates)
LSTM 单元包含三个门，控制信息流：
1.  **遗忘门 (Forget Gate)** $f_t$: 决定丢弃多少旧的细胞状态 $C_{t-1}$。
    $$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$
2.  **输入门 (Input Gate)** $i_t$: 决定更新多少新的信息 $\tilde{C}_t$ 到细胞状态。
    $$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) $$
    $$ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) $$
3.  **细胞状态更新**:
    $$ C_t = f_t * C_{t-1} + i_t * \tilde{C}_t $$
4.  **输出门 (Output Gate)** $o_t$: 决定输出多少隐藏状态 $h_t$。
    $$ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) $$
    $$ h_t = o_t * \tanh(C_t) $$

---

## 3. 实验任务详解

### 任务 1: 数据准备与预处理 (难度: ⭐⭐)
1.  **数据**: 使用 `yfinance` 下载某只股票（如 AAPL）过去 5 年的日收盘价。
2.  **平稳性检验**:
    *   绘制原始序列图。
    *   进行 ADF 检验 (`statsmodels.tsa.stattools.adfuller`)。
    *   若 $p > 0.05$，进行一阶差分，再次检验直到平稳。确定 $d$ 值。
3.  **数据切分**: 前 80% 为训练集，后 20% 为测试集。

### 任务 2: ARIMA 模型预测 (难度: ⭐⭐⭐)
1.  **定阶**:
    *   绘制训练集差分序列的 ACF 和 PACF 图。
    *   根据截尾性初步确定 $p, q$。或者使用 `pmdarima.auto_arima` 自动搜索最优 $(p,d,q)$。
2.  **拟合与预测**:
    *   使用 `statsmodels.tsa.arima.model.ARIMA` 拟合模型。
    *   对测试集进行滚动预测（Rolling Forecast）：每次预测一步，然后将真实值加入已知历史，再预测下一步。
3.  **评估**:
    *   计算 RMSE (Root Mean Squared Error) 和 MAE。
    *   绘制预测值与真实值的对比图。

### 任务 3: LSTM 模型预测 (难度: ⭐⭐⭐⭐)
1.  **数据构造**:
    *   将时间序列转换为监督学习样本 (X, y)。
    *   **滑动窗口**: 设窗口大小 $L=60$，即用过去 60 天预测第 61 天。
    *   **归一化**: 使用 `MinMaxScaler` 将数据缩放到 $[0, 1]$。
2.  **模型构建**:
    *   搭建 LSTM 网络（例如 2 层 LSTM + 1 层 Dense）。
    *   Loss: MSE, Optimizer: Adam。
3.  **训练与评估**:
    *   训练模型，注意防止过拟合（使用 Dropout 或 Early Stopping）。
    *   对测试集进行预测，反归一化。
    *   对比 LSTM 与 ARIMA 的预测精度。

### 任务 4: 量化策略回测 (难度: ⭐⭐⭐⭐⭐)
1.  **策略设计**:
    *   **双均线策略**: 当短期均线 (MA5) 上穿长期均线 (MA20) 时买入（金叉），下穿时卖出（死叉）。
    *   **动量策略**: 若过去 $N$ 天收益率为正，则持有；否则空仓。
2.  **回测假设**:
    *   初始资金 100,000 元。
    *   全仓买卖（不考虑仓位管理）。
    *   无交易成本（手续费、滑点设为 0）。
3.  **指标计算**:
    *   计算策略的**累计收益率**。
    *   计算**最大回撤** (Max Drawdown)。
    *   计算**夏普比率** (Sharpe Ratio)。
    *   绘制资金曲线，并与“买入持有”（Buy & Hold）策略对比。

---

## 4. 提交物要求
1.  **代码**: 包含 ARIMA, LSTM, 策略回测的完整代码。
2.  **报告**:
    *   ADF 检验结果及解释。
    *   ARIMA 定阶过程（ACF/PACF 图）。
    *   ARIMA vs LSTM 预测对比图及误差分析。
    *   量化策略的资金曲线图及绩效指标表。

## 5. 参考资料
1.  Box, G. E. P., et al. (2015). Time Series Analysis: Forecasting and Control.
2.  Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
3.  Welch, G., & Bishop, G. (1995). An Introduction to the Kalman Filter.
