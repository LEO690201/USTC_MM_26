# 作业4：微分方程模型 (Unfinished)

> 作业 4 文档目前尚未编写完成

> 作业 ddl：2025年5月17日（周日）06:00

> 迟交政策：5月20日课前提交不超过80分，课后提交不超过60分

> 提交内容：源代码 + 实验报告（pdf文件） + 其他补充说明文件（可选）

在复杂多变的现实世界中，许多自然现象和社会问题都蕴含着动态变化的规律，而这些规律往往可以通过微分方程（Differential Equations）来刻画。在本次实验中，我们将通过如下三个选项来练习求解微分方程的数值解、使用微分方程求解实际问题、利用微分方程得到高效算法。

## 选项 1：Deep Ritz

**关键词：机器学习，变分法，偏微分方程数值解**

偏微分方程数值解一直是科学计算领域最重要的课题之一。在过去的数十年中，数学家们对**有限差分法 (FDM, Finite Difference Method)**，**有限元法 (FEM, Finite Element Method)**，**谱方法 (Spectral Method)** 等传统方法进行了极其详尽的数值分析。然而，我们不难留意到这些方法都依赖于生成高质量的网格，这使得其求解复杂区域的 PDE 时往往面临困难。

近年来，随着机器学习领域的迅速发展和 **Universal Approximation Theorem**，不少学者开始关注其在 PDE 数值解领域的应用。基于将神经网络作为 PDE 解的近似器这一基本思想，**PINNs (Physics-Informed Neural Networks, 2019)**，**Deep Ritz (2018)**，**WAN (Weak Adversarial Networks, 2020)** 等无需网格的机器学习方法应运而生. 在本选项中, 我们的任务是复现 Deep Ritz 算法，理解其数学原理和思想。

考虑一具有变分结构的 Euler–Lagrange 方程，如 Poisson 方程。比如说，我们想要求解一 Dirichlet 边界条件的 Poisson 方程：

$$
\begin{cases}
-\Delta u(x) = f(x), & x \in \Omega, \\
u(x) = g(x), & x \in \partial\Omega.
\end{cases}
$$

变分法表明，求解此方程等价于最小化一能量泛函，见如下定理。

**定理** 考虑合适的函数空间，在容许函数类 $H=\{ u \mid u(x)=g(x),\ x\in\partial\Omega \}$ 上定义能量泛函

$$
J[u]=\int_{\Omega}\left(\frac12|\nabla u(x)|^2-f(x)u(x)\right)dx.
$$

若 $u^*=\arg\min_{u\in H}J[u]$，则

$$
\begin{cases}
-\Delta u^* (x) = f(x), & x \in \Omega, \\
u^* (x) = g(x), & x \in \partial\Omega.
\end{cases}
$$

请各位同学在报告中利用变分法证明此定理。

事实上，我们可以对能量泛函稍作修改，添加一边界罚项：

$$
J[u]=\int_{\Omega}\left(\frac12|\nabla u(x)|^2-f(x)u(x)\right) dx
+\beta\int_{\partial\Omega}(u(x)-g(x))^2 dS,
$$

其中 $\beta$ 为罚参数，一般取值较大。若能最小化此能量泛函，则对应的函数即为方程的解。

现在，我们可以用神经网络 $u(\theta)$ 来逼近函数的解，其中 $\theta$ 为网络参数。自然地，能量泛函可作为神经网络的损失函数：

$$
L(\theta)=\int_{\Omega}\left(\frac12|\nabla u(x;\theta)|^2-f(x)u(x;\theta)\right) dx
+\beta\int_{\partial\Omega}(u(x;\theta)-g(x))^2 dS.
$$

虽然我们无法直接计算积分，但可以在区域和边界上离散采样点以计算**数值积分 (Quadrature)**。此时，最小化损失函数即可得到方程的数值解。这就是 Deep Ritz 的基本原理。

此外，论文还给出了以下两个算法细节，鉴于其思想非常经典，故将其作为作业的必选内容：

- 使用**随机梯度下降 (SGD)**：每轮训练时，在区域和边界上随机采样小批量的点（独立采样，均匀分布）用于梯度下降。（加分项：若采用固定配点结果会如何？展示结果并分析 +2）

- 使用**残差连接 (ResNet)**：在神经网络中引入 ResNet 能够有效解决梯度消失问题并提高解的精度。（加分项：若不采用 ResNet 结构会如何？展示结果并分析 +2）

### 实验要求

选项 1 的基本要求如下：
- 
- 
- 

注：
- 本选项适合对神经网络和微分方程有一定了解的同学
- 请提交环境配置文件 (C++ / Python)
- 不必制作图形用户界面 (GUI)

参考文献：[Deep Ritz](https://link.springer.com/article/10.1007/s40304-018-0127-z)

## 选项 2：传染病模型

传染病模型用于研究疾病在人群中传播的动态变化规律。通过建立数学模型，可以预测疫情的发展趋势，评估防控措施的效果，并为公共卫生决策提供科学依据。常见的传染病模型包括SIR、SEIR和SIS模型，它们根据疾病的不同传播特点进行建模，能够有效刻画疾病在人群中的传播过程。

### SIR模型

#### 基本原理

SIR模型将总人口分为三个状态：
- **S (Susceptible)**：易感者，尚未感染但有可能感染疾病的人群；
- **I (Infectious)**：感染者，已经感染且具有传染性的人群；
- **R (Recovered)**：移除者，已经康复或死亡的人群，不再具有传染性。

模型假设总人口数固定（无出生、死亡或迁移），传播过程由以下微分方程描述：

$$
\frac{dS}{dt} = -\beta SI, \quad \frac{dI}{dt} = \beta SI - \gamma I, \quad \frac{dR}{dt} = \gamma I
$$

其中， $\beta$ 是传染率， $\gamma$ 是恢复率。

### SEIR模型

#### 基本原理

SEIR模型在SIR基础上增加了一个潜伏期阶段E（Exposed）：
- **E (Exposed)**：已感染但尚未具有传染性的人群。

微分方程组为：

$$
\frac{dS}{dt} = -\beta SI, \quad \frac{dE}{dt} = \beta SI - \sigma E, \quad \frac{dI}{dt} = \sigma E - \gamma I, \quad \frac{dR}{dt} = \gamma I
$$

其中， $\sigma$ 是潜伏期转为感染期的速率。

### SIS模型

#### 基本原理

SIS模型假设感染者康复后不会获得免疫，而是重新回到易感状态，常用于描述某些细菌感染或流行性感冒类疾病。模型包含两个状态：
- **S (Susceptible)**：易感者；
- **I (Infectious)**：感染者。

微分方程组为：

$$
\frac{dS}{dt} = -\beta SI + \gamma I, \quad \frac{dI}{dt} = \beta SI - \gamma I
$$

其中， $\gamma$ 仍表示恢复率。

### 如何求解

使用数值方法（如Euler法、Runge-Kutta）对微分方程组进行求解。常见流程为：
1. 给定初始条件 $S(0), I(0), R(0)$ 。
2. 选择步长 $dt$，迭代计算 $S(t), I(t), R(t)$ 。
3. 根据数值解绘制感染人数随时间变化的曲线。

### 实验要求

选项2的基本要求如下：
- 建立至少两种传染病模型（如SIR和SEIR），描述并推导其数学形式；
- 设计合理的参数并进行数值实验，绘制疫情传播随时间变化的曲线；
- 对比不同参数设置（如传染率、恢复率变化）对疾病传播趋势的影响，并进行合理分析。

注：
- 实验过程中需记录并展示参数设置与初值选择；
- 所有推导过程需完整展示。

可能的加分项：
- 使用多种不同的ODE求解手段，分析差异；
- 将模型扩展到**随机微分方程版**（SDE，如添加噪声项）；
- 设计控制策略（如增加隔离、疫苗接种），并用模型验证其效果。


## 选项 3：Poisson 图像融合

实际上，这是计算机图形学课程的第三个作业，相关的算法原理可以参考文档：[计算机图形学课程文档](https://github.com/USTC-CG/USTC_CG_25/tree/main/Homeworks/3_poisson_image_editing)，本课程为该算法提供了一个Matlab框架，可参考[Matlab框架](./poissonediting)

### 实验要求

选项 3 的基本要求如下：
- 完成文中的 Seamless cloning 的 Importing gradients 部分，务必写清楚该算法的原理，首先需要从偏微分方程开始分析，再到稀疏矩阵方程的构造；
- 实时拖动区域显示结果，需要录制视频；
- 请采用多个测试样例（大于等于4个）来直观说明算法的合理性。

注：
- 不选项不限制编程语言，可以利用框架[Matlab框架](./poissonediting)，也可以利用IMGUI等工具搭建C++框架，但是务必录制实时融合的视频；
- 由于Matlab框架可以实现不规则区域的融合，因此如果使用C++框架但没有实现该结果，会扣除很多分数。

可能的加分项：对梯度进行一些修正，比如使用Mixing gradients方法，并与现在的结果进行比较。
