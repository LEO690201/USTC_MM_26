# 作业4：Deep Ritz方法求解Poisson方程

## 项目概述

本项目实现了Deep Ritz方法，用于求解偏微分方程（特别是Poisson方程）。Deep Ritz是一种基于变分原理的无网格神经网络方法，将PDE求解转化为神经网络参数优化问题。

## 环境配置

### 使用的环境
- **Python环境**：Conda环境 `cupy`
- **主要依赖**：
  - PyTorch 2.10.0 (CUDA)
  - NumPy 2.3.5
  - SciPy 1.16.3
  - Matplotlib 3.10.8

### 激活环境
```bash
conda activate cupy
```

## 文件结构

```
hw_4/
├── README.md                    # 作业说明文档
├── report.tex                   # LaTeX实验报告源文件
├── report.pdf                   # 编译生成的实验报告（PDF）
├── refer.bib                    # 参考文献
├── network.py                   # 神经网络定义
├── solver.py                    # Deep Ritz求解器
├── examples.py                  # Poisson方程算例
├── deep_ritz.py                 # 主实验脚本
├── data/                        # 原有数据目录
├── poissonediting/              # 原有Poisson图像融合代码
└── results/                     # 实验输出目录
    ├── figures/                 # 可视化结果
    │   ├── loss_square_resnet.png
    │   ├── loss_square_no_resnet.png
    │   ├── solution_square_resnet.png
    │   └── solution_square_no_resnet.png
    └── errors/                  # 误差结果
        └── error_comparison.txt
```

## 核心模块说明

### 1. network.py
定义了两种神经网络架构：

- **DeepRitzNetwork**：使用ResNet块的深层网络
  - 输入层：投影到隐层（维度64）
  - 5个ResNet块，每块包含残差连接
  - 输出层：映射到标量值
  
- **DeepRitzNetworkNoResNet**：简单全连接网络（用于对比）

ResNet块的作用是缓解深层网络的梯度消失问题，改进收敛性。

### 2. solver.py
实现了Deep Ritz求解器类：

- **compute_gradients**：使用自动微分计算神经网络的导数
- **compute_loss**：实现带罚项的能量泛函
  - 内部积分：积分项 $\int (1/2|\nabla u|^2 - fu)dx$
  - 边界积分：罚项 $\beta \int (u-g)^2 dS$
  
- **train**：SGD训练循环，使用随机配点采样
- **predict**、**compute_l2_error**、**compute_linf_error**：评估函数

### 3. examples.py
定义了多个Poisson方程的测试算例：

- **SquarePoissonExample**：单位正方形，齐次边界条件
  - 精确解：$u(x,y) = \sin(\pi x)\sin(\pi y)$
  - 源项：$f = 2\pi^2\sin(\pi x)\sin(\pi y)$
  
- **SquareHomogeneousPoissonExample**：非齐次边界条件
- **CircularDiskPoissonExample**：圆形区域

### 4. deep_ritz.py
主实验脚本，包含：

- **run_experiment**：执行一次完整实验
- **plot_results**：绘制损失曲线和解的可视化
- 主程序运行两个对比实验（ResNet vs 无ResNet）

## 使用方法

### 运行实验

在cupy环境中执行：

```bash
conda activate cupy
cd hw_4
python deep_ritz.py
```

该脚本会：
1. 训练ResNet版本的Deep Ritz网络（3000轮迭代）
2. 训练无ResNet版本进行对比
3. 生成误差对比表和可视化图表
4. 保存结果到`results/`目录

### 自定义实验

可以在Python中调用`run_experiment`函数进行自定义实验：

```python
from deep_ritz import run_experiment, plot_results

# 运行自定义实验
solver, example, test_pts, result = run_experiment(
    example_name='square',
    use_resnet=True,
    num_epochs=5000,
    batch_size_interior=512,
    hidden_dim=128,
    num_blocks=6
)

# 绘制结果
plot_results(solver, example, test_pts, result)
```

## 数值实验结果

### 误差对比

| 方法 | L2误差 | L∞误差 |
|------|---------|---------|
| Deep Ritz (ResNet) | 1.004×10⁻¹ | 2.037×10⁻¹ |
| Deep Ritz (无ResNet) | 8.762×10⁻² | 1.807×10⁻¹ |

### 关键观察

1. **无ResNet版本精度更高**：在这个相对简单的问题上，简化的网络结构反而效率更高。这可能因为：
   - 问题相对简单，不需要很深的网络
   - 随机初始化导致的方差
   - ResNet块对这个问题引入了额外约束

2. **收敛性**：两种网络都能够收敛，损失函数单调递减

3. **边界条件**：使用罚参数$\beta=1000$，能够很好地满足边界条件

## 实验报告

详细的理论分析和实验说明见**report.pdf**，包含以下章节：

1. **一、问题描述**：Poisson方程的表述
2. **二、变分法理论与Deep Ritz原理**：定理证明和算法原理
3. **三、损失函数与数值求解**：具体实现细节
4. **四、网络结构与训练方法**：ResNet设计和SGD算法
5. **五、数值实验**：测试算例和结果
6. **六、关键技术讨论**：随机采样、罚参数等分析
7. **七、结论与展望**：主要发现和未来方向

## 关键技术特点

### 1. 随机配点采样（Random Sampling）
- 每轮训练随机采样内部和边界点
- 优势：避免过拟合，改进泛化性，符合蒙特卡洛积分理论

### 2. 残差连接（ResNet Blocks）
- 使用跳跃连接缓解梯度消失
- 允许更深的网络而不损失精度

### 3. 罚函数方法（Penalty Method）
- 将约束条件（边界条件）融入目标函数
- 通过罚参数$\beta$平衡两项

### 4. 自动微分
- 使用PyTorch自动微分计算一阶和二阶导数
- 高效且精确

## 扩展和改进方向

1. **自适应采样**：在误差大的区域增加采样点
2. **多种算例**：测试更复杂的PDE和非凸区域
3. **与PINNs结合**：直接编码PDE残差约束
4. **超参数优化**：使用贝叶斯优化选择最优参数
5. **高维问题**：探索应对维数诅咒的策略

## 编译LaTeX报告

如果需要重新编译报告：

```bash
cd hw_4
xelatex -synctex=1 -interaction=nonstopmode report.tex
bibtex report
xelatex -synctex=1 -interaction=nonstopmode report.tex
xelatex -synctex=1 -interaction=nonstopmode report.tex
```

或使用latexmk：

```bash
latexmk -xelatex -interaction=nonstopmode report.tex
```

## 参考文献

主要参考文献包含在refer.bib中：

- E et al. (2018) - The Deep Ritz Method: A Deep Learning-Based Numerical Algorithm
- Raissi et al. (2019) - Physics-Informed Neural Networks
- He et al. (2016) - Deep Residual Learning (ResNet)

## 许可证

本项目用于教学和研究目的。

## 联系方式

有任何问题或建议，请参考作业说明或与指导教师联系。
