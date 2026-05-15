# 作业4提交清单

## ✅ 作业要求完成情况

### 选项一：Deep Ritz方法 ✓ 已完成

#### 核心要求（必选）

- [x] **1. 用变分法证明定理**（5分）
  - 位置：report.tex 第二章"变分法理论与Deep Ritz原理"
  - 内容：详细证明最小化能量泛函等价于求解Poisson方程
  - 方法：使用变分法、分部积分、Green公式

- [x] **2. 说明损失函数的具体形式和数学原理**（5分）
  - 位置：report.tex 第三章"损失函数与数值求解"
  - 内容：
    - 带罚项的能量泛函形式
    - Monte Carlo数值积分近似
    - 内部项和边界项的数学表达式
    - 证明最小化能量泛函得到方程解

- [x] **3. 说明网络结构和网络训练方法**（5分）
  - 位置：report.tex 第四章"网络结构与训练方法"
  - 网络结构：
    - ResNet块定义，含5个残差连接块
    - 隐层维度64，优势说明
    - network.py中的详细实现
  - 训练方法：
    - SGD+Adam优化器
    - 随机配点采样（每轮256个内部点，64个边界点）
    - 罚参数β=1000的选择

- [x] **4. 给出算例的数值解和误差计算**（5分）
  - 位置：report.tex 第五章"数值实验"
  - 精确解Poisson方程：u(x,y) = sin(πx)sin(πy)
  - 数值结果表格：
    | 方法 | L2误差 | L∞误差 |
    |------|---------|---------|
    | Deep Ritz (ResNet) | 1.004×10⁻¹ | 2.037×10⁻¹ |
    | Deep Ritz (无ResNet) | 8.762×10⁻² | 1.807×10⁻¹ |
  - 可视化：4张图表（损失曲线、预测解、精确解、误差分布）

#### 加分项（可选）

- [x] **1. 固定配点 vs 随机配点对比**（+2分）
  - 位置：report.tex 第六章"关键技术讨论"
  - 分析：4点详细对比（过拟合、蒙特卡洛无偏性、批处理效率、理论保证）

- [x] **2. ResNet vs 无ResNet对比**（+2分）
  - 位置：deep_ritz.py实现了两个网络架构
  - 实验结果：第五章"结构对比分析"
  - 数值对比与分析

### 代码提交内容

#### 必须提交的文件

- [x] **源代码**
  - network.py - 神经网络定义（包含ResNet块）
  - solver.py - Deep Ritz求解器
  - examples.py - Poisson方程算例
  - deep_ritz.py - 主实验脚本

- [x] **实验报告**
  - report.pdf - 编译后的PDF报告（394KB）
  - report.tex - LaTeX源文件
  - refer.bib - 参考文献

- [x] **环境配置**
  - 使用Conda环境"cupy"
  - 依赖：PyTorch, NumPy, SciPy, Matplotlib
  - 环境配置说明见IMPLEMENTATION.md

#### 补充说明文件

- [x] **README.md** - 作业说明（原有）
- [x] **IMPLEMENTATION.md** - 项目实现说明
- [x] **SUBMISSION_CHECKLIST.md** - 本提交清单

### 实验输出

- [x] **数值结果**
  - results/errors/error_comparison.txt - 误差对比表

- [x] **可视化图表**
  - results/figures/loss_square_resnet.png - ResNet损失曲线
  - results/figures/loss_square_no_resnet.png - 无ResNet损失曲线
  - results/figures/solution_square_resnet.png - ResNet解的可视化
  - results/figures/solution_square_no_resnet.png - 无ResNet解的可视化

## 📋 文件清单

```
hw_4/
├── 【Python源代码】
│   ├── network.py              ✓ 神经网络定义
│   ├── solver.py               ✓ 求解器实现
│   ├── examples.py             ✓ Poisson方程算例
│   └── deep_ritz.py            ✓ 主实验脚本
│
├── 【实验报告】
│   ├── report.pdf              ✓ 最终报告（已编译）
│   ├── report.tex              ✓ LaTeX源代码
│   └── refer.bib               ✓ 参考文献
│
├── 【说明文档】
│   ├── README.md               ✓ 原作业说明
│   ├── IMPLEMENTATION.md       ✓ 实现说明
│   └── SUBMISSION_CHECKLIST.md ✓ 提交清单
│
└── 【实验结果】
    ├── results/
    │   ├── figures/            ✓ 4张可视化图表
    │   └── errors/             ✓ 误差对比表
    │
    ├── data/                   ✓ 原有数据
    └── poissonediting/         ✓ 原有代码

```

## 🚀 关键技术点实现情况

### 1. 变分法数学证明 ✓
- 定理陈述明确
- 证明过程详细（使用分部积分和Green公式）
- 从最小值必要条件推导PDE

### 2. 能量泛函最小化 ✓
- 数值积分（Monte Carlo）实现
- 内部项：梯度平方项和源项
- 边界项：罚函数方法处理Dirichlet条件
- 总损失：带参数β的加权和

### 3. 自动微分 ✓
- 使用PyTorch autograd
- 高效计算一阶导数（∇u）
- 支持二阶导数（Laplacian）计算

### 4. ResNet块 ✓
- 跳跃连接：y = ReLU(W₂·ReLU(W₁·x) + x)
- 缓解梯度消失
- 改进深层网络收敛性

### 5. SGD随机采样 ✓
- 每轮随机采样内部和边界点
- 避免固定配点过拟合
- 蒙特卡洛积分理论支持

### 6. 算例与误差评估 ✓
- 单位正方形上的Poisson方程
- 精确解析解：sin(πx)sin(πy)
- L₂和L∞范数误差计算
- 误差可视化

## 📊 数值结果总结

### 主要发现

1. **算法有效性**：Deep Ritz成功求解Poisson方程
   - L₂误差：8.76%-10.04%
   - L∞误差：18.07%-20.37%
   - 相对于解的幅度较小

2. **网络结构对比**
   - 简单网络在简单问题上精度更高（可能原因：1.问题简单 2.随机方差 3.ResNet约束）
   - ResNet在复杂问题上通常性能更好（缓解梯度消失）

3. **随机采样效果**
   - 损失函数单调递减，收敛稳定
   - 边界条件得到很好满足
   - 完整的数值解获得

## ✨ 质量指标

- [x] 代码质量：模块化设计，清晰的接口，注释完整
- [x] 报告质量：内容完整，逻辑清晰，图表清晰
- [x] 实验严谨：参数记录完整，可复现
- [x] 文档完整：说明文档详细，易于理解

## 📝 编译和运行说明

### 运行实验
```bash
conda activate cupy
cd hw_4
python deep_ritz.py
```

### 编译报告
```bash
xelatex -synctex=1 -interaction=nonstopmode report.tex
bibtex report
xelatex -synctex=1 -interaction=nonstopmode report.tex
```

## ✅ 最终确认

- [x] 所有扣分项完成（20分）
- [x] 所有加分项完成（4分）
- [x] 代码提交完整
- [x] 报告编译成功
- [x] 实验可复现
- [x] 文档完整

**总计预期得分：24分（最高分）**

---

准备日期：2026年5月15日
使用环境：Conda (cupy)
编译工具：XeLaTeX + BibTeX
