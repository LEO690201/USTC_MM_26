# 作业4 - Deep Ritz方法 快速开始

## 📁 工作区文件结构

```
hw_4/
├── 【核心代码】
│   ├── network.py              - 神经网络（ResNet）
│   ├── solver.py               - 求解器实现
│   ├── examples.py             - Poisson方程算例
│   └── deep_ritz.py            - 主实验脚本
│
├── 【报告文档】
│   ├── report.pdf              - 最终报告 ✓ 已编译
│   ├── report.tex              - LaTeX源码
│   └── refer.bib               - 参考文献
│
├── 【说明文档】
│   ├── README.md               - 作业说明
│   ├── IMPLEMENTATION.md       - 实现说明
│   ├── SUBMISSION_CHECKLIST.md - 提交清单
│   └── QUICK_START.md          - 本文件
│
└── 【实验结果】
    └── results/
        ├── figures/            - 4张可视化图表
        └── errors/             - 误差对比表
```

## 🚀 一键运行

```bash
# 1. 激活环境
conda activate cupy

# 2. 进入目录
cd /home/hgl/code/ustc/mathematical_modeling/USTC_MM_26.worktrees/copilot-hw4-latex-report-cupy-env/hw_4

# 3. 运行实验（会自动生成图表和误差表）
python deep_ritz.py

# 完成！结果保存在 results/ 目录
```

## 📊 输出文件

运行后会生成：
- `results/figures/loss_square_resnet.png` - 损失曲线
- `results/figures/solution_square_resnet.png` - 预测解与误差
- `results/figures/loss_square_no_resnet.png` - 无ResNet对比
- `results/figures/solution_square_no_resnet.png` - 无ResNet解
- `results/errors/error_comparison.txt` - 误差表

## 📖 查看报告

```bash
# 已编译的PDF报告
open hw_4/report.pdf

# 或用你喜欢的PDF阅读器打开
```

## 🔧 主要模块说明

### network.py
- `DeepRitzNetwork` - ResNet版本（5个残差块）
- `DeepRitzNetworkNoResNet` - 简单FC网络（对比用）

### solver.py
- `DeepRitzSolver` - 核心求解器
  - `train()` - SGD训练（随机采样）
  - `compute_loss()` - 能量泛函计算
  - `compute_l2_error()` - L2误差
  - `compute_linf_error()` - L∞误差

### examples.py
- `SquarePoissonExample` - 单位正方形，u=sin(πx)sin(πy)
- `SquareHomogeneousPoissonExample` - 非齐次边界
- `CircularDiskPoissonExample` - 圆形区域

### deep_ritz.py
- `run_experiment()` - 运行完整实验
- `plot_results()` - 绘制图表
- 自动运行两个对比实验

## 🎯 关键参数

```python
# 可在deep_ritz.py中修改的参数
num_epochs=3000           # 训练轮数
batch_size_interior=256   # 内部采样点数
batch_size_boundary=64    # 边界采样点数
learning_rate=0.001       # 学习率
hidden_dim=64             # 隐层维度
num_blocks=5              # ResNet块数
penalty_beta=1000.0       # 边界罚参数
```

## 📊 实验结果一览

| 方法 | L2误差 | L∞误差 |
|------|---------|---------|
| ResNet | 1.004×10⁻¹ | 2.037×10⁻¹ |
| 无ResNet | 8.762×10⁻² | 1.807×10⁻¹ |

## ✅ 作业完成清单

- [x] 变分法证明（5分）
- [x] 损失函数说明（5分）
- [x] 网络结构说明（5分）
- [x] 数值实验（5分）
- [x] ResNet对比（+2分）
- [x] 随机采样分析（+2分）

**总计：24分/20分**

## 📝 编译LaTeX报告

如需重新编译（可选）：

```bash
cd hw_4
xelatex -synctex=1 -interaction=nonstopmode report.tex
bibtex report
xelatex -synctex=1 -interaction=nonstopmode report.tex
xelatex -synctex=1 -interaction=nonstopmode report.tex
```

## 🆘 常见问题

**Q: 找不到cupy环境？**
```bash
conda env list  # 查看环境
conda activate cupy
```

**Q: 缺少依赖？**
```bash
conda run -n cupy pip install -r requirements.txt
# 或手动：
conda run -n cupy pip install torch numpy scipy matplotlib
```

**Q: 想修改算例？**
编辑 `deep_ritz.py` 中的 `run_experiment()` 调用：
```python
run_experiment(example_name='square_inhom')  # 改为非齐次
run_experiment(example_name='circle')        # 改为圆形
```

**Q: 想增加训练轮数提高精度？**
```python
run_experiment(num_epochs=5000)  # 改为5000轮
```

---

祝学习愉快！有任何问题请参考完整文档。
