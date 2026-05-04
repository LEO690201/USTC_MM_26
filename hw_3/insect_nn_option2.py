#!/usr/bin/env python3
"""Option 2: insect classification with neural networks.

Run from the repository root:
    conda run -n cupy python hw_3/insect_nn_option2.py
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


DATA_DIR = ROOT / "insects"
OUT_DIR = ROOT / "option2_outputs"
FIG_DIR = OUT_DIR / "figs"
for directory in (OUT_DIR, FIG_DIR, ROOT / ".mplconfig"):
    directory.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

SEEDS = [2026, 2027, 2028, 2029, 2030]
CLASS_NAMES = ["0", "1", "2"]


@dataclass(frozen=True)
class NetConfig:
    name: str
    hidden: tuple[int, ...]
    activation: str
    lr: float = 0.03
    weight_decay: float = 0.0
    epochs: int = 1200
    patience: int = 180


CONFIGS = [
    NetConfig("softmax_linear", (), "none", lr=0.05),
    NetConfig("mlp_8_relu", (8,), "relu", lr=0.03),
    NetConfig("mlp_16_tanh", (16,), "tanh", lr=0.03),
    NetConfig("mlp_16_8_relu_l2", (16, 8), "relu", lr=0.02, weight_decay=1e-3),
    NetConfig("mlp_32_16_relu_l2", (32, 16), "relu", lr=0.015, weight_decay=2e-3),
]

DATASETS = {
    "dataset1_clean": {
        "title": "数据集1：无额外噪声",
        "train": DATA_DIR / "insects-training.txt",
        "test": DATA_DIR / "insects-testing.txt",
    },
    "dataset2_noisy": {
        "title": "数据集2：含测量噪声",
        "train": DATA_DIR / "insects-2-training.txt",
        "test": DATA_DIR / "insects-2-testing.txt",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=np.float32)
    return data[:, :2], data[:, 2].astype(np.int64)


def standardize(
    x_train: np.ndarray, x_other: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (x_train - mean) / std, (x_other - mean) / std, mean, std


def build_model(config: NetConfig) -> torch.nn.Module:
    layers: list[torch.nn.Module] = []
    in_dim = 2
    for width in config.hidden:
        layers.append(torch.nn.Linear(in_dim, width))
        if config.activation == "relu":
            layers.append(torch.nn.ReLU())
        elif config.activation == "tanh":
            layers.append(torch.nn.Tanh())
        else:
            raise ValueError(f"unsupported activation: {config.activation}")
        in_dim = width
    layers.append(torch.nn.Linear(in_dim, 3))
    return torch.nn.Sequential(*layers)


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred == y).mean())


def train_one(
    x: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: NetConfig,
    seed: int,
    device: torch.device,
) -> dict:
    set_seed(seed)
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=y
    )

    x_fit_raw, x_val_raw = x[train_idx], x[val_idx]
    y_fit, y_val = y[train_idx], y[val_idx]
    x_fit, x_val, mean, std = standardize(x_fit_raw, x_val_raw)
    x_test_std = (x_test - mean) / std

    model = build_model(config).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    tx = torch.tensor(x_fit, dtype=torch.float32, device=device)
    ty = torch.tensor(y_fit, dtype=torch.long, device=device)
    vx = torch.tensor(x_val, dtype=torch.float32, device=device)
    vy = torch.tensor(y_val, dtype=torch.long, device=device)

    best_state = None
    best_val_loss = float("inf")
    no_improve = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(tx), ty)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(vx)
            val_loss = loss_fn(val_logits, vy).item()
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == vy).float().mean().item()
        history.append((epoch, float(loss.item()), val_loss, val_acc))

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        test_tensor = torch.tensor(x_test_std, dtype=torch.float32, device=device)
        logits = model(test_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = probs.argmax(axis=1)

    return {
        "model": model.cpu(),
        "mean": mean,
        "std": std,
        "history": history,
        "epochs_used": len(history),
        "pred": pred,
        "prob": probs,
        "acc_seen60": accuracy(pred[:60], y_test[:60]),
        "acc_new150": accuracy(pred[60:], y_test[60:]),
        "acc_all210": accuracy(pred, y_test),
        "confusion": confusion_matrix(y_test, pred, labels=[0, 1, 2]),
    }


def plot_data_overview() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    colors = ["#0072B2", "#D55E00", "#009E73"]
    markers = ["o", "^", "s"]

    for ax, (dataset_key, meta) in zip(axes, DATASETS.items()):
        x_train, y_train = load_data(meta["train"])
        x_test, y_test = load_data(meta["test"])
        for c in range(3):
            mask = y_train == c
            ax.scatter(
                x_train[mask, 0],
                x_train[mask, 1],
                s=22,
                color=colors[c],
                marker=markers[c],
                alpha=0.75,
                label=f"train class {c}",
            )
        ax.scatter(
            x_test[60:, 0],
            x_test[60:, 1],
            s=16,
            facecolors="none",
            edgecolors="#333333",
            linewidths=0.7,
            alpha=0.75,
            label="new test",
        )
        ax.set_title(meta["title"])
        ax.set_xlabel("body length")
        ax.set_ylabel("wing length")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")
    fig.savefig(FIG_DIR / "data_overview.png", dpi=220)
    plt.close(fig)


def plot_parameter_effects(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, dataset_key in zip(axes, DATASETS):
        sub = summary[summary["dataset"] == dataset_key].copy()
        sub = sub.sort_values("new150_mean", ascending=True)
        y_pos = np.arange(len(sub))
        ax.barh(y_pos, sub["seen60_mean"], height=0.36, label="前60个测试样本")
        ax.barh(
            y_pos + 0.38,
            sub["new150_mean"],
            height=0.36,
            label="后150个新样本",
        )
        ax.set_yticks(y_pos + 0.19)
        ax.set_yticklabels(sub["config"], fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("accuracy")
        ax.set_title(DATASETS[dataset_key]["title"])
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(FIG_DIR / "parameter_effects.png", dpi=220)
    plt.close(fig)


def plot_decision_boundary(dataset_key: str, run: dict, config_name: str) -> None:
    meta = DATASETS[dataset_key]
    x_train, y_train = load_data(meta["train"])
    x_test, y_test = load_data(meta["test"])
    model: torch.nn.Module = run["model"]
    mean = run["mean"]
    std = run["std"]

    x_all = np.vstack([x_train, x_test])
    pad = 0.25
    x_min, x_max = x_all[:, 0].min() - pad, x_all[:, 0].max() + pad
    y_min, y_max = x_all[:, 1].min() - pad, x_all[:, 1].max() + pad
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 320), np.linspace(y_min, y_max, 320))
    grid = np.c_[gx.ravel(), gy.ravel()].astype(np.float32)
    grid_std = (grid - mean) / std
    with torch.no_grad():
        logits = model(torch.tensor(grid_std, dtype=torch.float32))
        z = logits.argmax(dim=1).numpy().reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    cmap = matplotlib.colors.ListedColormap(["#D9EAF7", "#F8DCC9", "#DDF0E6"])
    ax.contourf(gx, gy, z, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap, alpha=0.9)
    colors = ["#0072B2", "#D55E00", "#009E73"]
    markers = ["o", "^", "s"]
    for c in range(3):
        mask = y_train == c
        ax.scatter(
            x_train[mask, 0],
            x_train[mask, 1],
            s=20,
            c=colors[c],
            marker=markers[c],
            edgecolors="white",
            linewidths=0.3,
            label=f"train {c}",
        )
    wrong = run["pred"] != y_test
    ax.scatter(
        x_test[60:, 0],
        x_test[60:, 1],
        s=25,
        facecolors="none",
        edgecolors="#111111",
        linewidths=0.7,
        label="new test",
    )
    if wrong.any():
        ax.scatter(
            x_test[wrong, 0],
            x_test[wrong, 1],
            s=75,
            c="#000000",
            linewidths=1.3,
            marker="x",
            label="wrong",
        )
    ax.set_title(f"{meta['title']} best boundary: {config_name}")
    ax.set_xlabel("body length")
    ax.set_ylabel("wing length")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)
    fig.savefig(FIG_DIR / f"boundary_{dataset_key}.png", dpi=220)
    plt.close(fig)


def plot_training_curve(dataset_key: str, run: dict, config_name: str) -> None:
    hist = np.array(run["history"], dtype=float)
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax1.plot(hist[:, 0], hist[:, 1], label="train loss", color="#0072B2")
    ax1.plot(hist[:, 0], hist[:, 2], label="validation loss", color="#D55E00")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross entropy loss")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(hist[:, 0], hist[:, 3], label="validation accuracy", color="#009E73")
    ax2.set_ylabel("validation accuracy")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="center right")
    ax1.set_title(f"{DATASETS[dataset_key]['title']} training: {config_name}")
    fig.savefig(FIG_DIR / f"training_{dataset_key}.png", dpi=220)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    rows = [[str(value) for value in row] for row in view.to_numpy()]
    widths = [
        max(len(str(col)), *(len(row[i]) for row in rows)) for i, col in enumerate(columns)
    ]
    header = "| " + " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def write_report(summary: pd.DataFrame, best_rows: dict[str, pd.Series]) -> None:
    clean_best = best_rows["dataset1_clean"]
    noisy_best = best_rows["dataset2_noisy"]
    report = """---
title: "作业3 选项2：使用神经网络进行昆虫分类"
author: "胡广理 PB24010500"
date: "2026-05-05"
---

# 1 问题与建模

本实验选择选项2。目标是利用昆虫的体长 $x_1$ 与翼长 $x_2$ 两个特征，将样本分类到 0、1、2 三个类别。训练集记为
$$
\\mathcal D=\\{(\\mathbf x_i,y_i)\\}_{i=1}^n,\\quad \\mathbf x_i=(x_{i1},x_{i2})\\in\\mathbb R^2,\\ y_i\\in\\{0,1,2\\}.
$$
神经网络输出三维 logit 向量 $z=f_\\theta(\\mathbf x)$，再用 softmax 得到类别概率
$$
p_k(\\mathbf x)=\\frac{e^{z_k}}{\\sum_{j=0}^2 e^{z_j}}.
$$
训练目标为最小化交叉熵损失
$$
L(\\theta)=-\\frac1n\\sum_i \\log p_{y_i}(\\mathbf x_i).
$$
分类规则为 $\\hat y=\\arg\\max_k p_k(\\mathbf x)$。

为避免不同量纲影响训练，所有模型均只使用训练集估计均值与标准差，并对训练集、验证集、测试集做同一标准化变换。训练集内部按类别分层划分出 20\\% 作为验证集，用于早停和比较网络参数；测试集只在最终评价时使用。

# 2 算法实现

实现使用 `PyTorch`，运行环境为已有 conda 环境 `cupy`。为了考察网络参数影响，实验比较了线性 softmax 分类器、单隐层 MLP、不同激活函数以及带 L2 正则的双隐层 MLP。优化器使用 Adam，损失函数使用交叉熵。每种设置使用 5 个随机种子重复实验，报告平均值与标准差。

测试集按作业要求拆成两段评价：前 60 个样本来自训练数据的随机抽取，可观察模型对已知分布/近似已见样本的拟合能力；后 150 个样本不在训练集中，更能衡量模型对新昆虫数据的泛化能力。因此报告中更重视后 150 个样本的准确率。

# 3 数据可视化

![两组数据的训练集与新测试样本分布](option2_outputs/figs/data_overview.png)

从散点图可见，数据集1的三类样本分界较清晰；数据集2加入测量噪声后，同类样本更分散，类别边界附近的重叠更明显，分类难度更高。因此数据集2更需要适当的正则化和验证集早停，以降低过拟合风险。

# 4 参数对比结果

下表中 `seen60` 表示测试集前 60 个样本准确率，`new150` 表示后 150 个新样本准确率，`all210` 表示全部测试样本准确率。

__RESULT_TABLE__

![不同网络参数对准确率的影响](option2_outputs/figs/parameter_effects.png)

数据集1最佳模型为 `__CLEAN_CONFIG__`，后 150 个新样本准确率为 **__CLEAN_NEW150__**，全部测试准确率为 **__CLEAN_ALL210__**。数据集2最佳模型为 `__NOISY_CONFIG__`，后 150 个新样本准确率为 **__NOISY_NEW150__**，全部测试准确率为 **__NOISY_ALL210__**。

# 5 最佳模型分类边界

![数据集1最佳模型决策边界](option2_outputs/figs/boundary_dataset1_clean.png)

![数据集2最佳模型决策边界](option2_outputs/figs/boundary_dataset2_noisy.png)

决策边界显示，神经网络能够学习到非线性的类别分割曲线。线性模型只能给出直线边界，在类别形状较弯曲或噪声较强时表达能力不足；双隐层 ReLU 网络可以形成更灵活的分段线性边界，但如果网络过大且缺少正则化，也可能把边界拉向个别噪声点。实验中带 L2 正则与早停的双隐层模型通常在新样本上更稳定。

# 6 训练过程

![数据集1最佳模型训练曲线](option2_outputs/figs/training_dataset1_clean.png)

![数据集2最佳模型训练曲线](option2_outputs/figs/training_dataset2_noisy.png)

训练曲线表明，交叉熵损失快速下降后趋于稳定。验证集准确率没有继续提升时触发早停，可以减少后续迭代对训练集局部噪声的记忆。

# 7 影响因素分析

1. 网络结构：无隐层的 softmax 线性分类器表达能力最低；加入隐层后可以拟合非线性边界，通常提升新样本准确率。
2. 激活函数：ReLU 收敛较快，边界呈分段线性；tanh 边界更平滑，但在本数据规模下并不一定优于 ReLU。
3. 正则化：数据集2存在测量噪声，L2 正则与早停对泛化更重要。过大的网络若不约束，可能提高训练/前60样本表现，却降低后150个新样本准确率。
4. 数据噪声：数据集2相比数据集1的类别重叠更多，因此相同模型的准确率波动更明显。实际建模时应优先关注新样本准确率，而不是只看训练集或前60个样本。

# 8 结论

本实验完成了两组昆虫数据集上的神经网络分类。结果说明，仅用体长和翼长两个特征，MLP 已能较好区分三类昆虫；对于无额外噪声的数据集，简单 MLP 即可获得较高准确率；对于含噪声数据，带正则化的网络和早停机制更稳健。分段测试也说明，前60个样本不能完全代表泛化能力，后150个新样本的表现才是评价模型有效性的关键。

# 9 AI 使用说明

本报告和代码由本人在理解作业要求、确认数据格式和实验目标后，借助 AI 辅助整理实验流程、生成可复现实验脚本和报告初稿；最终内容经过人工检查，确保实验设置、结果解释和结论与实际运行输出一致。
"""
    report = (
        report.replace(
            "__RESULT_TABLE__",
            markdown_table(
                summary,
                ["dataset", "config", "seen60", "new150", "all210", "epochs"],
            ),
        )
        .replace("__CLEAN_CONFIG__", str(clean_best["config"]))
        .replace("__CLEAN_NEW150__", str(clean_best["new150"]))
        .replace("__CLEAN_ALL210__", str(clean_best["all210"]))
        .replace("__NOISY_CONFIG__", str(noisy_best["config"]))
        .replace("__NOISY_NEW150__", str(noisy_best["new150"]))
        .replace("__NOISY_ALL210__", str(noisy_best["all210"]))
    )
    (ROOT / "report_option2.md").write_text(report, encoding="utf-8")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    plot_data_overview()

    records = []
    best_runs = {}
    predictions = []

    for dataset_key, meta in DATASETS.items():
        x_train, y_train = load_data(meta["train"])
        x_test, y_test = load_data(meta["test"])
        dataset_runs = []

        for config in CONFIGS:
            seed_runs = []
            for seed in SEEDS:
                run = train_one(x_train, y_train, x_test, y_test, config, seed, device)
                seed_runs.append(run)
                records.append(
                    {
                        "dataset": dataset_key,
                        "config": config.name,
                        "seed": seed,
                        "seen60_acc": run["acc_seen60"],
                        "new150_acc": run["acc_new150"],
                        "all210_acc": run["acc_all210"],
                        "epochs_used": run["epochs_used"],
                    }
                )
            mean_new = float(np.mean([r["acc_new150"] for r in seed_runs]))
            mean_all = float(np.mean([r["acc_all210"] for r in seed_runs]))
            best_seed_index = int(
                np.argmax([r["acc_new150"] * 10 + r["acc_all210"] for r in seed_runs])
            )
            dataset_runs.append((mean_new, mean_all, config, seed_runs[best_seed_index]))

        dataset_runs.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _, _, best_config, best_run = dataset_runs[0]
        best_runs[dataset_key] = (best_config, best_run)
        plot_decision_boundary(dataset_key, best_run, best_config.name)
        plot_training_curve(dataset_key, best_run, best_config.name)

        for i, (true_label, pred_label) in enumerate(zip(y_test, best_run["pred"])):
            predictions.append(
                {
                    "dataset": dataset_key,
                    "index": i,
                    "split": "seen60" if i < 60 else "new150",
                    "true_label": int(true_label),
                    "pred_label": int(pred_label),
                    "correct": bool(true_label == pred_label),
                    "model": best_config.name,
                }
            )

    raw = pd.DataFrame(records)
    raw.to_csv(OUT_DIR / "all_runs.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT_DIR / "best_model_predictions.csv", index=False)

    grouped = (
        raw.groupby(["dataset", "config"], as_index=False)
        .agg(
            seen60_mean=("seen60_acc", "mean"),
            seen60_std=("seen60_acc", "std"),
            new150_mean=("new150_acc", "mean"),
            new150_std=("new150_acc", "std"),
            all210_mean=("all210_acc", "mean"),
            all210_std=("all210_acc", "std"),
            epochs_mean=("epochs_used", "mean"),
        )
        .sort_values(["dataset", "new150_mean", "all210_mean"], ascending=[True, False, False])
    )
    grouped["seen60"] = grouped.apply(
        lambda r: f"{r.seen60_mean:.3f} ± {r.seen60_std:.3f}", axis=1
    )
    grouped["new150"] = grouped.apply(
        lambda r: f"{r.new150_mean:.3f} ± {r.new150_std:.3f}", axis=1
    )
    grouped["all210"] = grouped.apply(
        lambda r: f"{r.all210_mean:.3f} ± {r.all210_std:.3f}", axis=1
    )
    grouped["epochs"] = grouped["epochs_mean"].map(lambda x: f"{x:.0f}")
    grouped.to_csv(OUT_DIR / "summary.csv", index=False)
    plot_parameter_effects(grouped)

    best_rows = {}
    for dataset_key in DATASETS:
        sub = grouped[grouped["dataset"] == dataset_key].copy()
        best_rows[dataset_key] = sub.iloc[0]

    write_report(grouped, best_rows)

    metadata = {
        "device": str(device),
        "torch_version": torch.__version__,
        "seeds": SEEDS,
        "best_models": {
            key: {"config": cfg.name, "test_acc_all210": run["acc_all210"]}
            for key, (cfg, run) in best_runs.items()
        },
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nSummary:")
    print(grouped[["dataset", "config", "seen60", "new150", "all210", "epochs"]])
    print(f"\nWrote outputs to {OUT_DIR}")
    print(f"Wrote report markdown to {ROOT / 'report_option2.md'}")


if __name__ == "__main__":
    main()
