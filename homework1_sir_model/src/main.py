"""
主程序入口 - 传染病模型
"""

import numpy as np
import matplotlib.pyplot as plt
from sir_model import SIRModel, SIRModelWithDemographics
from seir_model import SEIRModel


def plot_sir_results(solution: dict, title: str = "SIR Model"):
    """绘制SIR结果"""
    plt.figure(figsize=(10, 6))
    plt.plot(solution['t'], solution['S'], label='Susceptible')
    plt.plot(solution['t'], solution['I'], label='Infected')
    plt.plot(solution['t'], solution['R'], label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_comparison(sir_solution: dict, seir_solution: dict):
    """对比SIR和SEIR模型"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # SIR
    axes[0].plot(sir_solution['t'], sir_solution['I'], 'r-', label='Infected (SIR)')
    axes[0].set_xlabel('Time (days)')
    axes[0].set_ylabel('Infected Population')
    axes[0].set_title('SIR Model')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SEIR
    axes[1].plot(seir_solution['t'], seir_solution['E'], 'orange', label='Exposed')
    axes[1].plot(seir_solution['t'], seir_solution['I'], 'r-', label='Infected (SEIR)')
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel('Population')
    axes[1].set_title('SEIR Model')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def problem_1_1():
    """子问题1.1: 基础SIR模型"""
    print("=" * 50)
    print("子问题1.1: 基础SIR模型")
    print("=" * 50)
    
    # 参数设置
    beta = 0.5    # 传染率
    gamma = 0.1  # 康复率
    N = 1000000  # 总人口
    
    # 创建模型
    model = SIRModel(beta, gamma, N)
    print(f"基本再生数 R0 = {model.R0:.2f}")
    
    # 初始条件
    S0 = N - 100
    I0 = 100
    R0 = 0
    
    # 求解
    solution = model.solve(S0, I0, R0, t_max=100)
    
    # 可视化
    plot_sir_results(solution, f"SIR Model (R0={model.R0:.2f})")
    
    # 分析峰值
    peak_time, peak_value = model.get_peak_infection(solution)
    print(f"感染峰值: {peak_value:.0f} at t = {peak_time:.1f} days")
    
    return solution


def problem_1_2():
    """子问题1.2: 带人口动态的SIR模型"""
    print("=" * 50)
    print("子问题1.2: 带人口动态的SIR模型")
    print("=" * 50)
    
    beta = 0.3
    gamma = 0.1
    N = 100000
    birth_rate = 50  # 假设每天新增50人
    death_rate = 0.00005  # 死亡率
    
    model = SIRModelWithDemographics(beta, gamma, N, birth_rate, death_rate)
    print(f"考虑人口动态后的有效再生数需重新计算")
    
    S0 = N - 100
    I0 = 100
    R0 = 0
    
    solution = model.solve(S0, I0, R0, t_max=200)
    plot_sir_results(solution, "SIR with Demographics")
    
    # 计算平衡点
    eq = model.get_equilibrium()
    print(f"平衡点: S* = {eq.get('S', 'N/A')}, I* = {eq.get('I', 'N/A')}")


def problem_1_3():
    """子问题1.3: SEIR模型"""
    print("=" * 50)
    print("子问题1.3: SEIR模型")
    print("=" * 50)
    
    beta = 0.5
    sigma = 1/5   # 潜伏期5天
    gamma = 0.1
    N = 1000000
    
    model = SEIRModel(beta, sigma, gamma, N)
    print(f"SEIR模型 R0 = {model.R0:.2f}")
    
    S0 = N - 100
    E0 = 50
    I0 = 50
    R0 = 0
    
    solution = model.solve(S0, E0, I0, R0, t_max=100)
    
    # 对比SIR
    from sir_model import SIRModel
    sir_model = SIRModel(beta, gamma, N)
    sir_solution = sir_model.solve(S0, I0, R0, t_max=100)
    
    plot_comparison(sir_solution, solution)


if __name__ == "__main__":
    # 运行所有子问题
    solution_1_1 = problem_1_1()
    # problem_1_2()  # 解开注释运行
    # problem_1_3()  # 解开注释运行
