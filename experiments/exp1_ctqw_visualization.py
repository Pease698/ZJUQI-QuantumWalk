#!/usr/bin/env python3
"""实验一：CTQW 概率分布可视化（理论 §13.2）。

目的：
  验证 CTQW 是否能够在团或密集子图附近形成概率集中。

实验方法：
  1. 构造含 planted clique / dense subgraph 的测试图
  2. 运行 CTQW 计算所有节点的 P_v(t)
  3. 比较团内节点与团外节点的平均概率
  4. 可视化节点概率分布图

评价指标：
  Ratio = Mean(P_v, v∈S_target) / Mean(P_v, v∉S_target)
  若 Ratio > 1，说明目标区域概率更高。

当前状态：
  CTQW 计算使用占位实现（QuantumScorer），输出图形结构和 Ratio 计算框架已就绪。
  真实 CTQW 接入后，仅需将 QuantumScorer 替换为真实实现，脚本其余部分无需修改。

用法:
  python3 experiments/exp1_ctqw_visualization.py
  python3 experiments/exp1_ctqw_visualization.py --task maximum_clique
  python3 experiments/exp1_ctqw_visualization.py --n 50 --k 8 --p 0.2
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# 确保可以从项目根目录导入 src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_utils import load_instance, GraphInstance
from src.hamiltonian import construct_hamiltonian
from src.initial_state import build_initial_state
from src.metrics import compute_ratio


# ============================================================
# 配置
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp1_ctqw_visualization"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 可视化使用的颜色和样式
COLOR_TARGET = "#d62728"       # 目标节点：红色
COLOR_BACKGROUND = "#1f77b4"   # 背景节点：蓝色
COLOR_EDGE_TARGET = "#ff7f0e"  # 目标区域内部边：橙色
COLOR_EDGE_BG = "#e0e0e0"      # 背景边：浅灰
NODE_SIZE_BASE = 300
NODE_SIZE_SCALE = 800


# ============================================================
# CTQW 概率计算
# ============================================================

def compute_ctqw_probabilities(instance: GraphInstance,
                               S: set[int] | None = None,
                               t: float = 1.0,
                               lam: float = 0.0,
                               init_method: str = "max_degree") -> np.ndarray:
    """计算 CTQW 节点概率分布 P_v(t)。

    通过 QuantumScorer 调用 scipy.linalg.expm 完成 |ψ(t)⟩ = e^{-iHt}|ψ₀⟩
    的矩阵指数计算，最终返回 P_v = |ψ_v(t)|²。

    参数:
        instance: 图实例。
        S: 种子集合（None 表示空集，此时按 init_method 选择初态）。
        t: 演化时间。
        lam: 扰动强度。
        init_method: 初始态构造方式。

    返回:
        长度为 n 的数组，P[v] = 节点 v 的概率，满足 Σ P = 1。
    """
    n = instance.num_nodes
    S = S or set()

    # 通过 QuantumScorer 计算（内部实现真实 CTQW）
    from src.scoring import QuantumScorer
    qs = QuantumScorer(t=t, lam=lam, init_method=init_method, seed=42)
    candidates = set(range(n))
    scores = qs.score_all(candidates, S, instance)
    probs = np.zeros(n, dtype=np.float64)
    for v, p in scores.items():
        probs[v] = p
    return probs


# ============================================================
# 可视化
# ============================================================

def plot_probability_distribution(instance: GraphInstance,
                                   probs: np.ndarray,
                                   title: str,
                                   save_path: str | None = None):
    """绘制节点概率分布图。

    节点颜色深度和大小均表示 CTQW 概率，目标区域以红框高亮。

    参数:
        instance: 图实例。
        probs: 长度为 n 的概率数组。
        title: 图表标题。
        save_path: 保存路径（None 则显示）。
    """
    G = instance.nx_graph
    answer_set = instance.answer_set
    n = instance.num_nodes

    # 使用 spring_layout 计算节点位置
    pos = nx.spring_layout(G, seed=42, iterations=50)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ---- 左图：CTQW 概率分布 ----
    ax = axes[0]
    node_colors = []
    node_sizes = []
    edge_colors = []
    edge_widths = []

    for u, v in G.edges():
        if u in answer_set and v in answer_set:
            edge_colors.append(COLOR_EDGE_TARGET)
            edge_widths.append(2.0)
        else:
            edge_colors.append(COLOR_EDGE_BG)
            edge_widths.append(0.5)

    for node in G.nodes():
        if node in answer_set:
            node_colors.append(COLOR_TARGET)
        else:
            node_colors.append(COLOR_BACKGROUND)
        node_sizes.append(NODE_SIZE_BASE + NODE_SIZE_SCALE * probs[node])

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           width=edge_widths, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85,
                           edgecolors="black", linewidths=0.5)

    # 为答案节点添加红色边框高亮
    if answer_set:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=list(answer_set),
                               node_color="none",
                               node_size=[NODE_SIZE_BASE + NODE_SIZE_SCALE * probs[v]
                                          for v in answer_set],
                               edgecolors=COLOR_TARGET,
                               linewidths=2.5, alpha=1.0)

    if n <= 50:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

    ax.set_title(f"CTQW Probability Distribution\n{title}", fontsize=12, fontweight="bold")
    ax.axis("off")

    # ---- 右图：概率对比（目标 vs 背景） ----
    ax = axes[1]
    target_probs = [probs[v] for v in answer_set]
    bg_probs = [probs[v] for v in range(n) if v not in answer_set]

    # 概率分布直方图
    bins = np.linspace(0, max(probs.max(), 0.05), 30)
    ax.hist(bg_probs, bins=bins, alpha=0.6, color=COLOR_BACKGROUND,
            label=f"Background (n={len(bg_probs)})")
    ax.hist(target_probs, bins=bins, alpha=0.8, color=COLOR_TARGET,
            label=f"Target (n={len(target_probs)})")

    # 标注均值线
    if target_probs:
        target_mean = np.mean(target_probs)
        ax.axvline(target_mean, color=COLOR_TARGET, linestyle="--", linewidth=2,
                   label=f"Target mean = {target_mean:.4f}")
    if bg_probs:
        bg_mean = np.mean(bg_probs)
        ax.axvline(bg_mean, color=COLOR_BACKGROUND, linestyle="--", linewidth=2,
                   label=f"Background mean = {bg_mean:.4f}")

    ratio = compute_ratio(target_probs, bg_probs)
    ax.set_xlabel("Probability P_v(t)")
    ax.set_ylabel("Node Count")
    ax.set_title(f"Probability Distribution\nRatio = {ratio:.3f}", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"CTQW Probability Visualization\n{instance.sample_id}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  图表已保存至: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_classical_vs_quantum(instance: GraphInstance,
                               ctqw_probs: np.ndarray,
                               save_path: str | None = None):
    """对比经典度数分布与 CTQW 概率分布。

    参数:
        instance: 图实例。
        ctqw_probs: CTQW 概率数组。
        save_path: 保存路径。
    """
    n = instance.num_nodes
    answer_set = instance.answer_set
    adjacency = instance.adjacency

    degrees = adjacency.sum(axis=1)
    deg_norm = degrees / degrees.sum() if degrees.sum() > 0 else degrees

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- 左图：散点图 —— 度数 vs CTQW 概率 ----
    ax = axes[0]
    colors = [COLOR_TARGET if i in answer_set else COLOR_BACKGROUND
              for i in range(n)]
    sizes = [80 if i in answer_set else 30 for i in range(n)]

    ax.scatter(deg_norm, ctqw_probs, c=colors, s=sizes, alpha=0.7,
               edgecolors="black", linewidths=0.3)

    # 标注对角线 (y=x)
    lim_max = max(deg_norm.max(), ctqw_probs.max()) * 1.1
    ax.plot([0, lim_max], [0, lim_max], "k--", alpha=0.3, linewidth=1)

    ax.set_xlabel("Normalized Degree")
    ax.set_ylabel("CTQW Probability P_v(t)")
    ax.set_title("Degree vs CTQW Probability")
    ax.grid(True, alpha=0.3)

    # ---- 右图：排名对比 ----
    ax = axes[1]
    x = np.arange(n)
    rank_deg = np.argsort(np.argsort(-deg_norm))  # 度数排名
    rank_ctqw = np.argsort(np.argsort(-ctqw_probs))  # CTQW 排名

    for i in range(n):
        color = COLOR_TARGET if i in answer_set else COLOR_BACKGROUND
        alpha_val = 0.9 if i in answer_set else 0.3
        ax.plot([0, 1], [rank_deg[i], rank_ctqw[i]],
                color=color, alpha=alpha_val, linewidth=0.8)
        ax.scatter([0], [rank_deg[i]], color=color, s=15, alpha=alpha_val)
        ax.scatter([1], [rank_ctqw[i]], color=color, s=15, alpha=alpha_val)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Degree Rank", "CTQW Rank"])
    ax.set_ylabel("Node Rank (0 = highest)")
    ax.set_title("Rank Comparison: Degree vs CTQW")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Classical vs Quantum Distribution\n{instance.sample_id}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  对比图已保存至: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ============================================================
# 主实验逻辑
# ============================================================

def run_visualization_experiment(instance: GraphInstance,
                                  t: float = 1.0,
                                  lam: float = 0.0,
                                  init_method: str = "max_degree"):
    """对单个实例运行完整的可视化实验。

    返回:
        dict: 包含 ratio、target_mean、background_mean 等指标的字典。
    """
    sample_id = instance.sample_id
    n = instance.num_nodes
    answer_set = instance.answer_set

    print(f"\n{'=' * 60}")
    print(f"实例: {sample_id}")
    print(f"  节点: {n}, 答案大小: {len(answer_set)}")
    print(f"  t={t}, λ={lam}, init={init_method}")
    print("-" * 60)

    # 1. 计算 CTQW 概率
    print("  计算 CTQW 概率分布...")
    probs = compute_ctqw_probabilities(instance, S=set(), t=t, lam=lam,
                                       init_method=init_method)

    # 2. 计算 Ratio
    target_probs = [float(probs[v]) for v in answer_set]
    bg_probs = [float(probs[v]) for v in range(n) if v not in answer_set]
    ratio = compute_ratio(target_probs, bg_probs)

    print(f"  目标节点平均概率: {np.mean(target_probs):.6f}")
    print(f"  背景节点平均概率: {np.mean(bg_probs):.6f}")
    print(f"  Ratio = {ratio:.4f}")
    print(f"  Ratio > 1: {'是 ✓ (目标区域概率集中)' if ratio > 1 else '否 ✗ (CTQW 未在目标区域形成概率集中)'}")

    # 3. 绘制概率分布图
    safe_id = sample_id.replace("/", "_")
    prob_path = RESULTS_DIR / f"{safe_id}_t{t}_lam{lam}_prob.png"
    plot_probability_distribution(instance, probs,
                                   title=f"t={t}, λ={lam}",
                                   save_path=str(prob_path))

    # 4. 绘制经典 vs 量子对比图
    cmp_path = RESULTS_DIR / f"{safe_id}_t{t}_lam{lam}_compare.png"
    plot_classical_vs_quantum(instance, probs, save_path=str(cmp_path))

    return {
        "sample_id": sample_id,
        "n": n,
        "answer_size": len(answer_set),
        "t": t,
        "lam": lam,
        "init_method": init_method,
        "ratio": ratio,
        "target_mean": float(np.mean(target_probs)) if target_probs else 0.0,
        "background_mean": float(np.mean(bg_probs)) if bg_probs else 0.0,
    }


def find_instance(task_type: str, n: int | None = None,
                  p: float | None = None, k: int | None = None) -> str | None:
    """根据条件查找一个测试实例。

    返回:
        匹配的 JSON 文件路径，或 None。
    """
    data_dir = Path(__file__).resolve().parent.parent / "datasets" / "data" \
               / "artificial" / task_type

    if not data_dir.is_dir():
        return None

    patterns = []
    if n is not None and p is not None and k is not None:
        # 精确匹配
        if task_type == "maximum_clique":
            patterns.append(f"*_n{n}_*p*_k{k}/*.json")
        else:
            patterns.append(f"*_n{n}_*p*_k{k}_*/*.json")

    # 宽松匹配
    if n is not None:
        patterns.append(f"*_n{n}_*/*.json")

    # 遍历所有匹配模式
    for pattern in patterns:
        for fpath in sorted(data_dir.glob(pattern)):
            return str(fpath)

    # 回退：返回第一个找到的 JSON
    for fpath in sorted(data_dir.rglob("*.json")):
        return str(fpath)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="实验一：CTQW 概率分布可视化（理论 §13.2）")
    parser.add_argument("--task", choices=["maximum_clique", "densest_subgraph"],
                        default="maximum_clique", help="任务类型")
    parser.add_argument("--instance", type=str, default=None,
                        help="指定 JSON 文件路径（覆盖 n/p/k）")
    parser.add_argument("--n", type=int, default=30, help="节点数")
    parser.add_argument("--p", type=float, default=None, help="背景边概率")
    parser.add_argument("--k", type=int, default=None, help="目标子图大小")
    parser.add_argument("--t", type=float, default=1.0, help="演化时间")
    parser.add_argument("--lam", type=float, default=0.0, help="扰动强度")
    parser.add_argument("--init", choices=["uniform", "max_degree", "random"],
                        default="max_degree", help="初态初始化方式")
    parser.add_argument("--save", action="store_true", default=True,
                        help="保存图表（默认启用）")
    args = parser.parse_args()

    # 查找测试实例
    if args.instance:
        instance_path = args.instance
    else:
        instance_path = find_instance(args.task, n=args.n, p=args.p, k=args.k)

    if not instance_path or not os.path.isfile(instance_path):
        print(f"错误: 未找到测试实例。请先运行生成脚本。")
        print(f"  task={args.task}, n={args.n}, p={args.p}, k={args.k}")
        return

    print(f"实验一：CTQW 概率分布可视化")
    print(f"  数据: {instance_path}")
    print(f"  参数: t={args.t}, λ={args.lam}, init={args.init}")

    instance = load_instance(instance_path)
    result = run_visualization_experiment(
        instance, t=args.t, lam=args.lam, init_method=args.init)

    # 汇总
    print(f"\n{'=' * 60}")
    print("实验一完成。结果汇总:")
    print(f"  Ratio = {result['ratio']:.4f}")
    print(f"  目标平均概率 = {result['target_mean']:.6f}")
    print(f"  背景平均概率 = {result['background_mean']:.6f}")
    print(f"  图表目录: {RESULTS_DIR}")

    # 保存数值结果
    import json
    result_path = RESULTS_DIR / f"{result['sample_id']}_t{args.t}_lam{args.lam}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  数值结果: {result_path}")


if __name__ == "__main__":
    main()
