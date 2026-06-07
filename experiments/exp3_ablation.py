#!/usr/bin/env python3
"""实验三：消融实验（理论 §13.4）。

目的：
  验证每个模块是否真的有贡献。

消融设置：
  ┌────────┬──────────┬──────────┬──────────┬──────────────────┐
  │ 实验组  │ 量子 Q   │ 经典 R   │ 种子 λ   │ 目的              │
  ├────────┼──────────┼──────────┼──────────┼──────────────────┤
  │ A      │ 否       │ 是       │ 否       │ 纯经典基线        │
  │ B      │ 是       │ 否       │ 否       │ 原始 CTQW 概率    │
  │ C      │ 是       │ 否       │ 是       │ 验证种子扰动      │
  │ D      │ 是       │ 是       │ 是       │ 完整算法          │
  └────────┴──────────┴──────────┴──────────┴──────────────────┘

如果 D 组表现最好，说明"量子概率 + 种子扰动 + 经典约束"三者结合有效。

当前状态：
  量子评分使用占位实现（正常 CTQW 接入后替换），框架逻辑已完整。
  经典部分的所有组合均可正常运行。

用法:
  python3 experiments/exp3_ablation.py
  python3 experiments/exp3_ablation.py --task maximum_clique --small
"""

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_utils import load_instance, load_instances_from_dir
from src.candidate_set import CliqueCandidateSet, DenseCandidateSet
from src.scoring import (ClassicalCliqueScorer, ClassicalDenseScorer,
                          ClassicalDegreeScorer, HybridScorer, QuantumScorer)
from src.algorithms.classical_greedy import ClassicalGreedy
from src.algorithms.quantum_greedy import QuantumGuidedGreedy
from src.metrics import mean_std, aggregate_results
from src.config import get_data_dirs

# ============================================================
# 配置
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp3_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPEAT_RUNS = 4
# 消融实验组的颜色
ABLATION_COLORS = {
    "A_ClassicalOnly": "#1f77b4",
    "B_QuantumNoLambda": "#ff7f0e",
    "C_QuantumWithLambda": "#2ca02c",
    "D_FullHybrid": "#d62728",
}


# ============================================================
# 消融实验构造
# ============================================================

def build_ablation_algorithms(task_type: str, seed: int = 0) -> dict:
    """构造四组消融实验的算法对象。

    ┌─────┬──────────┬──────────┬──────────┬──────────────────────┐
    │ 组   │ Q        │ R        │ λ        │ 说明                  │
    ├─────┼──────────┼──────────┼──────────┼──────────────────────┤
    │ A   │ 否       │ 是       │ 否 (0)   │ ClassicalGreedy       │
    │ B   │ 是(占位) │ 否       │ 否 (0)   │ QuantumGuided α=1,λ=0 │
    │ C   │ 是(占位) │ 否       │ 是 (0.5) │ QuantumGuided α=1,λ>0 │
    │ D   │ 是(占位) │ 是       │ 是 (0.5) │ QuantumGuided α=0.5   │
    └─────┴──────────┴──────────┴──────────┴──────────────────────┘
    """
    if task_type == "maximum_clique":
        builder = CliqueCandidateSet()
        classical_scorer = ClassicalCliqueScorer()
    else:
        builder = DenseCandidateSet()
        classical_scorer = ClassicalDenseScorer()

    algorithms = {}

    # A 组：纯经典基线 (Q=否, R=是, λ=否)
    algorithms["A_ClassicalOnly"] = ClassicalGreedy(
        builder, classical_scorer, name="A_ClassicalOnly")

    # B 组：纯量子，无种子扰动 (Q=是, R=否, λ=0)
    # 使用 QuantumGuidedGreedy with α=1.0, λ=0
    algorithms["B_QuantumNoLambda"] = QuantumGuidedGreedy(
        builder, t=1.0, lam=0.0, alpha=1.0, seed=seed,
        name="B_QuantumNoLambda")

    # C 组：纯量子，有种子扰动 (Q=是, R=否, λ>0)
    algorithms["C_QuantumWithLambda"] = QuantumGuidedGreedy(
        builder, t=1.0, lam=0.5, alpha=1.0, seed=seed,
        name="C_QuantumWithLambda")

    # D 组：完整算法 (Q=是, R=是, λ>0)
    algorithms["D_FullHybrid"] = QuantumGuidedGreedy(
        builder, t=1.0, lam=0.5, alpha=0.5, seed=seed,
        name="D_FullHybrid")

    return algorithms


# ============================================================
# 实验运行
# ============================================================

def run_ablation_on_instance(instance, algorithms: dict,
                              repeat: int = REPEAT_RUNS) -> list[dict]:
    """对单个实例运行所有消融实验组。"""
    rows = []
    for algo_name, algo_template in algorithms.items():
        for run_id in range(repeat):
            algo = _rebuild_ablation_algo(algo_template, run_id)
            result = algo.solve(instance)
            row = result.to_dict()
            row["run_id"] = run_id
            row["ablation_group"] = algo_name[0]  # A/B/C/D
            rows.append(row)
    return rows


def _rebuild_ablation_algo(template, seed: int):
    """根据模板算法创建带新 seed 的副本。"""
    from src.candidate_set import CliqueCandidateSet, DenseCandidateSet
    from src.scoring import ClassicalCliqueScorer, ClassicalDenseScorer

    name = template.name
    builder = template.candidate_builder

    if isinstance(template, ClassicalGreedy):
        return ClassicalGreedy(builder, template.scorer,
                               name=f"{name}(run={seed})")

    elif isinstance(template, QuantumGuidedGreedy):
        return QuantumGuidedGreedy(
            builder, t=template.t, lam=template.lam,
            alpha=template.alpha,
            init_method=template.init_method,
            seed=seed,
            name=f"{name}(run={seed})")
    return template


# ============================================================
# 批量实验与可视化
# ============================================================

def run_ablation_experiment(task_type: str, small: bool = True) -> pd.DataFrame:
    """批量运行消融实验。"""
    dirs = get_data_dirs(task_type)
    if not dirs:
        print(f"未找到 {task_type} 的数据目录")
        return pd.DataFrame()

    if small:
        dirs = [d for d in dirs if _extract_n(d.name) <= 50]
        print(f"小规模模式：筛选 {len(dirs)} 个目录 (n ≤ 50)")

    all_rows = []
    total_dirs = len(dirs)
    t_start = time.perf_counter()

    print(f"\n实验三：消融实验")
    print(f"  任务类型: {task_type}")
    print(f"  目标: 验证 Q / R / λ 各自对性能的贡献")
    print(f"  数据目录: {total_dirs} 个")
    print("-" * 50)

    for dir_idx, data_dir in enumerate(dirs):
        instances = load_instances_from_dir(data_dir)
        algo_templates = build_ablation_algorithms(task_type, seed=0)

        for inst in instances:
            rows = run_ablation_on_instance(inst, algo_templates, REPEAT_RUNS)
            all_rows.extend(rows)

        elapsed = time.perf_counter() - t_start
        if (dir_idx + 1) % 5 == 0 or dir_idx == total_dirs - 1:
            print(f"  [{dir_idx + 1:3d}/{total_dirs}] 已完成, "
                  f"耗时 {elapsed:.0f}s, 已收集 {len(all_rows)} 条记录")

    df = pd.DataFrame(all_rows)
    elapsed_total = time.perf_counter() - t_start
    print(f"\n全部完成，总耗时 {elapsed_total:.0f}s")

    suffix = "small" if small else "full"
    csv_path = RESULTS_DIR / f"{task_type}_{suffix}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"结果已保存至: {csv_path}")

    return df


def analyze_ablation(df: pd.DataFrame, task_type: str, tag: str = "small"):
    """分析消融实验结果并绘图。

    生成：
      1. 四组的箱线图对比
      2. 逐组统计汇总
      3. 性能提升相对基线的百分比
    """
    if df.empty:
        print("DataFrame 为空，跳过分析。")
        return

    print(f"\n{'=' * 60}")
    print(f"消融实验结果 ({task_type})")
    print(f"{'=' * 60}")

    # 按组汇总
    groups_order = ["A_ClassicalOnly", "B_QuantumNoLambda",
                    "C_QuantumWithLambda", "D_FullHybrid"]

    # ---- 数值统计 ----
    baseline_mean = None
    for grp in groups_order:
        sub = df[df["algorithm"].str.startswith(grp)]
        if sub.empty:
            print(f"\n  {grp}: 无数据")
            continue
        obj_mean, obj_std = mean_std(sub["objective"].tolist())
        rt_mean, rt_std = mean_std(sub["runtime"].tolist())

        if grp == "A_ClassicalOnly":
            baseline_mean = obj_mean

        delta = ""
        if baseline_mean is not None and grp != "A_ClassicalOnly":
            improvement = (obj_mean - baseline_mean) / baseline_mean * 100 \
                if baseline_mean > 0 else 0
            delta = f" (相对基线: {improvement:+.1f}%)"

        print(f"\n  {grp}:")
        print(f"    目标值: {obj_mean:.4f} ± {obj_std:.4f}{delta}")
        print(f"    运行时间: {rt_mean:.4f}s ± {rt_std:.4f}")

    # ---- 箱线图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 目标值箱线图
    ax = axes[0]
    data_by_group = []
    labels = []
    for grp in groups_order:
        sub = df[df["algorithm"].str.startswith(grp)]
        if not sub.empty:
            data_by_group.append(sub["objective"].dropna().tolist())
            labels.append(grp.replace("_", "\n"))

    colors_list = [ABLATION_COLORS.get(g, "#999999") for g in groups_order
                   if g in df["algorithm"].str[:1].values
                   or any(df["algorithm"].str.startswith(g))]

    bp = ax.boxplot(data_by_group, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, grp in zip(bp["boxes"], groups_order):
        if grp in ABLATION_COLORS:
            patch.set_facecolor(ABLATION_COLORS[grp])
        patch.set_alpha(0.75)

    ylabel = "Clique Size |S|" if task_type == "maximum_clique" else "Density ρ(S)"
    ax.set_ylabel(ylabel)
    ax.set_title(f"Ablation Study — Objective\n{task_type}", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # 运行时间箱线图
    ax = axes[1]
    rt_data = []
    rt_labels = []
    for grp in groups_order:
        sub = df[df["algorithm"].str.startswith(grp)]
        if not sub.empty:
            rt_data.append(sub["runtime"].dropna().tolist())
            rt_labels.append(grp.replace("_", "\n"))

    bp2 = ax.boxplot(rt_data, tick_labels=rt_labels, patch_artist=True, widths=0.5)
    for patch, grp in zip(bp2["boxes"], groups_order):
        if grp in ABLATION_COLORS:
            patch.set_facecolor(ABLATION_COLORS[grp])
        patch.set_alpha(0.75)

    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(f"Ablation Study — Runtime\n{task_type}", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Experiment 3: Ablation Study\n"
                 "A=Classical | B=Quantum(λ=0) | C=Quantum(λ>0) | D=Full(Q+R+λ)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    fig_path = RESULTS_DIR / f"{task_type}_{tag}_boxplot.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n消融实验箱线图已保存至: {fig_path}")
    plt.close(fig)

    # ---- 性能提升柱状图 ----
    if baseline_mean is not None and baseline_mean > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        improvements = []
        grp_labels = []
        for grp in groups_order[1:]:  # 跳过 A 组基线
            sub = df[df["algorithm"].str.startswith(grp)]
            if sub.empty:
                continue
            obj_mean, _ = mean_std(sub["objective"].tolist())
            imp = (obj_mean - baseline_mean) / baseline_mean * 100
            improvements.append(imp)
            grp_labels.append(grp.replace("_", "\n"))

        colors_bar = [ABLATION_COLORS.get(g, "#999999") for g in groups_order[1:]]
        bars = ax.bar(grp_labels, improvements, color=colors_bar, alpha=0.8,
                      edgecolor="black", linewidth=0.5)
        ax.axhline(y=0, color="black", linewidth=0.8)

        # 标注数值
        for bar, imp in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.5 if imp >= 0 else -1.5),
                    f"{imp:+.1f}%", ha="center", fontsize=11, fontweight="bold")

        ax.set_ylabel("Improvement over Classical Baseline (%)")
        ax.set_title(f"Ablation Study — Relative Performance\n{task_type}",
                     fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fig_path = RESULTS_DIR / f"{task_type}_{tag}_improvement.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"性能提升图已保存至: {fig_path}")
        plt.close(fig)

    # ---- 保存汇总统计 ----
    summary_rows = []
    for grp in groups_order:
        sub = df[df["algorithm"].str.startswith(grp)]
        if sub.empty:
            continue
        obj_mean, obj_std = mean_std(sub["objective"].tolist())
        rt_mean, rt_std = mean_std(sub["runtime"].tolist())
        improvement = ""
        if baseline_mean and grp != "A_ClassicalOnly":
            improvement = f"{(obj_mean - baseline_mean) / baseline_mean * 100:+.1f}%"
        summary_rows.append({
            "group": grp,
            "objective_mean": obj_mean,
            "objective_std": obj_std,
            "runtime_mean": rt_mean,
            "runtime_std": rt_std,
            "n_runs": len(sub),
            "vs_baseline": improvement,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / f"{task_type}_{tag}_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"汇总统计已保存至: {summary_path}")


def _extract_n(dirname: str) -> int:
    import re
    m = re.search(r'_n(\d+)_', dirname)
    return int(m.group(1)) if m else 999


def _base_name(full_name: str) -> str:
    for key in ["A_ClassicalOnly", "B_QuantumNoLambda",
                 "C_QuantumWithLambda", "D_FullHybrid"]:
        if full_name.startswith(key):
            return key
    return full_name


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="实验三：消融实验（理论 §13.4）")
    parser.add_argument("--task", choices=["maximum_clique", "densest_subgraph"],
                        default="maximum_clique", help="任务类型")
    parser.add_argument("--small", action="store_true", default=True,
                        help="仅使用小规模实例（默认）")
    parser.add_argument("--full", action="store_true", default=False,
                        help="使用全部数据集")
    parser.add_argument("--csv", type=str, default=None,
                        help="直接分析已有 CSV")
    args = parser.parse_args()

    if args.full:
        args.small = False

    if args.csv:
        df = pd.read_csv(args.csv)
        task = "maximum_clique" if "mc_" in args.csv else "densest_subgraph"
        analyze_ablation(df, task, tag="loaded")
        return

    print(f"实验三：消融实验（{args.task}）")
    print(f"  模式: {'小规模 (n≤50)' if args.small else '全部数据集'}")

    df = run_ablation_experiment(args.task, small=args.small)
    if not df.empty:
        tag = "small" if args.small else "full"
        analyze_ablation(df, args.task, tag)

    print(f"\n实验三完成。")


if __name__ == "__main__":
    main()
