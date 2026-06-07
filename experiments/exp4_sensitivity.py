#!/usr/bin/env python3
"""实验四：参数敏感性分析（理论 §13.5）。

目的：
  验证算法在不同参数下的稳定性。

扫描参数：
  - 演化时间 t   ∈ {0.5, 1, 2, 5, 10}
  - 扰动强度 λ   ∈ {0, 0.1, 0.5, 1, 2, 5}
  - 混合权重 α   ∈ {0, 0.25, 0.5, 0.75, 1}

输出：
  - t-λ 热力图（固定 α=0.5）
  - α-λ 热力图（固定 t=1.0）
  - t-α 热力图（固定 λ=0.5）
  - 各参数对 objective 的影响曲线（带误差带）

用法:
  python3 experiments/exp4_sensitivity.py
  python3 experiments/exp4_sensitivity.py --task maximum_clique --instance-idx 0
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

from src.graph_utils import load_instance
from src.candidate_set import CliqueCandidateSet, DenseCandidateSet
from src.algorithms.quantum_greedy import QuantumGuidedGreedy
from src.metrics import mean_std
from src.config import T_VALUES, LAMBDA_VALUES, ALPHA_VALUES, get_data_dirs

# ============================================================
# 配置
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp4_sensitivity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPEAT_RUNS = 2  # 敏感性扫描次数较多，减少重复以控制时间


# ============================================================
# 参数扫描
# ============================================================

def scan_t_lambda(instance, builder, base_alpha: float = 0.5,
                   t_values: list = None, lam_values: list = None
                   ) -> pd.DataFrame:
    """扫描 t 和 λ 对算法性能的影响（固定 α）。

    返回:
        包含 t, lam, objective, runtime 的 DataFrame。
    """
    t_values = t_values or T_VALUES
    lam_values = lam_values or LAMBDA_VALUES
    rows = []

    total = len(t_values) * len(lam_values)
    count = 0
    for t in t_values:
        for lam in lam_values:
            for run_id in range(REPEAT_RUNS):
                algo = QuantumGuidedGreedy(
                    builder, t=t, lam=lam, alpha=base_alpha,
                    init_method="max_degree", seed=run_id,
                    name=f"Sensitivity(t={t},λ={lam})")
                result = algo.solve(instance)
                row = result.to_dict()
                row["t"] = t
                row["lam"] = lam
                row["run_id"] = run_id
                rows.append(row)
            count += 1
            if count % 10 == 0:
                print(f"  t-λ 扫描: [{count}/{total}]")

    return pd.DataFrame(rows)


def scan_alpha_parameter(instance, builder, fixed_t: float = 1.0,
                          fixed_lam: float = 0.5,
                          param_name: str = "lam",
                          param_values: list = None,
                          alpha_values: list = None
                          ) -> pd.DataFrame:
    """扫描 α 与另一参数对算法性能的影响。

    参数:
        param_name: 变化参数的名称 ("lam" 或 "t")。
        param_values: 该参数的取值列表。
        alpha_values: α 取值列表。

    返回:
        包含 alpha, {param_name}, objective, runtime 的 DataFrame。
    """
    param_values = param_values or LAMBDA_VALUES
    alpha_values = alpha_values or ALPHA_VALUES
    rows = []

    total = len(param_values) * len(alpha_values)
    count = 0
    for pv in param_values:
        for alpha in alpha_values:
            for run_id in range(REPEAT_RUNS):
                if param_name == "lam":
                    algo = QuantumGuidedGreedy(
                        builder, t=fixed_t, lam=pv, alpha=alpha,
                        init_method="max_degree", seed=run_id,
                        name=f"Sensitivity(α={alpha},λ={pv})")
                else:
                    algo = QuantumGuidedGreedy(
                        builder, t=pv, lam=fixed_lam, alpha=alpha,
                        init_method="max_degree", seed=run_id,
                        name=f"Sensitivity(α={alpha},t={pv})")
                result = algo.solve(instance)
                row = result.to_dict()
                row["alpha"] = alpha
                row[param_name] = pv
                row["run_id"] = run_id
                rows.append(row)
            count += 1
            if count % 10 == 0:
                print(f"  α-{param_name} 扫描: [{count}/{total}]")

    return pd.DataFrame(rows)


# ============================================================
# 可视化
# ============================================================

def plot_heatmap(df: pd.DataFrame, row_param: str, col_param: str,
                  task_type: str, save_name: str):
    """绘制参数热力图。

    参数:
        df: 包含 row_param, col_param, objective 列。
        row_param: 行参数名（如 "t" 或 "lam"）。
        col_param: 列参数名（如 "lam" 或 "alpha"）。
    """
    # 聚合：对每个参数组合取均值
    agg = df.groupby([row_param, col_param])["objective"].mean().reset_index()
    pivot = agg.pivot(index=row_param, columns=col_param, values="objective")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                    origin="lower", interpolation="nearest")

    # 标注数值
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if val > pivot.values.max() * 0.7 else "black")

    ylabel = "Clique Size |S|" if task_type == "maximum_clique" else "Density ρ(S)"

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v}" for v in pivot.columns], rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v}" for v in pivot.index])

    ax.set_xlabel(col_param)
    ax.set_ylabel(row_param)
    ax.set_title(f"Parameter Sensitivity: {row_param} vs {col_param}\n"
                 f"{task_type} — {ylabel}", fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(ylabel)

    plt.tight_layout()
    fig_path = RESULTS_DIR / save_name
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  热力图已保存: {fig_path}")
    plt.close(fig)


def plot_sensitivity_curves(df: pd.DataFrame, x_param: str,
                             group_param: str,
                             task_type: str, save_name: str):
    """绘制带误差带的敏感性曲线。

    参数:
        df: 数据。
        x_param: x 轴参数（如 "t", "lam", "alpha"）。
        group_param: 分组参数（用于绘制多条曲线）。
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    groups = sorted(df[group_param].unique())
    cmap = plt.cm.viridis

    for i, grp in enumerate(groups):
        sub = df[df[group_param] == grp]
        x_vals = sorted(sub[x_param].unique())
        means = []
        stds = []
        for xv in x_vals:
            vals = sub[sub[x_param] == xv]["objective"].tolist()
            if vals:
                m, s = mean_std(vals)
                means.append(m)
                stds.append(s)
            else:
                means.append(np.nan)
                stds.append(np.nan)

        color = cmap(i / max(len(groups) - 1, 1))
        ax.errorbar(x_vals, means, yerr=stds, marker="o", markersize=5,
                     linewidth=1.5, capsize=3, color=color,
                     label=f"{group_param}={grp}")

    ylabel = "Clique Size |S|" if task_type == "maximum_clique" else "Density ρ(S)"
    ax.set_xlabel(x_param)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Parameter Sensitivity — {x_param}\n{task_type}",
                 fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = RESULTS_DIR / save_name
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  敏感性曲线已保存: {fig_path}")
    plt.close(fig)


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="实验四：参数敏感性分析（理论 §13.5）")
    parser.add_argument("--task", choices=["maximum_clique", "densest_subgraph"],
                        default="maximum_clique", help="任务类型")
    parser.add_argument("--instance", type=str, default=None,
                        help="指定 JSON 文件路径")
    parser.add_argument("--instance-idx", type=int, default=0,
                        help="使用第 N 个测试实例（默认 0）")
    args = parser.parse_args()

    # 查找测试实例
    if args.instance:
        instance_path = args.instance
    else:
        dirs = get_data_dirs(args.task)
        if not dirs:
            print(f"未找到 {args.task} 的数据目录")
            return
        # 选择第一个小规模目录
        small_dirs = [d for d in dirs if _extract_n(d.name) <= 50]
        target_dir = small_dirs[0] if small_dirs else dirs[0]
        json_files = sorted(target_dir.glob("*.json"))
        if not json_files:
            print(f"目录 {target_dir} 中无 JSON 文件")
            return
        instance_path = str(json_files[min(args.instance_idx, len(json_files) - 1)])

    print(f"实验四：参数敏感性分析")
    print(f"  数据: {instance_path}")
    print(f"  结果目录: {RESULTS_DIR}")

    instance = load_instance(instance_path)
    task_type = instance.task_type
    print(f"  实例: {instance.sample_id}")
    print(f"  节点数: {instance.num_nodes}, 答案大小: {len(instance.answer_nodes)}")

    if task_type == "maximum_clique":
        builder = CliqueCandidateSet()
    else:
        builder = DenseCandidateSet()

    t_start = time.perf_counter()

    # ---- 1. t-λ 热力图（固定 α=0.5） ----
    print("\n[1/5] 扫描 t-λ 空间...")
    df_t_lam = scan_t_lambda(instance, builder, base_alpha=0.5)
    plot_heatmap(df_t_lam, "t", "lam", task_type,
                  f"{instance.sample_id}_heatmap_t_lam.png")

    # ---- 2. α-λ 热力图（固定 t=1.0） ----
    print("\n[2/5] 扫描 α-λ 空间...")
    df_alpha_lam = scan_alpha_parameter(
        instance, builder, fixed_t=1.0, param_name="lam",
        param_values=LAMBDA_VALUES, alpha_values=ALPHA_VALUES)
    plot_heatmap(df_alpha_lam, "alpha", "lam", task_type,
                  f"{instance.sample_id}_heatmap_alpha_lam.png")

    # ---- 3. t-α 热力图（固定 λ=0.5） ----
    print("\n[3/5] 扫描 t-α 空间...")
    df_t_alpha = scan_alpha_parameter(
        instance, builder, fixed_lam=0.5, param_name="t",
        param_values=T_VALUES, alpha_values=ALPHA_VALUES)
    plot_heatmap(df_t_alpha, "alpha", "t", task_type,
                  f"{instance.sample_id}_heatmap_alpha_t.png")

    # ---- 4. α 敏感性曲线 ----
    print("\n[4/5] 绘制 α 敏感性曲线...")
    # 合并 α 在不同 t 下的结果
    plot_sensitivity_curves(df_t_alpha, "alpha", "t", task_type,
                             f"{instance.sample_id}_curve_alpha_by_t.png")
    # α 在不同 λ 下的结果
    plot_sensitivity_curves(df_alpha_lam, "alpha", "lam", task_type,
                             f"{instance.sample_id}_curve_alpha_by_lam.png")

    # ---- 5. 保存汇总结果 ----
    print("\n[5/5] 保存汇总数据...")
    all_dfs = {
        "t_lam": df_t_lam,
        "alpha_lam": df_alpha_lam,
        "t_alpha": df_t_alpha,
    }
    for name, df_part in all_dfs.items():
        csv_path = RESULTS_DIR / f"{instance.sample_id}_{name}.csv"
        df_part.to_csv(csv_path, index=False, encoding="utf-8")

    elapsed_total = time.perf_counter() - t_start
    print(f"\n实验四完成，总耗时 {elapsed_total:.0f}s")
    print(f"生成图表保存在: {RESULTS_DIR}")

    # 打印最优参数区域
    print(f"\n{'=' * 60}")
    print("参数最优区域汇总（基于当前占位 CTQW 实现）:")
    for name, df_part in all_dfs.items():
        best_row = df_part.loc[df_part["objective"].idxmax()]
        print(f"  {name}: best_obj={best_row['objective']:.4f}, "
              + ", ".join(f"{k}={best_row.get(k, '?')}"
                          for k in ["t", "lam", "alpha"] if k in best_row))
    print("\n注意: 以上结果基于 CTQW 占位实现，真实 CTQW 接入后需重新扫描。")


def _extract_n(dirname: str) -> int:
    import re
    m = re.search(r'_n(\d+)_', dirname)
    return int(m.group(1)) if m else 999


if __name__ == "__main__":
    main()
