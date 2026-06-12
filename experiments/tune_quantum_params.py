#!/usr/bin/env python3
"""量子参数调优脚本（Phase 3.1.5）。

目的：
  在 small 规模实例（n≤50）上扫描 QuantumGuidedGreedy 的关键参数
  (init_method, α)，找出在多个实例上稳健胜出的组合。
  扫描结果用于决定实验二 medium/full 规模运行时采用哪组参数。

扫描参数：
  - init_method: uniform / max_degree (2 种)
  - alpha: 0.0 / 0.25 / 0.5 / 0.75 / 1.0 (5 种，对应理论 §13.5)
  - 固定 t=1.0, lam=0.5

对照基线：
  ClassicalClique（MC 任务的强经典基线）—— 每实例独立运行作为参照。

输出：
  - results/tune_quantum_params/<task>_full_results.csv  原始数据
  - results/tune_quantum_params/<task>_summary.csv       按 (init,α) 聚合
  - results/tune_quantum_params/<task>_heatmap.png       热力图
  - 控制台：最佳参数排名表 + 相对基线的差异

用法：
  python experiments/tune_quantum_params.py --task maximum_clique
  python experiments/tune_quantum_params.py --task maximum_clique --quick  # 减少扫描点
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Windows 控制台 UTF-8 设置（避免 GBK 编码错误）
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_utils import load_instances_from_dir
from src.candidate_set import CliqueCandidateSet, DenseCandidateSet
from src.scoring import ClassicalCliqueScorer, ClassicalDenseScorer
from src.algorithms.classical_greedy import ClassicalGreedy
from src.algorithms.quantum_greedy import QuantumGuidedGreedy
from src.metrics import mean_std
from src.config import get_data_dirs

# ============================================================
# 配置
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "tune_quantum_params"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 每实例重复次数
REPEAT_RUNS = 4

# 默认扫描点
DEFAULT_INITS = ["uniform", "max_degree"]
DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
QUICK_ALPHAS = [0.0, 0.5, 1.0]  # --quick 时减少扫描

# 固定的 CTQW 参数（α 是扫描重点，t/lam 用合理默认值）
FIXED_T = 1.0
FIXED_LAM = 0.5


# ============================================================
# 数据筛选
# ============================================================

def _extract_n(dirname: str) -> int:
    """从目录名提取节点数 n。"""
    import re
    m = re.search(r'_n(\d+)_', dirname)
    return int(m.group(1)) if m else 999


def discover_small_instances(task_type: str,
                              min_n: int = 0, max_n: int = 50):
    """收集所有满足 min_n ≤ n ≤ max_n 的实例。

    返回:
        [(instance, dir_name)] 列表。
    """
    dirs = get_data_dirs(task_type)
    dirs = [d for d in dirs
            if min_n <= _extract_n(d.name) <= max_n]
    print(f"参数组数: {len(dirs)}（{min_n} ≤ n ≤ {max_n}）")

    instances = []
    for d in dirs:
        for inst in load_instances_from_dir(d):
            instances.append((inst, d.name))
    print(f"实例总数: {len(instances)}")
    return instances


# ============================================================
# 单组参数运行
# ============================================================

def run_one_config(instances, task_type: str,
                    init_method: str, alpha: float,
                    repeat: int) -> list[dict]:
    """对一组 (init, α) 配置在所有实例上运行 QuantumGuidedGreedy。

    参数:
        instances: [(GraphInstance, dir_name), ...]
        task_type: "maximum_clique" 或 "densest_subgraph"
        init_method: CTQW 初始化方式
        alpha: 混合评分权重
        repeat: 每实例重复次数

    返回:
        扁平结果字典列表。
    """
    builder = (CliqueCandidateSet() if task_type == "maximum_clique"
               else DenseCandidateSet())

    rows = []
    for inst, dir_name in instances:
        for run_id in range(repeat):
            algo = QuantumGuidedGreedy(
                builder,
                t=FIXED_T,
                lam=FIXED_LAM,
                init_method=init_method,
                alpha=alpha,
                seed=run_id,
                name=f"Q(init={init_method},α={alpha})",
            )
            result = algo.solve(inst)
            rows.append({
                "algorithm": "QuantumGuidedGreedy",
                "init_method": init_method,
                "alpha": alpha,
                "sample_id": inst.sample_id,
                "dir_name": dir_name,
                "run_id": run_id,
                "objective": result.objective,
                "runtime": result.runtime,
                "iterations": result.iterations,
            })
    return rows


def run_baseline(instances, task_type: str, repeat: int) -> list[dict]:
    """运行 ClassicalClique 基线，每实例重复 repeat 次。"""
    if task_type == "maximum_clique":
        builder = CliqueCandidateSet()
        scorer = ClassicalCliqueScorer()
        algo_name = "ClassicalClique"
    else:
        builder = DenseCandidateSet()
        scorer = ClassicalDenseScorer()
        algo_name = "ClassicalDense"

    rows = []
    for inst, dir_name in instances:
        for run_id in range(repeat):
            # 经典贪心是确定性的（无随机），但跑 4 次保持配对一致
            algo = ClassicalGreedy(builder, scorer, name=algo_name)
            result = algo.solve(inst)
            rows.append({
                "algorithm": algo_name,
                "init_method": "-",
                "alpha": -1.0,  # -1 表示基线，方便后续筛选
                "sample_id": inst.sample_id,
                "dir_name": dir_name,
                "run_id": run_id,
                "objective": result.objective,
                "runtime": result.runtime,
                "iterations": result.iterations,
            })
    return rows


# ============================================================
# 分析与可视化
# ============================================================

def analyze_results(df: pd.DataFrame, baseline_name: str, task_type: str
                     ) -> pd.DataFrame:
    """计算每组 (init, α) 的统计指标，并与基线做配对显著性检验。

    返回:
        DataFrame，每行一组参数，列含 mean / std / vs_baseline / p_value 等。
    """
    # 基线查找表：按 (sample_id, run_id) 索引
    base_df = df[df["algorithm"] == baseline_name]
    base_lookup = {
        (r["sample_id"], r["run_id"]): r["objective"]
        for _, r in base_df.iterrows()
    }
    base_obj_mean, base_obj_std = mean_std(base_df["objective"].tolist())

    quantum_df = df[df["algorithm"] == "QuantumGuidedGreedy"]

    rows = []
    # 基线行（参考用）
    rows.append({
        "init_method": "-",
        "alpha": "(baseline)",
        "obj_mean": base_obj_mean,
        "obj_std": base_obj_std,
        "runtime_mean": float(base_df["runtime"].mean()),
        "diff_vs_baseline": 0.0,
        "win_rate": float("nan"),
        "p_value": float("nan"),
        "verdict": "(baseline ClassicalClique)",
    })

    for (init, alpha), sub in quantum_df.groupby(["init_method", "alpha"]):
        algo_vals = []
        base_vals = []
        for _, r in sub.iterrows():
            key = (r["sample_id"], r["run_id"])
            if key in base_lookup:
                algo_vals.append(r["objective"])
                base_vals.append(base_lookup[key])

        algo_arr = np.array(algo_vals)
        base_arr = np.array(base_vals)
        diffs = algo_arr - base_arr
        obj_mean, obj_std = mean_std(algo_vals)

        # 胜率：QuantumGreedy 严格优于基线的比例（不含相等）
        n_wins = int((diffs > 0).sum())
        n_losses = int((diffs < 0).sum())
        n_ties = int((diffs == 0).sum())
        win_rate = n_wins / len(diffs) if len(diffs) > 0 else 0.0

        # Wilcoxon paired test
        if np.allclose(diffs, 0):
            p_val = 1.0
            verdict = "与基线无差异"
        else:
            try:
                _, p_val = wilcoxon(algo_arr, base_arr,
                                     zero_method="wilcox",
                                     alternative="two-sided")
            except ValueError:
                p_val = float("nan")

            mean_d = float(diffs.mean())
            if np.isnan(p_val):
                verdict = "无法计算 p-value"
            elif p_val < 0.05:
                verdict = ("✓ 显著优于基线" if mean_d > 0
                           else "✗ 显著劣于基线")
            else:
                verdict = f"~ 无显著差异 (p={p_val:.3f})"

        rows.append({
            "init_method": init,
            "alpha": alpha,
            "obj_mean": obj_mean,
            "obj_std": obj_std,
            "runtime_mean": float(sub["runtime"].mean()),
            "diff_vs_baseline": float(diffs.mean()),
            "win_rate": win_rate,
            "p_value": p_val,
            "verdict": verdict,
        })

    return pd.DataFrame(rows)


def plot_heatmap(summary_df: pd.DataFrame, task_type: str, save_path: Path):
    """画 init × α 的 mean objective 热力图。"""
    quantum_rows = summary_df[summary_df["alpha"] != "(baseline)"].copy()
    if quantum_rows.empty:
        print("无 QuantumGreedy 数据，跳过热力图。")
        return

    quantum_rows["alpha"] = quantum_rows["alpha"].astype(float)
    pivot = quantum_rows.pivot(index="init_method", columns="alpha",
                                values="obj_mean")
    pivot_diff = quantum_rows.pivot(index="init_method", columns="alpha",
                                     values="diff_vs_baseline")

    baseline_row = summary_df[summary_df["alpha"] == "(baseline)"]
    baseline_val = float(baseline_row["obj_mean"].iloc[0]) \
        if not baseline_row.empty else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 左图：原始目标值
    ax = axes[0]
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto",
                    origin="lower", interpolation="nearest")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            color = "white" if val > pivot.values.max() * 0.7 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                     fontsize=11, fontweight="bold", color=color)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"α={a}" for a in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"QuantumGreedy Mean Objective\n"
                 f"(baseline ClassicalClique = {baseline_val:.3f})",
                 fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.85)

    # 右图：相对基线差异（红=胜，蓝=负）
    ax = axes[1]
    vmax = np.nanmax(np.abs(pivot_diff.values))
    im = ax.imshow(pivot_diff.values, cmap="RdBu_r",
                    aspect="auto", origin="lower", interpolation="nearest",
                    vmin=-vmax, vmax=vmax)
    for i in range(pivot_diff.shape[0]):
        for j in range(pivot_diff.shape[1]):
            val = pivot_diff.values[i, j]
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                     fontsize=11, fontweight="bold")
    ax.set_xticks(range(pivot_diff.shape[1]))
    ax.set_xticklabels([f"α={a}" for a in pivot_diff.columns])
    ax.set_yticks(range(pivot_diff.shape[0]))
    ax.set_yticklabels(pivot_diff.index)
    ax.set_title("Difference vs Baseline (red = QuantumGreedy wins)",
                 fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(f"Parameter Tuning ({task_type})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n热力图已保存: {save_path}")


def print_ranking(summary_df: pd.DataFrame):
    """按目标值降序打印参数排名。"""
    quantum_rows = summary_df[summary_df["alpha"] != "(baseline)"].copy()
    quantum_rows = quantum_rows.sort_values("obj_mean", ascending=False)
    baseline_row = summary_df[summary_df["alpha"] == "(baseline)"]
    baseline_val = float(baseline_row["obj_mean"].iloc[0]) \
        if not baseline_row.empty else 0.0
    baseline_std = float(baseline_row["obj_std"].iloc[0]) \
        if not baseline_row.empty else 0.0

    print(f"\n{'=' * 78}")
    print(f"QuantumGreedy 参数扫描结果（按目标值降序）")
    print(f"基线 ClassicalClique: {baseline_val:.4f} ± {baseline_std:.4f}")
    print(f"{'=' * 78}")
    print(f"{'init':>12} {'α':>6}   {'obj_mean':>10} {'±std':>8}   "
          f"{'Δ vs base':>10} {'胜率':>6}   结论")
    print("-" * 78)

    for _, r in quantum_rows.iterrows():
        symbol = "★" if r["diff_vs_baseline"] > 0 else " "
        win_pct = f"{r['win_rate']*100:.1f}%" if not pd.isna(r['win_rate']) else "n/a"
        print(f"{r['init_method']:>12} {r['alpha']:>6.2f}   "
              f"{r['obj_mean']:>10.4f} {r['obj_std']:>8.4f}   "
              f"{r['diff_vs_baseline']:>+10.4f} {win_pct:>6}   "
              f"{symbol} {r['verdict']}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="QuantumGreedy 参数扫描（init × α）")
    parser.add_argument("--task", choices=["maximum_clique", "densest_subgraph"],
                        default="maximum_clique", help="任务类型")
    parser.add_argument("--quick", action="store_true",
                        help="只扫 α ∈ {0, 0.5, 1}（减少耗时）")
    parser.add_argument("--max-n", type=int, default=50,
                        help="实例规模上限（默认 50）")
    parser.add_argument("--min-n", type=int, default=0,
                        help="实例规模下限（默认 0，含所有 ≥ min-n 的图）")
    parser.add_argument("--init", choices=["uniform", "max_degree", "both"],
                        default="both",
                        help="只扫指定的 init 方式（默认 both = 两者都扫）")
    parser.add_argument("--tag", type=str, default="",
                        help="结果文件名后缀，避免覆盖之前的扫描结果")
    args = parser.parse_args()

    alphas = QUICK_ALPHAS if args.quick else DEFAULT_ALPHAS
    if args.init == "both":
        inits = DEFAULT_INITS
    else:
        inits = [args.init]

    print(f"参数扫描配置:")
    print(f"  任务:        {args.task}")
    print(f"  实例 n 范围:  [{args.min_n}, {args.max_n}]")
    print(f"  init 取值:   {inits}")
    print(f"  α 取值:      {alphas}")
    print(f"  固定 t=:    {FIXED_T}")
    print(f"  固定 λ=:    {FIXED_LAM}")
    print(f"  每实例重复:  {REPEAT_RUNS}")

    # ---- 收集实例 ----
    instances = discover_small_instances(args.task, min_n=args.min_n,
                                          max_n=args.max_n)
    if not instances:
        print("未找到任何实例，退出。")
        return

    n_configs = len(inits) * len(alphas)
    total_runs = (n_configs + 1) * len(instances) * REPEAT_RUNS
    print(f"\n  共 {n_configs} 组 Quantum 参数 + 1 组基线")
    print(f"  预计运行次数: {total_runs}")
    print("-" * 60)

    # ---- 跑基线 ----
    t_start = time.perf_counter()
    print(f"\n[1/{n_configs + 1}] 跑基线 ClassicalClique...")
    all_rows = run_baseline(instances, args.task, REPEAT_RUNS)
    print(f"  完成: {len(all_rows)} 条记录, 耗时 {time.perf_counter() - t_start:.1f}s")

    # ---- 跑各组 Quantum 参数 ----
    config_idx = 1
    for init in inits:
        for alpha in alphas:
            config_idx += 1
            cfg_start = time.perf_counter()
            print(f"\n[{config_idx}/{n_configs + 1}] "
                  f"init={init}, α={alpha} ...")
            rows = run_one_config(instances, args.task,
                                   init_method=init, alpha=alpha,
                                   repeat=REPEAT_RUNS)
            all_rows.extend(rows)
            print(f"  完成: {len(rows)} 条记录, "
                  f"本组耗时 {time.perf_counter() - cfg_start:.1f}s, "
                  f"累计 {time.perf_counter() - t_start:.0f}s")

    df = pd.DataFrame(all_rows)
    total_time = time.perf_counter() - t_start
    print(f"\n全部完成，总耗时 {total_time:.0f}s，共 {len(df)} 条记录")

    # 文件名 tag（区分不同规模的扫描结果）
    tag_suffix = f"_{args.tag}" if args.tag else ""

    # ---- 保存原始数据 ----
    full_path = RESULTS_DIR / f"{args.task}{tag_suffix}_full_results.csv"
    df.to_csv(full_path, index=False, encoding="utf-8")
    print(f"原始数据已保存: {full_path}")

    # ---- 分析 ----
    baseline_name = ("ClassicalClique" if args.task == "maximum_clique"
                     else "ClassicalDense")
    summary = analyze_results(df, baseline_name, args.task)
    summary_path = RESULTS_DIR / f"{args.task}{tag_suffix}_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"汇总统计已保存: {summary_path}")

    # ---- 排名打印 + 热力图 ----
    print_ranking(summary)
    heatmap_path = RESULTS_DIR / f"{args.task}{tag_suffix}_heatmap.png"
    plot_heatmap(summary, args.task, heatmap_path)

    # ---- 最佳参数 ----
    quantum_rows = summary[summary["alpha"] != "(baseline)"].copy()
    best = quantum_rows.sort_values("obj_mean", ascending=False).iloc[0]
    print(f"\n{'=' * 60}")
    print(f"最佳参数组合:")
    print(f"  init_method = {best['init_method']}")
    print(f"  alpha       = {best['alpha']}")
    print(f"  目标值      = {best['obj_mean']:.4f}")
    print(f"  vs 基线      = {best['diff_vs_baseline']:+.4f}")
    print(f"  结论        = {best['verdict']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
