#!/usr/bin/env python3
"""实验二：与经典贪心算法对比（理论 §13.3）。

目的：
  验证量子引导评分是否提升求解质量。

对照方法：
  1. 纯度数贪心           — 理论 §3.1 基线
  2. 候选子图度数贪心      — 理论 §8.1 最大团经典评分
  3. 模拟退火              — 理论 §13.3 经典启发式对照
  4. 密度增量贪心          — 理论 §8.1 密集子图经典评分（密集子图任务）
  5. 量子引导贪心          — 理论 §12 完整算法（CTQW 占位）

评价指标：
  - 最大团任务：团大小 |S|、recall
  - 密集子图任务：密度 ρ(S)
  - 运行时间、成功率、均值与方差

用法:
  python3 experiments/exp2_algorithm_comparison.py
  python3 experiments/exp2_algorithm_comparison.py --task maximum_clique --small
  python3 experiments/exp2_algorithm_comparison.py --task densest_subgraph --small
  python3 experiments/exp2_algorithm_comparison.py --full  # 全部数据集（耗时较长）
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

from src.graph_utils import load_instance, load_instances_from_dir, GraphInstance
from src.candidate_set import CliqueCandidateSet, DenseCandidateSet
from src.scoring import (ClassicalCliqueScorer, ClassicalDenseScorer,
                          ClassicalDegreeScorer, HybridScorer, QuantumScorer)
from src.algorithms.classical_greedy import ClassicalGreedy
from src.algorithms.simulated_annealing import SimulatedAnnealing
from src.algorithms.quantum_greedy import QuantumGuidedGreedy
from src.metrics import mean_std, success_rate, aggregate_results
from src.config import get_data_dirs, DATA_DIR

# ============================================================
# 配置
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp2_algorithm_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPEAT_RUNS = 4  # 每个实例重复运行次数

# 图表样式
COLORS = {
    "ClassicalDegree": "#1f77b4",
    "ClassicalClique": "#ff7f0e",
    "SimulatedAnnealing": "#2ca02c",
    "QuantumGuidedGreedy": "#9467bd",
    "ClassicalDense": "#d62728",
}


# ============================================================
# 算法工厂
# ============================================================

def build_algorithms_for_task(task_type: str, seed: int = 0) -> dict:
    """根据任务类型构造算法集合。

    参数:
        task_type: "maximum_clique" 或 "densest_subgraph"
        seed: 基础随机种子

    返回:
        {算法名称: 算法对象} 字典
    """
    if task_type == "maximum_clique":
        builder = CliqueCandidateSet()
        classical_scorer = ClassicalCliqueScorer()
    else:
        builder = DenseCandidateSet()
        classical_scorer = ClassicalDenseScorer()

    quantum_scorer = QuantumScorer(t=1.0, lam=0.5, seed=seed)
    hybrid_scorer = HybridScorer(quantum_scorer, classical_scorer, alpha=0.5)

    algorithms = {
        "ClassicalDegree": ClassicalGreedy(
            builder, ClassicalDegreeScorer(), name="ClassicalDegree"),
        "ClassicalClique" if task_type == "maximum_clique" else "ClassicalDense":
            ClassicalGreedy(builder, classical_scorer, name="ClassicalClique"
                            if task_type == "maximum_clique" else "ClassicalDense"),
        "SimulatedAnnealing": SimulatedAnnealing(
            builder, seed=seed, max_iterations=2000, name="SimulatedAnnealing"),
        "QuantumGuidedGreedy": QuantumGuidedGreedy(
            builder, t=1.0, lam=0.5, alpha=0.5, seed=seed,
            name="QuantumGuidedGreedy"),
    }
    return algorithms


# ============================================================
# 单实例实验
# ============================================================

def run_comparison_on_instance(instance: GraphInstance,
                                algorithms: dict,
                                repeat: int = REPEAT_RUNS) -> list[dict]:
    """对单个实例运行所有算法，每种重复 repeat 次。"""
    rows = []
    for algo_name, algo_template in algorithms.items():
        for run_id in range(repeat):
            # 用不同 seed 重新构造算法
            seed = run_id
            algo = _rebuild_algo(algo_template, seed)
            result = algo.solve(instance)
            row = result.to_dict()
            row["run_id"] = run_id
            row["seed"] = seed
            rows.append(row)
    return rows


def _rebuild_algo(template, seed: int):
    """根据模板算法创建带有新 seed 的副本。"""
    from src.candidate_set import CliqueCandidateSet, DenseCandidateSet
    from src.scoring import (ClassicalCliqueScorer, ClassicalDenseScorer,
                              ClassicalDegreeScorer)

    name = template.name
    builder = template.candidate_builder

    if isinstance(template, ClassicalGreedy):
        scorer = template.scorer
        # 重建 scorer（ClassicalDegreeScorer 无状态，直接复用）
        if isinstance(scorer, ClassicalDegreeScorer):
            new_scorer = ClassicalDegreeScorer()
        elif isinstance(scorer, ClassicalCliqueScorer):
            new_scorer = ClassicalCliqueScorer()
        elif isinstance(scorer, ClassicalDenseScorer):
            new_scorer = ClassicalDenseScorer()
        else:
            new_scorer = scorer
        return ClassicalGreedy(builder, new_scorer, name=f"{name}(run={seed})")

    elif isinstance(template, SimulatedAnnealing):
        return SimulatedAnnealing(
            builder, seed=seed,
            T0=template.T0,
            cooling_rate=template.cooling_rate,
            max_iterations=template.max_iterations,
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
# 批量实验
# ============================================================

def run_batch_experiment(task_type: str,
                          small: bool = True,
                          data_source: str = "artificial") -> pd.DataFrame:
    """批量运行对比实验。

    参数:
        task_type: 任务类型。
        small: True 时仅使用小规模实例（n ≤ 50），仅对 artificial 有效。
        data_source: "artificial"（人工数据）或 "external"（外部 DIMACS 数据）。

    返回:
        汇总 DataFrame。
    """
    # 根据数据来源发现实例
    instances_with_labels = _discover_instances(task_type, data_source, small)
    if not instances_with_labels:
        return pd.DataFrame()

    # 确定合理的重复次数
    if data_source == "external":
        repeat_runs = 10  # 外部数据单实例，需更多运行内统计
    else:
        repeat_runs = REPEAT_RUNS  # 4 次，配合 5 实例 → 20 数据点

    all_rows = []
    total_instances = len(instances_with_labels)

    print(f"\n实验二：算法对比实验")
    print(f"  任务类型: {task_type}")
    print(f"  数据来源: {data_source}")
    print(f"  实例总数: {total_instances}")
    print(f"  每实例重复: {repeat_runs} 次")
    print(f"  预期总记录: {total_instances * 4 * repeat_runs} 条")
    print("-" * 50)

    t_start = time.perf_counter()

    for idx, (inst, label) in enumerate(instances_with_labels):
        algo_template = build_algorithms_for_task(task_type, seed=0)

        rows = run_comparison_on_instance(inst, algo_template, repeat_runs)
        for row in rows:
            row["source_label"] = label
        all_rows.extend(rows)

        elapsed = time.perf_counter() - t_start
        if (idx + 1) % 5 == 0 or idx == total_instances - 1:
            print(f"  [{idx + 1:3d}/{total_instances}] 已完成, "
                  f"耗时 {elapsed:.0f}s, 已收集 {len(all_rows)} 条记录")

    df = pd.DataFrame(all_rows)
    elapsed_total = time.perf_counter() - t_start
    print(f"\n全部完成，总耗时 {elapsed_total:.0f}s, "
          f"共 {len(df)} 条记录")

    # 保存
    suffix = f"{data_source}_small" if small and data_source == "artificial" else data_source
    csv_path = RESULTS_DIR / f"{task_type}_{suffix}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"结果已保存至: {csv_path}")

    return df


def _discover_instances(task_type: str, data_source: str, small: bool
                         ) -> list[tuple]:
    """根据数据来源发现所有测试实例。

    返回:
        [(GraphInstance, label), ...] 列表，label 为参数组名或数据集名。
    """
    instances = []

    if data_source == "artificial":
        dirs = get_data_dirs(task_type)
        if not dirs:
            return []
        if small:
            dirs = [d for d in dirs if _extract_n(d.name) <= 50]
            print(f"人工数据（小规模 n≤50）：筛选到 {len(dirs)} 个参数组目录")
        else:
            print(f"人工数据：{len(dirs)} 个参数组目录")

        for d in dirs:
            for inst in load_instances_from_dir(d):
                instances.append((inst, d.name))

    elif data_source == "external":
        ext_dir = DATA_DIR / "external" / task_type
        if not ext_dir.is_dir():
            print(f"错误: 外部数据目录不存在: {ext_dir}")
            print(f"请先运行: cd datasets && python -m converters.convert_dimacs")
            return []

        json_files = sorted(ext_dir.glob("*.json"))
        print(f"外部数据：{len(json_files)} 个实例")

        for fpath in json_files:
            stem = fpath.stem
            if stem.startswith("ext_mc_"):
                label = stem[7:]
            elif stem.startswith("ext_ds_"):
                label = stem[7:]
            else:
                label = stem
            instances.append((load_instance(fpath), label))

    return instances


def _extract_n(dirname: str) -> int:
    """从目录名提取节点数 n。"""
    import re
    m = re.search(r'_n(\d+)_', dirname)
    return int(m.group(1)) if m else 999


# ============================================================
# 结果分析与可视化
# ============================================================

def analyze_and_plot(df: pd.DataFrame, task_type: str, tag: str = "small"):
    """分析实验结果并生成图表。

    按算法基名（去掉 run_id 后缀）聚合所有运行结果，生成：
      1. 各算法 objective 的箱线图
      2. 各算法运行时间的箱线图
      3. 按参数组合分组的柱状图
      4. 汇总统计表（CSV）
    """
    if df.empty:
        print("DataFrame 为空，跳过分析和绘图。")
        return

    # 按基名聚合
    df = df.copy()
    df["algo_base"] = df["algorithm"].apply(_base_name)

    # ---- 汇总统计 ----
    print(f"\n{'=' * 60}")
    print(f"实验结果汇总 ({task_type})")
    print(f"{'=' * 60}")

    base_names = sorted(df["algo_base"].unique())

    for base in base_names:
        sub = df[df["algo_base"] == base]
        obj_mean, obj_std = mean_std(sub["objective"].tolist())
        rt_mean, rt_std = mean_std(sub["runtime"].tolist())
        print(f"\n  {base}:")
        print(f"    目标值: {obj_mean:.4f} ± {obj_std:.4f}")
        print(f"    运行时间: {rt_mean:.4f}s ± {rt_std:.4f}")
        print(f"    样本数: {len(sub)}")

    # ---- 箱线图：目标值 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    algo_order = sorted(base_names,
                         key=lambda a: df[df["algo_base"] == a]["objective"].mean(),
                         reverse=True)
    data_by_algo = [df[df["algo_base"] == a]["objective"].dropna().tolist()
                    for a in algo_order]
    short_labels = [_short_name(a) for a in algo_order]

    bp = ax.boxplot(data_by_algo, tick_labels=short_labels, patch_artist=True,
                     widths=0.5)
    for patch, algo in zip(bp["boxes"], algo_order):
        patch.set_facecolor(COLORS.get(algo, "#999999"))
        patch.set_alpha(0.7)

    ylabel = "Clique Size |S|" if task_type == "maximum_clique" else "Density ρ(S)"
    ax.set_ylabel(ylabel)
    ax.set_title(f"Algorithm Comparison — Objective\n{task_type}", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_ticklabels(), rotation=15, ha="right", fontsize=9)

    # ---- 箱线图：运行时间 ----
    ax = axes[1]
    rt_data = [df[df["algo_base"] == a]["runtime"].dropna().tolist()
               for a in algo_order]
    bp2 = ax.boxplot(rt_data, tick_labels=short_labels, patch_artist=True, widths=0.5)
    for patch, algo in zip(bp2["boxes"], algo_order):
        patch.set_facecolor(COLORS.get(algo, "#999999"))
        patch.set_alpha(0.7)

    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(f"Algorithm Comparison — Runtime\n{task_type}", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_ticklabels(), rotation=15, ha="right", fontsize=9)

    fig.suptitle("Experiment 2: Algorithm Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig_path = RESULTS_DIR / f"{task_type}_{tag}_boxplot.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n箱线图已保存至: {fig_path}")
    plt.close(fig)

    # ---- 按参数组合分组的柱状图 ----
    if "sample_id" in df.columns:
        plot_grouped_bars(df, task_type, tag)

    # ---- 保存汇总统计表 ----
    summary_rows = []
    for base in base_names:
        sub = df[df["algo_base"] == base]
        obj_mean, obj_std = mean_std(sub["objective"].tolist())
        rt_mean, rt_std = mean_std(sub["runtime"].tolist())
        summary_rows.append({
            "algorithm": base,
            "objective_mean": obj_mean,
            "objective_std": obj_std,
            "runtime_mean": rt_mean,
            "runtime_std": rt_std,
            "n_runs": len(sub),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / f"{task_type}_{tag}_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"汇总统计已保存至: {summary_path}")


def plot_grouped_bars(df: pd.DataFrame, task_type: str, tag: str):
    """按参数组合绘制分组柱状图。聚合到算法基名。"""
    df = df.copy()
    if "algo_base" not in df.columns:
        df["algo_base"] = df["algorithm"].apply(_base_name)

    # 按节点数n分组
    df["n_group"] = df["sample_id"].apply(
        lambda s: f"n={s.split('_')[1]}" if "_" in s else s)

    groups = sorted(df["n_group"].unique(), key=lambda g: int(g.split("=")[1])
                    if "=" in g and g.split("=")[1].isdigit() else 999)
    if len(groups) > 20:
        print(f"  参数组过多 ({len(groups)}), 跳过柱状图")
        return

    base_names = sorted(df["algo_base"].unique())
    n_groups = len(groups)
    n_algos = len(base_names)
    x = np.arange(n_groups)
    width = 0.8 / n_algos

    fig, ax = plt.subplots(figsize=(max(12, n_groups * 1.2), 6))

    for i, algo in enumerate(base_names):
        means = []
        stds = []
        for grp in groups:
            sub = df[(df["algo_base"] == algo) &
                     (df["n_group"] == grp)]
            if len(sub) > 0:
                m, s = mean_std(sub["objective"].tolist())
                means.append(m)
                stds.append(s)
            else:
                means.append(0)
                stds.append(0)

        offset = (i - n_algos / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=_short_name(algo),
               color=COLORS.get(algo, "#999999"),
               alpha=0.8, capsize=3, error_kw={"linewidth": 1})

    ylabel = "Clique Size |S|" if task_type == "maximum_clique" else "Density ρ(S)"
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Graph Size (n)")
    ax.set_title(f"Algorithm Comparison by Graph Size\n{task_type}",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=0, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = RESULTS_DIR / f"{task_type}_{tag}_grouped.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"分组柱状图已保存至: {fig_path}")
    plt.close(fig)


def _short_name(full_name: str) -> str:
    """缩短算法名称用于图表标注。"""
    mapping = {
        "ClassicalDegree": "DegGreedy",
        "ClassicalClique": "CliqueGreedy",
        "ClassicalDense": "DenseGreedy",
        "SimulatedAnnealing": "SimAnneal",
        "QuantumGuidedGreedy": "QuantumGreedy",
    }
    for key, short in mapping.items():
        if full_name.startswith(key):
            return short
    return full_name[:15]


def _base_name(full_name: str) -> str:
    """提取算法基名（去掉 run_id 后缀）。"""
    for key in ["ClassicalDegree", "ClassicalClique", "ClassicalDense",
                 "SimulatedAnnealing", "QuantumGuidedGreedy"]:
        if full_name.startswith(key):
            return key
    return full_name


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="实验二：与经典贪心算法对比（理论 §13.3）")
    parser.add_argument("--task", choices=["maximum_clique", "densest_subgraph"],
                        default="maximum_clique", help="任务类型")
    parser.add_argument("--data-source", choices=["artificial", "external"],
                        default="artificial",
                        help="数据来源: artificial(人工植入数据) / external(DIMACS等外部数据)")
    parser.add_argument("--small", action="store_true", default=True,
                        help="仅使用小规模实例 n≤50（默认开启，仅对 artificial 有效）")
    parser.add_argument("--full", action="store_true", default=False,
                        help="使用全部数据集（耗时较长）")
    parser.add_argument("--csv", type=str, default=None,
                        help="直接分析已有的 CSV 文件，跳过运行")
    args = parser.parse_args()

    if args.full:
        args.small = False

    if args.csv:
        df = pd.read_csv(args.csv)
        task = "maximum_clique" if "mc_" in args.csv else "densest_subgraph"
        analyze_and_plot(df, task, tag="loaded")
        return

    print(f"实验二：算法对比（{args.task}）")
    print(f"  数据来源: {args.data_source}")
    if args.data_source == "artificial":
        print(f"  模式: {'小规模 (n≤50)' if args.small else '全部数据集'}")
    print(f"  结果目录: {RESULTS_DIR}")

    df = run_batch_experiment(args.task, small=args.small,
                               data_source=args.data_source)
    if not df.empty:
        tag = f"{args.data_source}_small" if args.small and args.data_source == "artificial" else args.data_source
        analyze_and_plot(df, args.task, tag)

    print(f"\n实验二完成。")


if __name__ == "__main__":
    main()
