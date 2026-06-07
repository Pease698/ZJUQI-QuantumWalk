#!/usr/bin/env python3
"""通用实验运行器 —— 自由组合数据源、算法、评分方法和参数。

与四个预设实验脚本的关系：
  - exp1~exp4 是固定流程的"一键出图"方案
  - run_experiment.py 是灵活的"自由搭建"工具

功能:
  1. 自由选择数据来源：人工数据集 / 外部数据集 / 指定目录 / 指定文件
  2. 自由选择算法组合和评分方法
  3. 自动根据数据来源设置合理的重复次数：
     - 人工数据：每个实例 4 次（每组 5×4=20 数据点，符合理论 §13.1）
     - 外部数据：每个实例 10 次（单实例无跨图方差，需更多运行内统计）
     - 文件/目录模式：默认 4 次，可手动覆盖
  4. 统一输出 CSV + 箱线图

用法示例:
  # 人工数据上比较所有经典算法（小规模）
  python3 experiments/run_experiment.py \\
      --task maximum_clique --data-source artificial --small \\
      --algorithms ClassicalDegree ClassicalClique SimulatedAnnealing

  # 外部数据上比较所有算法
  python3 experiments/run_experiment.py \\
      --task maximum_clique --data-source external \\
      --algorithms ClassicalDegree ClassicalClique SimulatedAnnealing QuantumGuidedGreedy

  # 指定目录
  python3 experiments/run_experiment.py \\
      --task maximum_clique --data-source dir \\
      --data-path datasets/data/artificial/maximum_clique/mc_n100_p02_k10 \\
      --algorithms ClassicalClique SimulatedAnnealing

  # 单文件，自定义量子参数
  python3 experiments/run_experiment.py \\
      --task maximum_clique --data-source file \\
      --data-path datasets/data/external/maximum_clique/ext_mc_C250-9.json \\
      --algorithms QuantumGuidedGreedy --scorer hybrid --alpha 0.7 --t 2.0 --lam 1.0
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

from src.graph_utils import load_instance, GraphInstance
from src.candidate_set import CliqueCandidateSet, DenseCandidateSet
from src.scoring import (
    ClassicalCliqueScorer, ClassicalDenseScorer, ClassicalDegreeScorer,
    QuantumScorer, HybridScorer, Scorer,
)
from src.algorithms.classical_greedy import ClassicalGreedy
from src.algorithms.simulated_annealing import SimulatedAnnealing
from src.algorithms.quantum_greedy import QuantumGuidedGreedy
from src.algorithms.base import BaseAlgorithm
from src.metrics import mean_std
from src.config import get_data_dirs, DATA_DIR, ensure_results_dir

# ============================================================
# 常量
# ============================================================

DEFAULT_REPEAT_ARTIFICIAL = 4     # 人工数据：5实例×4次=20数据点
DEFAULT_REPEAT_EXTERNAL = 10      # 外部数据：无跨图方差，需更多运行内统计
DEFAULT_REPEAT_OTHER = 4

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "run_experiment"

ALGO_COLORS = {
    "ClassicalDegree": "#1f77b4",
    "ClassicalClique": "#ff7f0e",
    "ClassicalDense": "#d62728",
    "SimulatedAnnealing": "#2ca02c",
    "QuantumGuidedGreedy": "#9467bd",
}


# ============================================================
# 数据发现
# ============================================================

def discover_instances(data_source: str, task_type: str,
                        data_path: str | None = None,
                        small: bool = False) -> list[tuple[GraphInstance, str]]:
    """根据数据来源发现所有测试实例。

    参数:
        data_source: "artificial" | "external" | "dir" | "file"
        task_type: "maximum_clique" | "densest_subgraph"
        data_path: dir/file 模式下的路径
        small: 仅 artificial 模式有效，过滤 n≤50

    返回:
        [(GraphInstance, source_label), ...] 列表。
        source_label 用于结果分组（参数组名或数据集名）。
    """
    instances = []

    if data_source == "artificial":
        dirs = get_data_dirs(task_type)
        if small:
            dirs = [d for d in dirs if _extract_n(d.name) <= 50]
            print(f"人工数据（小规模 n≤50）：筛选到 {len(dirs)} 个参数组目录")
        else:
            print(f"人工数据：{len(dirs)} 个参数组目录")

        for d in dirs:
            json_files = sorted(d.glob("*.json"))
            for fpath in json_files:
                instances.append((load_instance(fpath), d.name))

    elif data_source == "external":
        ext_dir = DATA_DIR / "external" / task_type
        if not ext_dir.is_dir():
            print(f"错误: 外部数据目录不存在: {ext_dir}")
            print(f"请先运行: cd datasets && python -m converters.convert_dimacs")
            return []

        json_files = sorted(ext_dir.glob("*.json"))
        print(f"外部数据：{len(json_files)} 个实例")

        for fpath in json_files:
            # 用数据集名（去掉 ext_mc_/ext_ds_ 前缀和 .json 后缀）作为标签
            stem = fpath.stem
            if stem.startswith("ext_mc_"):
                label = stem[7:]
            elif stem.startswith("ext_ds_"):
                label = stem[7:]
            else:
                label = stem
            instances.append((load_instance(fpath), label))

    elif data_source == "dir":
        dir_path = Path(data_path)
        if not dir_path.is_dir():
            print(f"错误: 目录不存在: {dir_path}")
            return []

        json_files = sorted(dir_path.glob("*.json"))
        print(f"指定目录：{len(json_files)} 个实例 ({dir_path})")

        for fpath in json_files:
            instances.append((load_instance(fpath), dir_path.name))

    elif data_source == "file":
        file_path = Path(data_path)
        if not file_path.is_file():
            print(f"错误: 文件不存在: {file_path}")
            return []

        inst = load_instance(file_path)
        instances.append((inst, inst.sample_id))
        print(f"单文件：{inst.sample_id}")

    return instances


# ============================================================
# 算法构造
# ============================================================

def build_algorithms(algo_names: list[str], task_type: str,
                      scorer_type: str = "classical",
                      t: float = 1.0, lam: float = 0.5,
                      init_method: str = "max_degree",
                      alpha: float = 0.5) -> dict[str, BaseAlgorithm]:
    """根据名称列表构造算法对象。

    参数:
        algo_names: 算法名称列表。
        task_type: 任务类型。
        scorer_type: QuantumGuidedGreedy 使用的评分方式。
            "classical" — 纯经典评分 (alpha=0)
            "degree"   — 纯度数评分
            "quantum"  — 纯量子评分 (alpha=1)
            "hybrid"   — 混合评分
        t, lam, init_method, alpha: 量子相关参数。

    返回:
        {算法名称: 算法对象} 字典。
    """
    if task_type == "maximum_clique":
        builder = CliqueCandidateSet()
        default_classical = ClassicalCliqueScorer()
    else:
        builder = DenseCandidateSet()
        default_classical = ClassicalDenseScorer()

    # 为 QuantumGuidedGreedy 选择 scorer
    if scorer_type == "classical":
        q_scorer = default_classical
        q_alpha = 0.0
    elif scorer_type == "degree":
        q_scorer = ClassicalDegreeScorer()
        q_alpha = 0.0
    elif scorer_type == "quantum":
        q_scorer = QuantumScorer(t=t, lam=lam, init_method=init_method)
        q_alpha = 1.0
    elif scorer_type == "hybrid":
        quantum_scorer = QuantumScorer(t=t, lam=lam, init_method=init_method)
        q_scorer = HybridScorer(quantum_scorer, default_classical, alpha=alpha)
        q_alpha = alpha
    else:
        raise ValueError(f"不支持的 scorer 类型: {scorer_type}")

    algorithms = {}
    for name in algo_names:
        if name == "ClassicalDegree":
            algorithms[name] = ClassicalGreedy(
                builder, ClassicalDegreeScorer(), name="ClassicalDegree")
        elif name == "ClassicalClique":
            algorithms[name] = ClassicalGreedy(
                builder, ClassicalCliqueScorer(), name="ClassicalClique")
        elif name == "ClassicalDense":
            algorithms[name] = ClassicalGreedy(
                builder, ClassicalDenseScorer(), name="ClassicalDense")
        elif name == "SimulatedAnnealing":
            algorithms[name] = SimulatedAnnealing(
                builder, max_iterations=2000, name="SimulatedAnnealing")
        elif name == "QuantumGuidedGreedy":
            algorithms[name] = QuantumGuidedGreedy(
                builder, t=t, lam=lam, init_method=init_method,
                alpha=q_alpha, name="QuantumGuidedGreedy")
        else:
            print(f"警告: 未知算法 '{name}'，已跳过")

    return algorithms


# ============================================================
# 运行逻辑
# ============================================================

def run_experiment(instances: list[tuple[GraphInstance, str]],
                    algorithms: dict[str, BaseAlgorithm],
                    repeat: int,
                    base_seed: int = 0) -> pd.DataFrame:
    """对一组实例运行所有算法。

    参数:
        instances: [(GraphInstance, label), ...]
        algorithms: {name: algo} 映射。
        repeat: 每个实例的重复次数。
        base_seed: 基础随机种子。

    返回:
        包含所有运行结果的 DataFrame。
    """
    all_rows = []
    total = len(instances) * len(algorithms) * repeat

    print(f"\n{'=' * 60}")
    print(f"实验配置")
    print(f"  实例数:       {len(instances)}")
    print(f"  算法数:       {len(algorithms)} ({', '.join(algorithms.keys())})")
    print(f"  每实例重复:   {repeat}")
    print(f"  总运行次数:   {total}")
    print(f"{'=' * 60}\n")

    t_start = time.perf_counter()
    count = 0

    for inst, label in instances:
        for algo_name, algo_template in algorithms.items():
            for run_id in range(repeat):
                seed = base_seed + run_id
                algo = _rebuild_with_seed(algo_template, seed)
                result = algo.solve(inst)
                row = result.to_dict()
                row["source_label"] = label
                row["run_id"] = run_id
                row["seed"] = seed
                all_rows.append(row)
                count += 1

        elapsed = time.perf_counter() - t_start
        if count % max(total // 10, 1) == 0 or count == total:
            print(f"  [{count}/{total}] 已完成, 耗时 {elapsed:.0f}s")

    df = pd.DataFrame(all_rows)
    elapsed_total = time.perf_counter() - t_start
    print(f"\n全部完成，总耗时 {elapsed_total:.0f}s, 共 {len(df)} 条记录")
    return df


def _rebuild_with_seed(template: BaseAlgorithm, seed: int) -> BaseAlgorithm:
    """根据模板创建带新 seed 的副本。"""
    name = template.name
    builder = template.candidate_builder

    if isinstance(template, ClassicalGreedy):
        return ClassicalGreedy(builder, template.scorer,
                               name=f"{name}(run={seed})")

    elif isinstance(template, SimulatedAnnealing):
        return SimulatedAnnealing(
            builder, seed=seed,
            T0=template.T0, cooling_rate=template.cooling_rate,
            max_iterations=template.max_iterations,
            name=f"{name}(run={seed})")

    elif isinstance(template, QuantumGuidedGreedy):
        return QuantumGuidedGreedy(
            builder, t=template.t, lam=template.lam,
            alpha=template.alpha, init_method=template.init_method,
            seed=seed, name=f"{name}(run={seed})")

    return template


# ============================================================
# 结果汇总与可视化
# ============================================================

def summarize(df: pd.DataFrame, task_type: str, output_dir: Path,
               no_plot: bool = False):
    """生成汇总统计和图表。

    产出:
      - summary.csv: 按算法的汇总统计
      - boxplot_objective.png: 目标值箱线图
      - boxplot_runtime.png: 运行时间箱线图
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print("DataFrame 为空，跳过汇总。")
        return

    # 按基名聚合
    df = df.copy()
    df["algo_base"] = df["algorithm"].apply(
        lambda a: a.split("(run=")[0] if "(run=" in a else a)

    base_names = sorted(df["algo_base"].unique())

    # ---- 控制台汇总 ----
    print(f"\n{'=' * 60}")
    print(f"实验结果汇总 ({task_type})")
    print(f"{'=' * 60}")

    ylabel = "Clique Size |S|" if task_type == "maximum_clique" else "Density ρ(S)"

    for base in base_names:
        sub = df[df["algo_base"] == base]
        obj_mean, obj_std = mean_std(sub["objective"].tolist())
        rt_mean, rt_std = mean_std(sub["runtime"].tolist())
        print(f"\n  {base}:")
        print(f"    {ylabel}: {obj_mean:.4f} ± {obj_std:.4f}")
        print(f"    运行时间: {rt_mean:.4f}s ± {rt_std:.4f}")
        print(f"    样本数: {len(sub)}")

    # ---- 保存汇总 CSV ----
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
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\n汇总统计已保存至: {summary_path}")

    # ---- 保存完整结果 ----
    full_path = output_dir / "full_results.csv"
    df.to_csv(full_path, index=False, encoding="utf-8")
    print(f"完整结果已保存至: {full_path}")

    if no_plot:
        return

    # ---- 箱线图 ----
    algo_order = sorted(base_names,
                         key=lambda a: df[df["algo_base"] == a]["objective"].mean(),
                         reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 目标值
    ax = axes[0]
    data_obj = [df[df["algo_base"] == a]["objective"].dropna().tolist()
                for a in algo_order]
    bp = ax.boxplot(data_obj, tick_labels=algo_order, patch_artist=True, widths=0.5)
    for patch, algo in zip(bp["boxes"], algo_order):
        patch.set_facecolor(ALGO_COLORS.get(algo, "#999999"))
        patch.set_alpha(0.7)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Objective — {task_type}", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_ticklabels(), rotation=15, ha="right", fontsize=9)

    # 运行时间
    ax = axes[1]
    data_rt = [df[df["algo_base"] == a]["runtime"].dropna().tolist()
               for a in algo_order]
    bp2 = ax.boxplot(data_rt, tick_labels=algo_order, patch_artist=True, widths=0.5)
    for patch, algo in zip(bp2["boxes"], algo_order):
        patch.set_facecolor(ALGO_COLORS.get(algo, "#999999"))
        patch.set_alpha(0.7)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(f"Runtime — {task_type}", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_ticklabels(), rotation=15, ha="right", fontsize=9)

    fig.suptitle("Experiment Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = output_dir / "boxplot.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"箱线图已保存至: {fig_path}")
    plt.close(fig)


# ============================================================
# 辅助函数
# ============================================================

def _extract_n(dirname: str) -> int:
    import re
    m = re.search(r'_n(\d+)_', dirname)
    return int(m.group(1)) if m else 999


def _determine_repeat(data_source: str, user_repeat: int | None) -> int:
    """确定合理的重复次数。"""
    if user_repeat is not None:
        return user_repeat

    if data_source == "artificial":
        return DEFAULT_REPEAT_ARTIFICIAL
    elif data_source == "external":
        print(f"外部数据默认重复 {DEFAULT_REPEAT_EXTERNAL} 次 "
              f"(单实例无跨图方差，需更多运行内统计)")
        return DEFAULT_REPEAT_EXTERNAL
    else:
        return DEFAULT_REPEAT_OTHER


def _default_output_name(task_type: str, data_source: str) -> str:
    """生成默认的输出目录名。"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{task_type}_{data_source}_{timestamp}"


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="通用实验运行器 — 自由组合数据源、算法、评分方法和参数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 人工数据小规模实验
  %(prog)s --task maximum_clique --data-source artificial --small \\
      --algorithms ClassicalDegree ClassicalClique SimulatedAnnealing

  # 外部数据对比实验（仅实验二）
  %(prog)s --task maximum_clique --data-source external \\
      --algorithms ClassicalDegree ClassicalClique SimulatedAnnealing QuantumGuidedGreedy

  # 指定目录
  %(prog)s --task maximum_clique --data-source dir \\
      --data-path datasets/data/artificial/maximum_clique/mc_n100_p02_k10 \\
      --algorithms ClassicalClique

  # 单文件 + 自定义量子参数
  %(prog)s --task maximum_clique --data-source file \\
      --data-path datasets/data/external/maximum_clique/ext_mc_C250-9.json \\
      --algorithms QuantumGuidedGreedy --scorer hybrid --alpha 0.7 --t 2.0
        """,
    )

    # ---- 数据参数 ----
    parser.add_argument("--task", required=True,
                        choices=["maximum_clique", "densest_subgraph"],
                        help="任务类型")
    parser.add_argument("--data-source", required=True,
                        choices=["artificial", "external", "dir", "file"],
                        help="数据来源: artificial(全部人工) / external(全部外部) / "
                             "dir(指定目录) / file(指定文件)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="当 --data-source=dir 或 file 时，指定路径")
    parser.add_argument("--small", action="store_true", default=False,
                        help="仅在 --data-source=artificial 时有效，过滤 n≤50")

    # ---- 算法参数 ----
    parser.add_argument("--algorithms", nargs="+", required=True,
                        choices=["ClassicalDegree", "ClassicalClique",
                                 "ClassicalDense", "SimulatedAnnealing",
                                 "QuantumGuidedGreedy"],
                        help="要运行的算法列表")
    parser.add_argument("--scorer", type=str, default="hybrid",
                        choices=["classical", "degree", "quantum", "hybrid"],
                        help="QuantumGuidedGreedy 使用的评分方式 (默认 hybrid)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="混合权重 α∈[0,1], 仅在 scorer=hybrid 时有效 (默认 0.5)")
    parser.add_argument("--t", type=float, default=1.0,
                        help="CTQW 演化时间 (默认 1.0)")
    parser.add_argument("--lam", type=float, default=0.5,
                        help="种子扰动强度 λ (默认 0.5)")
    parser.add_argument("--init", type=str, default="max_degree",
                        choices=["uniform", "max_degree", "random"],
                        help="初态初始化方式 (默认 max_degree)")

    # ---- 运行参数 ----
    parser.add_argument("--repeat", type=int, default=None,
                        help="每实例重复次数 (默认: artificial=4, external=10, "
                             "dir/file=4)")
    parser.add_argument("--seed", type=int, default=0,
                        help="基础随机种子 (默认 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出目录 (默认 results/run_experiment/<自动命名>)")
    parser.add_argument("--no-plot", action="store_true", default=False,
                        help="不生成图表，仅输出 CSV")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="打印详细运行信息")

    args = parser.parse_args()

    # ---- 参数校验 ----
    if args.data_source in ("dir", "file") and not args.data_path:
        parser.error(f"--data-source={args.data_source} 需要 --data-path")

    if not 0.0 <= args.alpha <= 1.0:
        parser.error(f"--alpha 必须在 [0,1] 内，当前为 {args.alpha}")

    # ---- 确定重复次数 ----
    repeat = _determine_repeat(args.data_source, args.repeat)

    # ---- 发现实例 ----
    print(f"通用实验运行器")
    print(f"  任务: {args.task}")
    print(f"  数据来源: {args.data_source}")

    instances = discover_instances(
        args.data_source, args.task, args.data_path, args.small)
    if not instances:
        print("未发现任何测试实例，退出。")
        return

    # ---- 构造算法 ----
    print(f"  算法: {', '.join(args.algorithms)}")
    print(f"  评分方式: {args.scorer}" +
          (f" (α={args.alpha})" if args.scorer == "hybrid" else ""))
    print(f"  CTQW 参数: t={args.t}, λ={args.lam}, init={args.init}")
    print(f"  重复次数: {repeat}")

    algorithms = build_algorithms(
        args.algorithms, args.task,
        scorer_type=args.scorer,
        t=args.t, lam=args.lam, init_method=args.init, alpha=args.alpha)

    if not algorithms:
        print("未构造出任何有效算法，退出。")
        return

    # ---- 运行实验 ----
    df = run_experiment(instances, algorithms, repeat, base_seed=args.seed)

    # ---- 输出结果 ----
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = RESULTS_DIR / _default_output_name(args.task, args.data_source)

    summarize(df, args.task, output_dir, no_plot=args.no_plot)

    print(f"\n实验完成。结果目录: {output_dir}")


if __name__ == "__main__":
    main()
