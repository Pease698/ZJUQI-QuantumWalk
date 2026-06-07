"""实验运行器。

负责：
  1. 遍历数据集目录，加载所有测试实例
  2. 对每个实例使用配置的算法集合运行
  3. 处理重复运行（同一实例不同随机种子）
  4. 收集结果并输出为 pandas DataFrame
"""

import json
import time
from pathlib import Path

import pandas as pd

from .graph_utils import load_instance, GraphInstance
from .algorithms.base import BaseAlgorithm, AlgorithmResult
from .config import ExperimentConfig, ensure_results_dir


def run_single_instance(instance: GraphInstance,
                        algorithms: dict[str, BaseAlgorithm],
                        config: ExperimentConfig) -> list[dict]:
    """对单个测试实例运行所有算法，每种重复 repeat_runs 次。

    参数:
        instance: 加载好的测试实例。
        algorithms: {算法名称: 算法对象} 的映射。
        config: 实验配置。

    返回:
        扁平化的结果字典列表，每行代表一次算法运行。
    """
    rows = []
    for algo_name, algo in algorithms.items():
        for run_id in range(config.repeat_runs):
            # 设置不同的 seed
            algo_run = _clone_with_seed(algo, algo_name, config, run_id)
            result = algo_run.solve(instance)
            row = result.to_dict()
            row["run_id"] = run_id
            row["seed"] = config.seed + run_id
            rows.append(row)
    return rows


def run_experiment(data_dir: str | Path,
                   algorithms: dict[str, BaseAlgorithm],
                   config: ExperimentConfig,
                   output_csv: str | None = None) -> pd.DataFrame:
    """遍历数据目录，对每个 JSON 实例运行所有算法。

    参数:
        data_dir: 包含 JSON 文件的数据目录路径。
        algorithms: {算法名称: 算法对象} 的映射。
        config: 实验配置（其中 repeat_runs 控制每实例重复次数）。
        output_csv: 可选，结果保存的 CSV 路径。

    返回:
        包含所有运行结果的 DataFrame。
    """
    data_path = Path(data_dir)
    json_files = sorted(data_path.glob("*.json"))
    if not json_files:
        print(f"警告: {data_dir} 中未找到 JSON 文件")
        return pd.DataFrame()

    all_rows = []
    total = len(json_files)

    print(f"实验数据目录: {data_dir}")
    print(f"测试实例数:   {total}")
    print(f"算法数量:     {len(algorithms)}")
    print(f"每实例重复:   {config.repeat_runs}")
    print(f"总运行次数:   {total * len(algorithms) * config.repeat_runs}")
    print("-" * 50)

    t_start = time.perf_counter()

    for idx, fpath in enumerate(json_files):
        instance = load_instance(fpath)
        rows = run_single_instance(instance, algorithms, config)
        all_rows.extend(rows)

        if (idx + 1) % 10 == 0 or idx == total - 1:
            elapsed = time.perf_counter() - t_start
            print(f"  [{idx + 1:4d}/{total}] 已完成, "
                  f"耗时 {elapsed:.1f}s, "
                  f"已收集 {len(all_rows)} 条记录")

    df = pd.DataFrame(all_rows)
    elapsed_total = time.perf_counter() - t_start
    print(f"全部完成，总耗时 {elapsed_total:.1f}s")
    print(f"收集到 {len(df)} 条运行记录")

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"结果已保存至: {output_path}")

    return df


def run_all_datasets(task_type: str,
                     algorithms: dict[str, BaseAlgorithm],
                     config: ExperimentConfig,
                     output_dir: str | None = None) -> pd.DataFrame:
    """遍历所有参数组合的数据目录，运行实验。

    参数:
        task_type: "maximum_clique" 或 "densest_subgraph"
        algorithms: 算法映射。
        config: 实验配置。
        output_dir: 可选，保存 CSV 的目录。

    返回:
        汇总的 DataFrame。
    """
    from .config import get_data_dirs
    dirs = get_data_dirs(task_type)
    if not dirs:
        print(f"警告: 未找到 {task_type} 的数据目录")
        return pd.DataFrame()

    all_dfs = []
    for data_dir in dirs:
        csv_name = None
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            csv_name = str(out_path / f"{data_dir.name}.csv")
        df = run_experiment(data_dir, algorithms, config, output_csv=csv_name)
        if not df.empty:
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        if output_dir:
            combined_path = Path(output_dir) / f"{task_type}_all.csv"
            combined.to_csv(combined_path, index=False, encoding="utf-8")
            print(f"汇总结果已保存至: {combined_path}")
        return combined
    return pd.DataFrame()


def _clone_with_seed(algo: BaseAlgorithm, algo_name: str,
                     config: ExperimentConfig, run_id: int) -> BaseAlgorithm:
    """根据算法名称创建带有特定 seed 的副本。

    这是一种轻量级的"克隆"方法，避免引入 copy.deepcopy 的复杂性。
    由于算法对象通常足够轻量，直接重新构造即可。
    """
    from .candidate_set import CliqueCandidateSet, DenseCandidateSet
    from .scoring import (ClassicalCliqueScorer, ClassicalDenseScorer,
                          ClassicalDegreeScorer, HybridScorer, QuantumScorer)
    from .algorithms.classical_greedy import ClassicalGreedy
    from .algorithms.simulated_annealing import SimulatedAnnealing
    from .algorithms.quantum_greedy import QuantumGuidedGreedy

    seed = config.seed + run_id

    # 根据算法名称和配置重建
    if algo_name.startswith("ClassicalDegree"):
        builder = CliqueCandidateSet()
        return ClassicalGreedy(
            builder, ClassicalDegreeScorer(), name=f"ClassicalDegree(run={run_id})")

    elif algo_name.startswith("ClassicalClique"):
        builder = CliqueCandidateSet()
        return ClassicalGreedy(
            builder, ClassicalCliqueScorer(), name=f"ClassicalClique(run={run_id})")

    elif algo_name.startswith("ClassicalDense"):
        builder = DenseCandidateSet()
        return ClassicalGreedy(
            builder, ClassicalDenseScorer(), name=f"ClassicalDense(run={run_id})")

    elif algo_name.startswith("SimulatedAnnealing"):
        builder = CliqueCandidateSet()
        return SimulatedAnnealing(
            builder, seed=seed, name=f"SimulatedAnnealing(run={run_id})")

    elif algo_name.startswith("QuantumGuidedGreedy"):
        builder = CliqueCandidateSet()
        return QuantumGuidedGreedy(
            builder, t=config.t, lam=config.lam,
            init_method=config.init_method, alpha=config.alpha,
            seed=seed, name=f"QuantumGuidedGreedy(run={run_id})")

    else:
        # 回退：返回原对象（不修改 seed）
        return algo
