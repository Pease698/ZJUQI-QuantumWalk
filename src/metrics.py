"""评价指标（理论 §3, §13）。

提供实验评估所需的各类指标计算：
  - compute_ratio: CTQW 概率分布中的目标/背景比率（验证假设一）
  - evaluate_solution: 对单个解计算目标函数值
  - success_rate: 多次运行的成功率
  - mean_std: 均值和标准差
"""

import numpy as np


def compute_ratio(target_probs: list[float],
                  background_probs: list[float]) -> float:
    """计算目标区域与背景区域的平均概率比率（理论 §3.1 假设一）。

    Ratio = Mean(P_v, v∈S_target) / Mean(P_v, v∉S_target)

    若 Ratio > 1，说明 CTQW 概率在目标区域存在集中。

    参数:
        target_probs: 目标区域节点的概率列表。
        background_probs: 背景区域节点的概率列表。

    返回:
        比率值。
    """
    target_mean = float(np.mean(target_probs)) if target_probs else 0.0
    bg_mean = float(np.mean(background_probs)) if background_probs else 0.0
    if bg_mean == 0.0:
        return float("inf") if target_mean > 0 else 0.0
    return target_mean / bg_mean


def evaluate_solution(instance, solution: list[int]) -> float:
    """对算法输出的解计算目标函数值。

    参数:
        instance: GraphInstance 对象。
        solution: 算法选出的节点列表。

    返回:
        最大团任务返回团大小 |S|，密集子图任务返回密度 ρ(S)。
    """
    if not solution:
        return 0.0

    if instance.task_type == "maximum_clique":
        return float(len(solution))

    # densest_subgraph
    edge_set = instance.edge_set
    k = len(solution)
    if k <= 1:
        return 1.0
    max_edges = k * (k - 1) / 2
    actual = 0
    for i in range(k):
        for j in range(i + 1, k):
            u, v = solution[i], solution[j]
            if (u, v) in edge_set or (v, u) in edge_set:
                actual += 1
    return actual / max_edges


def success_rate(results: list[float], threshold: float) -> float:
    """计算解质量达到阈值的成功率。

    成功定义为 objective ≥ threshold。

    参数:
        results: 多次运行的目标函数值。
        threshold: 成功阈值。

    返回:
        成功率 ∈ [0, 1]。
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r >= threshold) / len(results)


def mean_std(values: list[float]) -> tuple[float, float]:
    """计算均值和标准差。

    返回:
        (mean, std) 元组。
    """
    arr = np.array(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 \
        else 0.0


def aggregate_results(results: list[dict]) -> dict:
    """对多次运行结果进行统计汇总。

    参数:
        results: 多次运行的结果字典列表，每个字典应包含 "objective" 键。

    返回:
        包含 mean、std、min、max、success_count 等统计量的字典。
    """
    objectives = [r["objective"] for r in results]
    runtimes = [r.get("runtime", 0.0) for r in results]

    obj_mean, obj_std = mean_std(objectives)
    rt_mean, rt_std = mean_std(runtimes)

    return {
        "n_runs": len(results),
        "objective_mean": obj_mean,
        "objective_std": obj_std,
        "objective_min": float(np.min(objectives)) if objectives else 0.0,
        "objective_max": float(np.max(objectives)) if objectives else 0.0,
        "runtime_mean": rt_mean,
        "runtime_std": rt_std,
    }
