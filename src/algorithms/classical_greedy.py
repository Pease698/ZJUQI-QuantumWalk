"""经典贪心算法（理论 §8.1）。

实现两种经典贪心变体，通过组合不同的 CandidateSetBuilder 和 Scorer 区分：

  1. 纯度数贪心（理论 §3.1 基线）：
     ClassicalGreedy + ClassicalDegreeScorer
     每轮选度数最高的候选节点。

  2. 候选子图度数贪心（理论 §8.1 最大团）：
     ClassicalGreedy + CliqueCandidateSet + ClassicalCliqueScorer
     每轮在合法候选节点中选候选子图度数最高者。

  3. 密度增量贪心（理论 §8.1 密集子图）：
     ClassicalGreedy + DenseCandidateSet + ClassicalDenseScorer
     每轮选加入后密度增量最大的候选节点。

算法终止条件：候选集合为空（无法继续扩展）。
"""

import time

import numpy as np

from .base import BaseAlgorithm, AlgorithmResult
from ..graph_utils import GraphInstance
from ..candidate_set import CandidateSetBuilder
from ..scoring import Scorer


class ClassicalGreedy(BaseAlgorithm):
    """经典贪心算法。

    每轮：
      1. 构造候选集合 C(S)
      2. 若 C(S) 为空则停止
      3. 对 C(S) 中每个节点计算评分
      4. 选择评分最高的节点加入 S

    算法行为完全由 candidate_builder 和 scorer 决定。
    """

    def __init__(self, candidate_builder: CandidateSetBuilder,
                 scorer: Scorer, name: str = "ClassicalGreedy"):
        super().__init__(candidate_builder, scorer, name)

    def solve(self, instance: GraphInstance) -> AlgorithmResult:
        t_start = time.perf_counter()

        adjacency = instance.adjacency
        edge_set = instance.edge_set
        all_nodes = set(range(instance.num_nodes))
        S: set[int] = set()
        history: list[dict] = []
        iterations = 0

        # 密集子图任务的目标大小（理论 §7.2）
        # MC 任务终止于"候选集合为空"（团扩张到极限）；
        # 但 DS 任务的候选集合定义宽松，需要显式停在 |S|=k 处。
        # answer_size 即植入子图大小，DS 数据集每个 JSON 都有此字段。
        target_size = None
        if instance.task_type == "densest_subgraph":
            target_size = instance.parameters.get("answer_size")

        while True:
            # 1. 构造候选集合
            candidates = self.candidate_builder.build(
                adjacency, edge_set, S, all_nodes)

            # 2. 终止条件 A：无候选节点（MC 任务的自然停止）
            if not candidates:
                break

            # 3. 终止条件 B：DS 任务达到目标团大小
            if target_size is not None and len(S) >= target_size:
                break

            # 4. 评分
            scores = self.scorer.score_all(candidates, S, instance)

            # 5. 选择评分最高的节点
            best_v = max(candidates, key=lambda v: scores.get(v, float("-inf")))

            # 6. 更新
            S.add(best_v)

            history.append({
                "iteration": iterations,
                "chosen": best_v,
                "S_size": len(S),
                "candidates_size": len(candidates),
                "max_score": scores.get(best_v, 0.0),
                "mean_score": float(np.mean(list(scores.values())))
                if scores else 0.0,
            })
            iterations += 1

        runtime = time.perf_counter() - t_start
        solution = list(S)

        # 计算目标函数值
        if instance.task_type == "maximum_clique":
            objective = float(len(solution))
        else:
            objective = _compute_density(edge_set, solution)

        return self._build_result(
            instance=instance,
            solution=solution,
            objective=objective,
            runtime=runtime,
            iterations=iterations,
            history=history,
        )


def _compute_density(edge_set: set, nodes: list[int]) -> float:
    """计算节点集合的子图密度 ρ(S)（理论 §2.2）。"""
    k = len(nodes)
    if k <= 1:
        return 1.0
    max_edges = k * (k - 1) / 2
    actual = 0
    for i in range(k):
        for j in range(i + 1, k):
            u, v = nodes[i], nodes[j]
            if (u, v) in edge_set or (v, u) in edge_set:
                actual += 1
    return actual / max_edges
