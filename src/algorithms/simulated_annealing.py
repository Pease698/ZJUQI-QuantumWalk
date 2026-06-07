"""模拟退火算法（理论 §13.3 对照方法）。

基于 Metropolis 准则的全局随机搜索算法，作为经典启发式方法的对照。

算法要点：
  - 初始解：随机单个节点
  - 邻域操作：随机加入（保持合法性）或随机移除一个节点
  - 温度调度：指数衰减 T = T0 × cooling_rate^iter
  - 接受准则：若新解更优则接受；否则以概率 exp(-ΔE/T) 接受

由于模拟退火是全局搜索而非逐轮贪心构建，它不依赖 Scorer 接口，
但在合法性检查上复用 CandidateSetBuilder。
"""

import math
import random
import time

import numpy as np

from .base import BaseAlgorithm, AlgorithmResult
from ..graph_utils import GraphInstance
from ..candidate_set import CandidateSetBuilder


class SimulatedAnnealing(BaseAlgorithm):
    """模拟退火算法。

    在保持解合法性的前提下，通过随机扰动和温度控制的接受准则
    探索解空间，避免陷入局部最优。
    """

    def __init__(self, candidate_builder: CandidateSetBuilder,
                 T0: float = 1.0,
                 cooling_rate: float = 0.995,
                 max_iterations: int = 5000,
                 seed: int = 0,
                 name: str = "SimulatedAnnealing"):
        """
        参数:
            candidate_builder: 用于检查解合法性的候选集合构造器。
            T0: 初始温度。
            cooling_rate: 每轮温度衰减因子，∈ (0, 1)。
            max_iterations: 最大迭代次数。
            seed: 随机种子。
        """
        super().__init__(candidate_builder, scorer=None, name=name)
        self.T0 = T0
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.seed = seed

    def solve(self, instance: GraphInstance) -> AlgorithmResult:
        rng = random.Random(self.seed)
        np_rng = np.random.RandomState(self.seed)
        t_start = time.perf_counter()

        adjacency = instance.adjacency
        edge_set = instance.edge_set
        all_nodes = set(range(instance.num_nodes))
        n = instance.num_nodes

        # 初始解：随机选一个节点
        S: set[int] = set()
        init_v = rng.randint(0, n - 1)
        S.add(init_v)

        # 计算初始解的评估值
        s_list = list(S)
        if instance.task_type == "maximum_clique":
            # 初始解只有 1 个节点，必定是合法团
            best_obj = 1.0
        else:
            best_obj = _compute_density(edge_set, s_list)
        best_S = S.copy()
        best_obj = float(len(S)) if instance.task_type == "maximum_clique" \
            else _compute_density(edge_set, list(S))

        T = self.T0
        history: list[dict] = []
        iterations = 0

        for it in range(self.max_iterations):
            # 邻域操作：50% 概率加入，50% 概率移除
            if len(S) <= 1 or rng.random() < 0.5:
                # 尝试加入一个随机候选节点
                candidates = self.candidate_builder.build(
                    adjacency, edge_set, S, all_nodes)
                if candidates:
                    v = rng.choice(list(candidates))
                    S.add(v)
                else:
                    # 没有合法候选，跳过本轮
                    iterations += 1
                    continue
            else:
                # 尝试移除一个随机节点
                v = rng.choice(list(S))
                S.remove(v)

            # 计算新解的目标函数值
            s_list_new = list(S)
            if instance.task_type == "maximum_clique":
                new_obj = float(len(s_list_new))
            else:
                new_obj = _compute_density(edge_set, s_list_new)

            delta = new_obj - best_obj

            # Metropolis 准则
            if delta > 0:
                # 更优解，无条件接受
                pass
            else:
                # 温差接受：exp(-|delta|/T)
                if T > 0:
                    accept_prob = math.exp(delta / T)
                    if rng.random() >= accept_prob:
                        # 拒绝：回退操作
                        if len(S) > len(best_S):
                            S.discard(v)
                        elif len(S) < len(best_S):
                            S.add(v)
                        iterations += 1
                        continue

            # 更新最优解
            if new_obj > best_obj:
                best_obj = new_obj
                best_S = S.copy()

            # 记录历史
            if it % 100 == 0:
                history.append({
                    "iteration": it,
                    "S_size": len(S),
                    "objective": new_obj,
                    "best_objective": best_obj,
                    "T": T,
                })

            # 温度衰减
            T *= self.cooling_rate
            iterations += 1

            # 早停：温度过低
            if T < 1e-6:
                break

        runtime = time.perf_counter() - t_start

        return self._build_result(
            instance=instance,
            solution=list(best_S),
            objective=best_obj,
            runtime=runtime,
            iterations=iterations,
            history=history,
            extra_params={"T0": self.T0, "cooling_rate": self.cooling_rate,
                          "seed": self.seed},
        )


def _compute_density(edge_set: set, nodes: list[int]) -> float:
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
