"""Multi-Start 贪心算法系列（实验五，H4 验证）。

核心思想：
  实验报告 §6.4 指出 H2 失败的根本原因——CTQW 用在贪心**内部**时，初始态退化为
  种子集合均匀叠加，CTQW 信号方向与 R 评分完全重合但精度更低。
  本模块把 CTQW 放在贪心**外部**作为"种子选择器"，利用实验一已验证的
  全图均匀叠加下的全局识别能力。

算法 MultiStartCTQWGreedy（理论 H4）：
  1. ψ₀ ← (1/√n) Σ_v |v⟩                # 全图均匀叠加（§实验一已验证）
  2. H  ← A                              # S=∅，无 λ 扰动
  3. P  ← |exp(-iHt)·ψ₀|²                # CTQW 概率
  4. SeedSet ← TopK(P, K)                # 概率最高的 K 个节点
  5. for v in SeedSet:                   # 每个起点独立跑 ClassicalCliqueGreedy
       S_v ← ClassicalCliqueGreedy(G, start_node=v).solve()
  6. return argmax_v |S_v|               # 返回最大的团

为了能严格剥离"CTQW 起点选得好" vs "多起点本身就涨"两类贡献，
同一文件提供两个对照实现：
  - MultiStartRandomGreedy: 起点随机
  - MultiStartDegreeGreedy: 按度数取 Top-K
"""

import time

import numpy as np

from .base import BaseAlgorithm, AlgorithmResult
from .classical_greedy import ClassicalGreedy
from ..graph_utils import GraphInstance
from ..candidate_set import CandidateSetBuilder, CliqueCandidateSet
from ..scoring import ClassicalCliqueScorer
from ..ctqw_evolution import compute_ctqw_evolution


class _MultiStartBase(BaseAlgorithm):
    """Multi-Start 贪心的共同骨架。

    子类只需实现 _select_seeds(instance, K) -> list[int]。
    每个起点独立跑一次 ClassicalCliqueGreedy(start_node=v)，返回最大解。
    """

    def __init__(self, K: int = 5, name: str = "MultiStart",
                 seed: int = 0):
        builder = CliqueCandidateSet()
        scorer = ClassicalCliqueScorer()
        super().__init__(builder, scorer, name)
        self.K = K
        self.seed = seed

    def _select_seeds(self, instance: GraphInstance, K: int) -> list[int]:
        raise NotImplementedError

    def solve(self, instance: GraphInstance) -> AlgorithmResult:
        t_start = time.perf_counter()

        n = instance.num_nodes
        K = min(self.K, n)
        seeds = self._select_seeds(instance, K)

        best_solution: list[int] = []
        best_objective: float = -1.0
        per_seed_log: list[dict] = []
        total_iter = 0

        for v in seeds:
            inner = ClassicalGreedy(
                self.candidate_builder, ClassicalCliqueScorer(),
                name=f"_inner_seed{v}", start_node=v)
            result = inner.solve(instance)
            per_seed_log.append({
                "seed_node": v,
                "objective": result.objective,
                "solution_size": len(result.solution),
                "iterations": result.iterations,
            })
            total_iter += result.iterations
            if result.objective > best_objective:
                best_objective = result.objective
                best_solution = result.solution

        runtime = time.perf_counter() - t_start

        return self._build_result(
            instance=instance,
            solution=best_solution,
            objective=float(best_objective),
            runtime=runtime,
            iterations=total_iter,
            history=per_seed_log,
            extra_params={"K": self.K, "n_seeds_tried": len(seeds)},
        )


class MultiStartCTQWGreedy(_MultiStartBase):
    """用 CTQW 全图均匀叠加的概率分布选 Top-K 起点。

    这是 H4 假设要验证的目标算法。
    """

    def __init__(self, K: int = 5, t: float = 1.0,
                 evolution_method: str = "auto",
                 krylov_dim: int | None = None,
                 cheb_degree: int | None = None,
                 name: str | None = None, seed: int = 0):
        super().__init__(K=K, name=name or f"MultiStartCTQW(K={K})", seed=seed)
        self.t = t
        self.evolution_method = evolution_method
        self.krylov_dim = krylov_dim
        self.cheb_degree = cheb_degree

    def _select_seeds(self, instance: GraphInstance, K: int) -> list[int]:
        n = instance.num_nodes
        adjacency = instance.adjacency

        # 全图均匀叠加，避开实验报告 §6.4 揭示的种子集合均匀叠加错配
        psi0 = np.ones(n, dtype=np.complex128) / np.sqrt(n)
        H = adjacency  # S=∅、λ=0 时退化为 A

        # CTQW 演化：支持 exact / krylov / chebyshev 三种方法
        psi_t = compute_ctqw_evolution(
            H, psi0, self.t,
            method=self.evolution_method,
            krylov_dim=self.krylov_dim,
            cheb_degree=self.cheb_degree,
        )
        probs = np.abs(psi_t) ** 2

        # 概率降序取 Top-K（np.argsort 升序，所以取后 K 个倒过来）
        top_idx = np.argsort(probs)[-K:][::-1]
        return [int(v) for v in top_idx]


class MultiStartRandomGreedy(_MultiStartBase):
    """随机选 K 个起点的对照。

    用于剥离"多起点本身就涨" vs "CTQW 选起点选得好"两类贡献。
    """

    def __init__(self, K: int = 5, seed: int = 0,
                 name: str | None = None):
        super().__init__(K=K, name=name or f"MultiStartRandom(K={K})", seed=seed)

    def _select_seeds(self, instance: GraphInstance, K: int) -> list[int]:
        rng = np.random.default_rng(self.seed)
        n = instance.num_nodes
        return [int(v) for v in rng.choice(n, size=K, replace=False)]


class MultiStartDegreeGreedy(_MultiStartBase):
    """按度数取 Top-K 起点的对照。

    用于剥离"任何全局信号都行" vs "CTQW 提供了度数之外的信息"两类贡献。
    """

    def __init__(self, K: int = 5, name: str | None = None, seed: int = 0):
        super().__init__(K=K, name=name or f"MultiStartDegree(K={K})", seed=seed)

    def _select_seeds(self, instance: GraphInstance, K: int) -> list[int]:
        adjacency = instance.adjacency
        degrees = adjacency.sum(axis=1)
        top_idx = np.argsort(degrees)[-K:][::-1]
        return [int(v) for v in top_idx]
