"""量子引导贪心算法（理论 §12）。

Algorithm: Quantum-guided Greedy Algorithm

算法流程图见理论文档 §12，核心步骤：
  1. 初始化种子集合 S
  2. 构造候选集合 C(S)
  3. 构造哈密顿量 H = A + λ·Σ|u⟩⟨u|
  4. 构造初始态 |ψ₀⟩
  5. 执行 CTQW 演化 |ψ(t)⟩ = exp(-iHt)|ψ₀⟩
  6. 计算量子概率 Q(v,S) = |⟨v|ψ(t)⟩|²
  7. 计算混合评分 Score(v,S) = α·Q_norm + (1-α)·R_norm
  8. 选择评分最高节点加入 S
  9. 重复直到停止条件满足

CTQW 计算通过 scipy.linalg.expm 实现矩阵指数运算。
"""

import time

import numpy as np
from scipy.linalg import expm

from .base import BaseAlgorithm, AlgorithmResult
from ..graph_utils import GraphInstance
from ..candidate_set import CandidateSetBuilder
from ..scoring import ClassicalCliqueScorer, ClassicalDenseScorer
from ..hamiltonian import construct_hamiltonian
from ..initial_state import build_initial_state


class QuantumGuidedGreedy(BaseAlgorithm):
    """量子引导贪心算法（理论 §12 完整算法流程）。

    每一轮选择都重新构造哈密顿量、初始态并演化 CTQW，
    因此量子评分会随当前部分解 S 动态变化（不是一次性静态排名）。
    """

    def __init__(self, candidate_builder: CandidateSetBuilder,
                 t: float = 1.0,
                 lam: float = 0.0,
                 init_method: str = "max_degree",
                 alpha: float = 0.5,
                 seed: int = 0,
                 name: str = "QuantumGuidedGreedy"):
        """
        参数:
            candidate_builder: 候选集合构造器。
            t: CTQW 演化时间。
            lam: 扰动强度 λ。
            init_method: 初态初始化方式（uniform / max_degree / random）。
            alpha: 混合评分权重，α=0 纯经典，α=1 纯量子。
            seed: 随机种子（仅 init_method='random' 时影响初态选择）。
        """
        super().__init__(candidate_builder, scorer=None, name=name)
        self.t = t
        self.lam = lam
        self.init_method = init_method
        self.alpha = alpha
        self.seed = seed

    def solve(self, instance: GraphInstance) -> AlgorithmResult:
        t_start = time.perf_counter()

        adjacency = instance.adjacency
        edge_set = instance.edge_set
        all_nodes = set(range(instance.num_nodes))
        n = instance.num_nodes

        # 选择经典评分器（按任务类型）
        if instance.task_type == "maximum_clique":
            classical_scorer = ClassicalCliqueScorer()
        else:
            classical_scorer = ClassicalDenseScorer()

        # 初始化种子集合（理论 §6）
        # max_degree 模式下从度数最高节点出发；其他情况留空，由候选集合首轮全开
        S: set[int] = set()
        if self.init_method == "max_degree" and len(all_nodes) > 0:
            degrees = adjacency.sum(axis=1)
            best = int(np.argmax(degrees))
            S.add(best)

        history: list[dict] = []
        iterations = 0

        # 密集子图任务的目标大小（理论 §7.2）
        # MC 任务终止于"候选集合为空"；DS 任务的候选集合定义宽松，
        # 需要显式停在 |S|=k 处。answer_size 即植入子图大小。
        target_size = None
        if instance.task_type == "densest_subgraph":
            target_size = instance.parameters.get("answer_size")

        while True:
            # 1. 构造候选集合 C(S)
            candidates = self.candidate_builder.build(
                adjacency, edge_set, S, all_nodes)
            if not candidates:
                break

            # DS 任务额外终止条件：达到目标大小
            if target_size is not None and len(S) >= target_size:
                break

            # 2-5. CTQW 演化得到节点概率分布 P_v(t)
            quantum_probs = self._compute_ctqw_probs(
                adjacency, S, n)

            # 6. 量子评分 Q(v,S) —— 仅在候选节点上取值
            q_scores = {v: quantum_probs[v] for v in candidates}

            # 7. 经典评分 R(v,S)
            r_scores = classical_scorer.score_all(candidates, S, instance)

            # 8. min-max 归一化（理论 §8.3，仅在候选集合上计算）
            q_norm = _minmax_normalize(q_scores)
            r_norm = _minmax_normalize(r_scores)

            # 9. 混合评分 Score(v,S) = α·Q_norm + (1-α)·R_norm
            combined = {
                v: self.alpha * q_norm.get(v, 0.0)
                + (1 - self.alpha) * r_norm.get(v, 0.0)
                for v in candidates
            }

            # 10. 选择最高分节点
            best_v = max(candidates, key=lambda v: combined[v])

            # 11. 更新 S
            S.add(best_v)

            history.append({
                "iteration": iterations,
                "chosen": best_v,
                "S_size": len(S),
                "candidates_size": len(candidates),
                "q_score": q_scores.get(best_v, 0.0),
                "r_score": r_scores.get(best_v, 0.0),
                "combined": combined[best_v],
            })
            iterations += 1

        runtime = time.perf_counter() - t_start
        solution = list(S)

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
            extra_params={"t": self.t, "lam": self.lam,
                          "alpha": self.alpha,
                          "init_method": self.init_method},
        )

    def _compute_ctqw_probs(self, adjacency: np.ndarray,
                            S: set[int], n: int) -> np.ndarray:
        """计算 CTQW 节点概率分布 P_v(t) = |⟨v|e^{-iHt}|ψ₀⟩|²。

        参数:
            adjacency: n×n 邻接矩阵。
            S: 当前已选节点集合（影响哈密顿量扰动项和初始态）。
            n: 节点总数。

        返回:
            长度为 n 的实数数组，probs[v] = 节点 v 的 CTQW 概率，
            满足 Σ probs[v] = 1。
        """
        # 1. 构造哈密顿量 H = A + λ·Σ_{u∈S}|u⟩⟨u|（理论 §5）
        H = construct_hamiltonian(adjacency, S, self.lam)

        # 2. 构造初始态 |ψ₀⟩（理论 §6）
        psi0 = build_initial_state(n, S, adjacency, self.init_method)

        # 3. 演化算子 U(t) = exp(-iHt)，幺正矩阵
        U = expm(-1j * H * self.t)

        # 4. 演化态 |ψ(t)⟩ = U|ψ₀⟩
        psi_t = U @ psi0

        # 5. 节点概率 P_v(t) = |ψ_v(t)|²（满足 Σ P_v = 1）
        return np.abs(psi_t) ** 2


def _minmax_normalize(scores: dict[int, float],
                       eps: float = 1e-10) -> dict[int, float]:
    """Min-max 归一化（理论 §8.3）。"""
    if not scores:
        return {}
    values = list(scores.values())
    v_min, v_max = min(values), max(values)
    denom = v_max - v_min + eps
    if denom == 0:
        return {k: 0.5 for k in scores}
    return {k: (v - v_min) / denom for k, v in scores.items()}


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
