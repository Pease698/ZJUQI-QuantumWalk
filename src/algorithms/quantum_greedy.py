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

当前状态：CTQW 计算模块（步骤 3-6）为占位实现。
待 CTQW 模块完成后，替换 __ctqw_probabilities 方法即可。
"""

import time

import numpy as np

from .base import BaseAlgorithm, AlgorithmResult
from ..graph_utils import GraphInstance
from ..candidate_set import CandidateSetBuilder
from ..scoring import Scorer, QuantumScorer, ClassicalCliqueScorer, \
    ClassicalDenseScorer
from ..hamiltonian import construct_hamiltonian
from ..initial_state import build_initial_state


class QuantumGuidedGreedy(BaseAlgorithm):
    """量子引导贪心算法。

    理论 §12 的完整算法流程。当前 CTQW 演化部分为占位实现，
    使用随机概率替代真实量子概率，保证框架可以端到端运行。
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
            init_method: 初态初始化方式。
            alpha: 混合评分权重。
            seed: 随机种子（占位实现使用）。
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

        # 选择经典评分器
        if instance.task_type == "maximum_clique":
            classical_scorer = ClassicalCliqueScorer()
        else:
            classical_scorer = ClassicalDenseScorer()

        quantum_scorer = QuantumScorer(
            t=self.t, lam=self.lam,
            init_method=self.init_method, seed=self.seed)

        # 初始化种子集合（理论 §6）
        S: set[int] = set()
        if self.init_method == "max_degree" and len(all_nodes) > 0:
            degrees = adjacency.sum(axis=1)
            best = int(np.argmax(degrees))
            S.add(best)

        history: list[dict] = []
        iterations = 0
        rng = np.random.RandomState(self.seed)

        while True:
            # 1. 构造候选集合 C(S)
            candidates = self.candidate_builder.build(
                adjacency, edge_set, S, all_nodes)
            if not candidates:
                break

            # 2. 构造哈密顿量 H（理论 §5）
            H = construct_hamiltonian(adjacency, S, self.lam)

            # 3. 构造初始态 |ψ₀⟩（理论 §6）
            psi0 = build_initial_state(n, S, adjacency, self.init_method)

            # 4-5. CTQW 演化 + 量子概率计算
            # TODO: 替换为真实的 exp(-iHt)|ψ₀⟩ 计算
            quantum_probs = self._placeholder_ctqw(H, psi0, n, rng)

            # 6. 计算量子评分 Q(v,S)
            q_scores = {v: quantum_probs[v] for v in candidates}

            # 7. 计算经典评分 R(v,S)
            r_scores = classical_scorer.score_all(candidates, S, instance)

            # 8. 归一化 + 混合评分 Score(v,S) = α·Q_norm + (1-α)·R_norm
            q_norm = _minmax_normalize(q_scores)
            r_norm = _minmax_normalize(r_scores)

            combined = {
                v: self.alpha * q_norm.get(v, 0.0)
                + (1 - self.alpha) * r_norm.get(v, 0.0)
                for v in candidates
            }

            # 9. 选择最高分节点
            best_v = max(candidates, key=lambda v: combined[v])

            # 10. 更新
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

    @staticmethod
    def _placeholder_ctqw(H: np.ndarray, psi0: np.ndarray,
                          n: int, rng: np.random.RandomState) -> np.ndarray:
        """CTQW 占位实现：返回均匀随机概率分布。

        TODO: 替换为 |ψ(t)⟩ = scipy.linalg.expm(-1j * H * t) @ psi0
              然后 P_v = |⟨v|ψ(t)⟩|²
        """
        probs = rng.rand(n)
        return probs / probs.sum()


def _minmax_normalize(scores: dict[int, float],
                       eps: float = 1e-10) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    v_min, v_max = min(values), max(values)
    denom = v_max - v_min + eps
    if denom == 0:
        return {k: 0.5 for k in scores}
    return {k: (v - v_min) / denom for k, v in scores.items()}


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
