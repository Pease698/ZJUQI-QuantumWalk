"""评分函数（理论 §8）。

本项目采用三类评分函数，直接对应实验二（算法对比）和实验三（消融实验）的需求：

  8.1  经典贪心评分 — R(v,S)，基于度数、密度增量等局部结构指标
  8.2  量子概率评分 — Q(v,S)，由 CTQW 演化产生的节点概率分布
  8.3  混合评分      — α·Q_norm + (1-α)·R_norm，融合全局与局部信号

设计要点：
  - Scorer 是统一接口，所有评分函数实现 score() 和 score_all()
  - HybridScorer 是组合器，接受任意两个 Scorer + alpha 权重
  - 归一化在 HybridScorer 内部完成（min-max，仅在候选集合上计算）
  - QuantumScorer 当前为占位实现，待 CTQW 模块完成后替换
"""

from abc import ABC, abstractmethod

import numpy as np


class Scorer(ABC):
    """评分函数抽象基类。

    每个 Scorer 对一个候选节点给出评分；score_all 批量评估候选集合。
    """

    @abstractmethod
    def score(self, v: int, S: set[int], instance) -> float:
        """计算单个候选节点 v 的评分。"""
        ...

    def score_all(self, candidates: set[int], S: set[int],
                  instance) -> dict[int, float]:
        """对候选集合中所有节点计算评分。

        默认逐节点调用 score()；子类可覆盖为批量计算。
        """
        return {v: self.score(v, S, instance) for v in candidates}

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ============================================================
# 经典评分函数
# ============================================================

class ClassicalCliqueScorer(Scorer):
    """最大团任务的经典评分（理论 §8.1）。

    使用候选子图度数：R(v,S) = deg_{C(S)}(v)

    原因：最大团任务中所有合法候选都满足 R_connection(v,S)=1，
    无法区分优劣。因此改用候选节点在候选诱导子图中的度数，
    衡量选择 v 后还有多少节点可能继续加入当前团。
    """

    def score(self, v: int, S: set[int], instance) -> float:
        # 需要在候选集合上计算，由 score_all 重载
        raise NotImplementedError("请使用 score_all 批量计算")

    def score_all(self, candidates: set[int], S: set[int],
                  instance) -> dict[int, float]:
        edge_set = instance.edge_set
        scores = {}
        for v in candidates:
            # v 在候选集合诱导子图中的度数
            deg = sum(
                1 for u in candidates
                if u != v and ((u, v) in edge_set or (v, u) in edge_set)
            )
            scores[v] = float(deg)
        return scores

    @property
    def name(self) -> str:
        return "ClassicalClique"


class ClassicalDenseScorer(Scorer):
    """密集子图任务的经典评分（理论 §8.1）。

    使用密度增量：R(v,S) = ρ(S∪{v}) - ρ(S)

    直接对应目标函数，增量越大说明 v 对提升子图密度贡献越大。
    """

    def score(self, v: int, S: set[int], instance) -> float:
        if not S:
            return 1.0
        s_list = list(S)
        s_size = len(s_list)
        edge_set = instance.edge_set

        current_edges = sum(
            1 for i in range(s_size) for j in range(i + 1, s_size)
            if (s_list[i], s_list[j]) in edge_set
            or (s_list[j], s_list[i]) in edge_set
        )
        current_max = s_size * (s_size - 1) / 2
        current_density = current_edges / current_max if current_max > 0 else 0.0

        new_edges = sum(
            1 for u in s_list
            if (u, v) in edge_set or (v, u) in edge_set
        )
        new_size = s_size + 1
        new_max = new_size * (new_size - 1) / 2
        new_density = (current_edges + new_edges) / new_max if new_max > 0 else 0.0

        return new_density - current_density

    @property
    def name(self) -> str:
        return "ClassicalDense"


class ClassicalDegreeScorer(Scorer):
    """简单的度数评分（理论 §3.1 基线方法）。

    Score(v) = deg(v)，完全不考虑当前部分解 S。

    用于对照实验中的纯度数贪心基线。
    """

    def __init__(self):
        self._degrees: dict[int, float] = {}

    def score(self, v: int, S: set[int], instance) -> float:
        if not self._degrees:
            n = instance.num_nodes
            adj = instance.adjacency
            self._degrees = {i: float(adj[i].sum()) for i in range(n)}
        return self._degrees.get(v, 0.0)

    def score_all(self, candidates: set[int], S: set[int],
                  instance) -> dict[int, float]:
        if not self._degrees:
            n = instance.num_nodes
            adj = instance.adjacency
            self._degrees = {i: float(adj[i].sum()) for i in range(n)}
        return {v: self._degrees.get(v, 0.0) for v in candidates}

    @property
    def name(self) -> str:
        return "ClassicalDegree"


# ============================================================
# 量子概率评分 (占位)
# ============================================================

class QuantumScorer(Scorer):
    """纯量子概率评分（理论 §8.2）—— 当前为占位实现。

    Q(v,S) = P_v(t) = |⟨v|e^{-iHt}|ψ₀⟩|²

    占位实现返回候选集合上的均匀随机评分，保证框架可运行。
    后续 CTQW 模块完成后替换为真实计算。
    """

    def __init__(self, t: float = 1.0, lam: float = 0.0,
                 init_method: str = "max_degree", seed: int = 0):
        self.t = t
        self.lam = lam
        self.init_method = init_method
        self._rng = np.random.RandomState(seed)

    def score_all(self, candidates: set[int], S: set[int],
                  instance) -> dict[int, float]:
        # 占位：在候选节点上生成均匀随机评分
        # TODO: 替换为真实的 CTQW 概率计算
        n_candidates = len(candidates)
        if n_candidates == 0:
            return {}
        raw = self._rng.rand(n_candidates)
        raw_sum = raw.sum()
        if raw_sum > 0:
            raw = raw / raw_sum
        return dict(zip(candidates, raw))

    def score(self, v: int, S: set[int], instance) -> float:
        # 占位：返回随机值
        return float(self._rng.rand())

    @property
    def name(self) -> str:
        return "Quantum"


# ============================================================
# 混合评分 (组合器)
# ============================================================

class HybridScorer(Scorer):
    """混合评分函数（理论 §8.3）。

    Score(v,S) = α · Q_norm(v,S) + (1-α) · R_norm(v,S)

    其中归一化采用 min-max 归一化，仅在当前候选集合 C(S) 上计算：
        x_norm = (x - x_min) / (x_max - x_min + ε)

    设计要点：
      - 接受任意两个 Scorer 对象（不限定为量子+经典）
      - 归一化保证不同尺度的评分可公平混合
      - alpha=0 退化为纯 scorer_b，alpha=1 退化为纯 scorer_a
    """

    def __init__(self, scorer_a: Scorer, scorer_b: Scorer,
                 alpha: float = 0.5, eps: float = 1e-10):
        """
        参数:
            scorer_a: 第一个评分器（通常为量子评分）。
            scorer_b: 第二个评分器（通常为经典评分）。
            alpha: scorer_a 的权重，α ∈ [0, 1]。
            eps: 防止归一化时分母为零的小常数。
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha 必须在 [0, 1] 内，当前为 {alpha}")
        self.scorer_a = scorer_a
        self.scorer_b = scorer_b
        self.alpha = alpha
        self.eps = eps

    def score_all(self, candidates: set[int], S: set[int],
                  instance) -> dict[int, float]:
        if not candidates:
            return {}

        scores_a = self.scorer_a.score_all(candidates, S, instance)
        scores_b = self.scorer_b.score_all(candidates, S, instance)

        a_norm = _minmax_normalize(scores_a, self.eps)
        b_norm = _minmax_normalize(scores_b, self.eps)

        return {
            v: self.alpha * a_norm.get(v, 0.0)
            + (1 - self.alpha) * b_norm.get(v, 0.0)
            for v in candidates
        }

    def score(self, v: int, S: set[int], instance) -> float:
        a = self.scorer_a.score(v, S, instance)
        b = self.scorer_b.score(v, S, instance)
        return self.alpha * a + (1 - self.alpha) * b

    @property
    def name(self) -> str:
        return f"Hybrid(α={self.alpha},{self.scorer_a.name}+{self.scorer_b.name})"


def _minmax_normalize(scores: dict[int, float],
                       eps: float = 1e-10) -> dict[int, float]:
    """Min-max 归一化（理论 §8.3）。

    仅在候选集合上计算 min/max，避免非候选节点的极端值影响。
    """
    if not scores:
        return {}
    values = list(scores.values())
    v_min = min(values)
    v_max = max(values)
    denom = v_max - v_min + eps
    if denom == 0:
        return {k: 0.5 for k in scores}
    return {k: (v - v_min) / denom for k, v in scores.items()}
