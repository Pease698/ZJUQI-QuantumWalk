"""候选集合构造（理论 §7）。

候选集合是量子概率分布到最终节点选择的桥梁：
先由约束筛选出合法候选，再由评分函数在候选集合上排序。

设计原理见理论文档 §7：
  - 最大团任务：候选节点必须与 S 中所有节点相连
  - 密集子图任务：候选节点加入后密度不低于阈值
"""

from abc import ABC, abstractmethod


class CandidateSetBuilder(ABC):
    """候选集合构造器抽象基类。"""

    @abstractmethod
    def build(self, adjacency, edge_set: set, S: set[int],
              all_nodes: set[int]) -> set[int]:
        """根据当前部分解 S 构造候选节点集合 C(S)。

        参数:
            adjacency: n×n 邻接矩阵（np.ndarray）。
            edge_set: 无序边集合，用于 O(1) 邻接查询。
            S: 已选节点集合。
            all_nodes: 图中所有节点的集合 {0, 1, ..., n-1}。

        返回:
            候选节点索引集合。
        """
        ...


class CliqueCandidateSet(CandidateSetBuilder):
    """最大团任务的候选集合（理论 §7.1）。

    候选节点必须与 S 中所有节点都相连：
        C(S) = {v ∈ V-S | ∀u ∈ S, (u,v) ∈ E}

    正确性由归纳法保证：若 S 是团，且 v 与 S 中所有节点相连，
    则 S∪{v} 仍是团。
    """

    def build(self, adjacency, edge_set: set, S: set[int],
              all_nodes: set[int]) -> set[int]:
        if not S:
            # S 为空时所有节点均为候选
            return all_nodes.copy()

        candidates = set()
        remaining = all_nodes - S
        for v in remaining:
            if all((u, v) in edge_set or (v, u) in edge_set for u in S):
                candidates.add(v)
        return candidates


class DenseCandidateSet(CandidateSetBuilder):
    """密集子图任务的候选集合（理论 §7.2）。

    候选节点加入后子图密度不低于阈值：
        C(S) = {v ∈ V-S | ρ(S∪{v}) ≥ θ}

    若未设置阈值 θ，则所有非 S 节点均为候选（由算法控制停止条件）。
    """

    def __init__(self, theta: float = 0.0):
        """
        参数:
            theta: 密度阈值，范围 [0, 1]。默认 0 表示不设硬阈值。
        """
        self.theta = theta

    def build(self, adjacency, edge_set: set, S: set[int],
              all_nodes: set[int]) -> set[int]:
        if not S:
            return all_nodes.copy()

        if self.theta <= 0.0:
            return all_nodes - S

        candidates = set()
        remaining = all_nodes - S
        s_list = list(S)
        s_size = len(s_list)
        current_edges = _count_internal_edges(edge_set, s_list)

        for v in remaining:
            new_edges = sum(
                1 for u in s_list
                if (u, v) in edge_set or (v, u) in edge_set
            )
            new_size = s_size + 1
            max_edges = new_size * (new_size - 1) / 2
            density = (current_edges + new_edges) / max_edges
            if density >= self.theta:
                candidates.add(v)

        return candidates


def _count_internal_edges(edge_set: set, nodes: list[int]) -> int:
    """计算节点列表内部边数。"""
    count = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if (u, v) in edge_set or (v, u) in edge_set:
                count += 1
    return count
