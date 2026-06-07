"""初始态构造（理论 §6）。

构造量子游走的初始态 |ψ₀⟩，支持三种初始化方式：
  - uniform:     均匀初态，从所有节点等权出发
  - max_degree:  最高度节点初始化（适合团和密集子图任务）
  - random:      随机单节点初始化

当 S 非空时，使用基于种子集合的均匀初态：
  |ψ₀⟩ = (1/√|S|) · Σ_{u∈S} |u⟩
"""

import numpy as np


def build_initial_state(n: int, S: set[int], adjacency: np.ndarray,
                        method: str = "max_degree") -> np.ndarray:
    """构造归一化的初始态 |ψ₀⟩。

    参数:
        n: 节点总数。
        S: 当前已选节点集合（可能为空）。
        adjacency: n×n 邻接矩阵，用于计算度数。
        method: 初始化方式。
            "uniform"     — 均匀分布
            "max_degree"  — 选度数最高节点
            "random"      — 随机选一个节点（固定 seed=0 以保证可复现）

    返回:
        形状为 (n,) 的复数数组，满足 Σ|ψ_i|² = 1。
    """
    psi = np.zeros(n, dtype=np.complex128)

    if S:
        # 基于种子集合的均匀初态（理论 §6 主方案）
        amp = 1.0 / np.sqrt(len(S))
        for u in S:
            psi[u] = amp
        return psi

    # S 为空时的初始化
    if method == "uniform":
        amp = 1.0 / np.sqrt(n)
        psi.fill(amp)

    elif method == "max_degree":
        degrees = adjacency.sum(axis=1)
        best = int(np.argmax(degrees))
        psi[best] = 1.0

    elif method == "random":
        rng = np.random.RandomState(0)
        chosen = rng.randint(0, n)
        psi[chosen] = 1.0

    else:
        raise ValueError(f"不支持的 init_method: {method}")

    return psi
