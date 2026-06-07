"""哈密顿量构造（理论 §5）。

H = A + λ · Σ_{u∈S} |u⟩⟨u|

其中：
  A       — 图邻接矩阵
  S       — 当前种子集合
  |u⟩⟨u|  — 节点 u 的投影算子（对角元为 1 的矩阵）
  λ       — 扰动强度

后续 CTQW 模块将从这里获取哈密顿量矩阵。
"""

import numpy as np


def construct_hamiltonian(adjacency: np.ndarray, S: set[int],
                          lam: float = 0.0) -> np.ndarray:
    """构造哈密顿量 H = A + λ · Σ|u⟩⟨u|。

    参数:
        adjacency: n×n 邻接矩阵。
        S: 当前已选节点（种子）集合。
        lam: 扰动强度 λ。λ=0 时退化为 H=A。

    返回:
        n×n Hermitian 矩阵（当前为实对称）。
    """
    H = adjacency.copy()
    if lam != 0.0 and S:
        for u in S:
            H[u, u] += lam
    return H
