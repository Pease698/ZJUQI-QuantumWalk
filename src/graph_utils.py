"""图加载与邻接矩阵构建。

从统一 JSON 格式加载测试实例，构建 NumPy 邻接矩阵和 NetworkX 图对象。
"""

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse


@dataclass
class GraphInstance:
    """从 JSON 加载后的内存数据结构。

    包含图结构信息、任务类型和 ground truth（若存在）。
    """

    sample_id: str
    num_nodes: int
    num_edges: int
    edges: list[tuple[int, int]]
    task_type: str                     # "maximum_clique" | "densest_subgraph"
    is_artificial: bool
    answer_nodes: list[int]
    answer_edges: list[tuple[int, int]]
    parameters: dict                   # 生成参数

    # 惰性构建的缓存
    _adjacency: np.ndarray | None = None
    _adjacency_sparse: sparse.csr_matrix | None = None
    _nx_graph: nx.Graph | None = None

    @property
    def adjacency(self) -> np.ndarray:
        """邻接矩阵 A，形状 (n, n)，dtype=float64。"""
        if self._adjacency is None:
            self._adjacency = build_adjacency(self.num_nodes, self.edges)
        return self._adjacency

    @property
    def adjacency_sparse(self) -> sparse.csr_matrix:
        """稀疏 CSR 邻接矩阵，供大图 CTQW 近似方法使用。"""
        if self._adjacency_sparse is None:
            self._adjacency_sparse = build_sparse_adjacency(
                self.num_nodes, self.edges)
        return self._adjacency_sparse

    @property
    def nx_graph(self) -> nx.Graph:
        """NetworkX 图对象，用于布局计算和经典算法。"""
        if self._nx_graph is None:
            self._nx_graph = nx.Graph()
            self._nx_graph.add_nodes_from(range(self.num_nodes))
            self._nx_graph.add_edges_from(self.edges)
        return self._nx_graph

    @property
    def answer_set(self) -> set[int]:
        """答案节点集合。"""
        return set(self.answer_nodes)

    @property
    def edge_set(self) -> set[tuple[int, int]]:
        """边集合（无序），用于 O(1) 邻接查询。"""
        return {
            (u, v) if u < v else (v, u)
            for u, v in self.edges
        }


def load_instance(filepath: str | Path) -> GraphInstance:
    """从统一 JSON 格式加载一个测试实例。

    参数:
        filepath: JSON 文件路径。

    返回:
        GraphInstance 对象。
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    edges = [tuple(e) for e in data["edges"]]
    answer_edges = [tuple(e) for e in data.get("answer_edges", [])]

    return GraphInstance(
        sample_id=data["sample_id"],
        num_nodes=data["num_nodes"],
        num_edges=data["num_edges"],
        edges=edges,
        task_type=data["task_type"],
        is_artificial=data["is_artificial"],
        answer_nodes=data.get("answer_nodes", []),
        answer_edges=answer_edges,
        parameters=data.get("parameters", {}),
    )


def load_instances_from_dir(directory: str | Path) -> list[GraphInstance]:
    """加载目录中所有 JSON 实例。

    参数:
        directory: 包含 *.json 文件的目录路径。

    返回:
        GraphInstance 列表，按 sample_id 排序。
    """
    dir_path = Path(directory)
    instances = []
    for fpath in sorted(dir_path.glob("*.json")):
        instances.append(load_instance(fpath))
    return instances


def build_adjacency(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """从边列表构建邻接矩阵。

    参数:
        n: 节点总数。
        edges: 边列表，每条边为 (u, v)。

    返回:
        n×n 的 float64 对称矩阵。
    """
    A = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A


def build_sparse_adjacency(n: int, edges: list[tuple[int, int]]) -> sparse.csr_matrix:
    """从边列表构建 CSR 稀疏邻接矩阵。"""
    if not edges:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    rows = []
    cols = []
    for u, v in edges:
        rows.extend([u, v])
        cols.extend([v, u])
    data = np.ones(len(rows), dtype=np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
