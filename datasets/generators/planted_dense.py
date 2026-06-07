"""植入密集子图生成器 —— 用于密集子图问题的测试数据生成。

算法: 在 Erdos-Renyi 背景图 G(n, p) 中随机植入一个大小为 k、边密度约为 ρ 的密集子图。
"""

import random

from .base import generate_sample_id, save_json, verify_density


def generate_planted_dense(n: int, p: float, k: int, rho: float,
                           seed: int, index: int = 0) -> dict:
    """生成一个包含植入密集子图的无向图。

    参数:
        n: 图中节点总数。
        p: 非答案节点对之间的背景边概率。
        k: 植入密集子图的大小（节点数）。
        rho: 植入子图的目标边密度，0 < rho <= 1。实际密度会因取整而略有偏差。
        seed: 随机种子，用于保证可复现性。
        index: 实例序号（0 起始），用于生成样本 ID。

    返回:
        统一 JSON 格式的字典。
    """
    rng = random.Random(seed)

    # 随机选取 k 个节点作为植入密集子图
    all_nodes = list(range(n))
    answer_nodes = sorted(rng.sample(all_nodes, k))
    answer_set = set(answer_nodes)

    # 构造答案子图内部所有可能的边
    possible_internal = []
    for i in range(k):
        for j in range(i + 1, k):
            possible_internal.append((answer_nodes[i], answer_nodes[j]))

    # 从所有可能的内部边中随机选取目标数量条边
    target_internal = round(rho * len(possible_internal))
    selected_internal = set(rng.sample(possible_internal, target_internal))

    edges = []
    answer_edges = []

    for i in range(n):
        for j in range(i + 1, n):
            if i in answer_set and j in answer_set:
                edge_key = (i, j) if i < j else (j, i)
                if edge_key in selected_internal:
                    edges.append([i, j])
                    answer_edges.append([i, j])
            else:
                # 背景边：以概率 p 存在
                if rng.random() < p:
                    edges.append([i, j])

    # 计算并记录实际密度（因四舍五入可能偏离目标值）
    edges_set = set(tuple(sorted(e)) for e in edges)
    actual_rho = verify_density(edges_set, answer_nodes)

    sample_id = generate_sample_id("densest_subgraph", n, p, k, rho, index)

    return {
        "sample_id": sample_id,
        "num_nodes": n,
        "num_edges": len(edges),
        "edges": edges,
        "task_type": "densest_subgraph",
        "is_artificial": True,
        "parameters": {
            "num_nodes": n,
            "bg_edge_prob": p,
            "answer_size": k,
            "answer_edge_density": round(actual_rho, 4),
        },
        "answer_nodes": answer_nodes,
        "answer_edges": answer_edges,
    }


def generate_dense_batch(n: int, p: float, k: int, rho: float,
                         num_instances: int,
                         output_dir: str,
                         start_seed: int = 0) -> list[str]:
    """批量生成植入密集子图测试实例。

    实例 i 的随机种子为 start_seed + i，样本序号为 i。

    返回:
        已生成文件的路径列表。
    """
    paths = []
    for i in range(num_instances):
        seed = start_seed + i
        data = generate_planted_dense(n, p, k, rho, seed, index=i)
        sample_id = data["sample_id"]
        filepath = f"{output_dir}/{sample_id}.json"
        save_json(data, filepath)
        paths.append(filepath)
    return paths
