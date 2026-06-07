"""植入团生成器 —— 用于最大团问题的测试数据生成。

算法: 在 Erdos-Renyi 背景图 G(n, p) 中随机植入一个 k-团。
"""

import random

from .base import generate_sample_id, save_json, verify_clique


def generate_planted_clique(n: int, p: float, k: int, seed: int,
                            index: int = 0) -> dict:
    """生成一个包含植入 k-团的无向图。

    参数:
        n: 图中节点总数。
        p: 非团节点对之间的背景边概率。
        k: 植入团的大小（节点数）。
        seed: 随机种子，用于保证可复现性。
        index: 实例序号（0 起始），用于生成样本 ID。

    返回:
        包含 sample_id、num_nodes、num_edges、edges、task_type、
        is_artificial、parameters、answer_nodes、answer_edges 的字典。
    """
    rng = random.Random(seed)

    # 随机选取 k 个节点作为植入团
    all_nodes = list(range(n))
    answer_nodes = sorted(rng.sample(all_nodes, k))
    answer_set = set(answer_nodes)

    edges = []
    answer_edges = []

    for i in range(n):
        for j in range(i + 1, n):
            if i in answer_set and j in answer_set:
                # 团内边：必定存在
                edges.append([i, j])
                answer_edges.append([i, j])
            else:
                # 背景边：以概率 p 存在
                if rng.random() < p:
                    edges.append([i, j])

    # 验证植入团确为合法团（安全检查）
    edges_set = set(tuple(sorted(e)) for e in edges)
    assert verify_clique(edges_set, answer_nodes), \
        f"植入团验证失败，seed={seed}"

    sample_id = generate_sample_id("maximum_clique", n, p, k, None, index)

    return {
        "sample_id": sample_id,
        "num_nodes": n,
        "num_edges": len(edges),
        "edges": edges,
        "task_type": "maximum_clique",
        "is_artificial": True,
        "parameters": {
            "num_nodes": n,
            "bg_edge_prob": p,
            "answer_size": k,
            "answer_edge_density": 1.0,
        },
        "answer_nodes": answer_nodes,
        "answer_edges": answer_edges,
    }


def generate_clique_batch(n: int, p: float, k: int,
                          num_instances: int,
                          output_dir: str,
                          start_seed: int = 0) -> list[str]:
    """批量生成植入团测试实例。

    实例 i 的随机种子为 start_seed + i，样本序号为 i。

    参数:
        n: 节点总数
        p: 背景边概率
        k: 植入团大小
        num_instances: 需生成的实例数量
        output_dir: 输出目录路径
        start_seed: 起始随机种子

    返回:
        已生成文件的路径列表。
    """
    paths = []
    for i in range(num_instances):
        seed = start_seed + i
        data = generate_planted_clique(n, p, k, seed, index=i)
        sample_id = data["sample_id"]
        filepath = f"{output_dir}/{sample_id}.json"
        save_json(data, filepath)
        paths.append(filepath)
    return paths
