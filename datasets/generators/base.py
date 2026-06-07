"""公共工具函数：JSON 保存、样本 ID 生成、答案验证。"""

import json
import os


def format_prob(p: float) -> str:
    """将概率值格式化为文件名安全的数字字符串。

    示例: 0.1 -> '01', 0.05 -> '005', 1.0 -> '1'
    不包含前缀字母（p/r），前缀由调用方添加。
    """
    if p == int(p):
        return f"{int(p):d}"
    s = f"{p:.2f}".rstrip("0").replace(".", "")
    return s


def generate_sample_id(task_type: str, n: int, p: float, k: int,
                       rho: float | None, index: int) -> str:
    """生成统一的样本 ID。

    最大团示例: mc_n100_p02_k10_000
    密集子图示例: ds_n50_p01_k8_r06_005

    参数:
        task_type: "maximum_clique" 或 "densest_subgraph"
        n: 节点总数
        p: 背景边概率
        k: 答案子图大小
        rho: 答案边密度（最大团为 None）
        index: 实例序号（0 起始）
    """
    prefix = "mc" if task_type == "maximum_clique" else "ds"
    p_str = "p" + format_prob(p)
    if rho is not None:
        r_str = "r" + format_prob(rho)
        return f"{prefix}_n{n}_{p_str}_k{k}_{r_str}_{index:03d}"
    return f"{prefix}_n{n}_{p_str}_k{k}_{index:03d}"


def save_json(data: dict, filepath: str) -> None:
    """将数据字典保存为格式统一的 JSON 文件。

    自动创建父目录，使用 UTF-8 编码，缩进 2 空格。
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def verify_clique(edges_set: set, nodes: list[int]) -> bool:
    """验证给定节点集合是否构成团（任意两点之间均存在边）。"""
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            edge = (u, v) if u < v else (v, u)
            if edge not in edges_set:
                return False
    return True


def verify_density(edges_set: set, nodes: list[int]) -> float:
    """计算给定节点集合内部的边密度。

    返回: 实际边数 / 最大可能边数，范围 [0, 1]。
    """
    k = len(nodes)
    if k <= 1:
        return 1.0
    max_edges = k * (k - 1) / 2
    actual = 0
    for i in range(k):
        for j in range(i + 1, k):
            u, v = nodes[i], nodes[j]
            edge = (u, v) if u < v else (v, u)
            if edge in edges_set:
                actual += 1
    return actual / max_edges
