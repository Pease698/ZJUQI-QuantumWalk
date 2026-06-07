"""DIMACS 数据集转换器 —— 将 .mtx 格式转为项目统一 JSON 格式。

DIMACS 数据集同时用于最大团和最密子图评测（植入团即为密度最高的子图），
默认同时生成两份 JSON，仅 task_type 和 sample_id 前缀不同。

用法:
    python -m converters.convert_dimacs                          # 同时输出 MC + DS
    python -m converters.convert_dimacs --task-type maximum_clique  # 仅输出 MC
    python -m converters.convert_dimacs --task-type densest_subgraph  # 仅输出 DS
    python -m converters.convert_dimacs --dry-run                # 预览
"""

import argparse
import os
import sys

# 确保可以从 datasets 目录导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.base import save_json

# ============================================================
# 已下载数据集的目前最佳团大小（来源: DIMACS 基准测试）
# 键名为 .mtx 文件所在目录名
# ============================================================
BEST_KNOWN_CLIQUE_SIZES: dict[str, int] = {
    "brock400-4": 22,
    "brock800-4": 19,
    "C250-9": 40,
    "C500-9": 49,
    "C1000-9": 53,
    "C2000-9": 56,
    "gen200-p0-9-44": 36,
    "gen400-p0-9-55": 45,
    "p-hat300-3": 33,
    "p-hat700-2": 41,
}

# task_type → (sample_id 前缀, 默认输出子目录)
_TASK_META = {
    "maximum_clique": ("ext_mc", "maximum_clique"),
    "densest_subgraph": ("ext_ds", "densest_subgraph"),
}


def parse_mtx(filepath: str) -> tuple[list[list[int]], int, int]:
    """解析 Matrix Market (.mtx) 格式的图文件。

    参数:
        filepath: .mtx 文件路径。

    返回:
        (edges, num_nodes, num_entries_from_header)
        edges 中每条边为 [u, v] 且 u < v（0 起始）。
    """
    edges = []
    num_nodes = 0
    num_entries = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue

            parts = line.split()
            # 尺寸行: rows cols num_entries
            if num_nodes == 0 and len(parts) == 3:
                num_nodes = int(parts[0])
                num_entries = int(parts[2])
                continue

            # 数据行: col row（1 起始，下三角存储）
            if len(parts) == 2:
                a = int(parts[0]) - 1
                b = int(parts[1]) - 1
                u, v = (a, b) if a < b else (b, a)
                edges.append([u, v])

    return edges, num_nodes, num_entries


def build_json(dataset_name: str, task_type: str, edges: list[list[int]],
               num_nodes: int) -> dict:
    """根据解析后的图数据构造统一 JSON 字典。

    参数:
        dataset_name: DIMACS 数据集名称（如 "brock400-4"）。
        task_type: "maximum_clique" 或 "densest_subgraph"。
        edges: 0 起始的边列表。
        num_nodes: 节点总数。

    返回:
        统一 JSON 格式的字典。
    """
    prefix, _ = _TASK_META[task_type]
    sample_id = f"{prefix}_{dataset_name}"
    best_known = BEST_KNOWN_CLIQUE_SIZES.get(dataset_name)

    return {
        "sample_id": sample_id,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "edges": edges,
        "task_type": task_type,
        "is_artificial": False,
        "parameters": {
            "source": "DIMACS",
            "dataset": dataset_name,
            "best_known_clique_size": best_known,
        },
        "answer_nodes": [],
        "answer_edges": [],
    }


def convert_all(raw_dir: str, external_dir: str,
                task_types: list[str], dry_run: bool = False) -> int:
    """扫描原始目录，解析 .mtx 文件，按指定的任务类型输出 JSON。

    每个 .mtx 文件仅解析一次；当 task_types 包含多个值时，
    对同一份解析结果生成多份 JSON（分别对应不同 task_type）。

    返回:
        生成的 JSON 文件总数。
    """
    if not os.path.isdir(raw_dir):
        print(f"错误: 原始数据目录不存在: {raw_dir}")
        return 0

    total = 0
    entries = sorted(os.listdir(raw_dir))

    for entry in entries:
        subdir = os.path.join(raw_dir, entry)
        if not os.path.isdir(subdir):
            continue

        mtx_files = [f for f in os.listdir(subdir) if f.endswith(".mtx")]
        if not mtx_files:
            print(f"  跳过 {entry}: 未找到 .mtx 文件")
            continue

        mtx_path = os.path.join(subdir, mtx_files[0])
        dataset_name = entry

        if dry_run:
            bk = BEST_KNOWN_CLIQUE_SIZES.get(dataset_name, "?")
            types_str = "+".join(task_types)
            print(f"  {entry}: {mtx_files[0]}  best_known={bk}  -> {types_str}")
            total += len(task_types)
            continue

        # 解析 .mtx（每个数据集只解析一次）
        edges, num_nodes, _ = parse_mtx(mtx_path)
        if num_nodes == 0:
            print(f"  跳过 {entry}: 解析失败")
            continue

        best_known = BEST_KNOWN_CLIQUE_SIZES.get(dataset_name)
        note = f"best_known={best_known}" if best_known else "no best_known"

        for task_type in task_types:
            prefix, subdir_name = _TASK_META[task_type]
            output_dir = os.path.join(external_dir, subdir_name)
            os.makedirs(output_dir, exist_ok=True)

            data = build_json(dataset_name, task_type, edges, num_nodes)
            output_path = os.path.join(output_dir, f"{data['sample_id']}.json")
            save_json(data, output_path)
            total += 1
            print(f"  {dataset_name}: n={num_nodes}, |E|={len(edges)} "
                  f"({note}) -> {subdir_name}/{data['sample_id']}.json")

    return total


def main():
    parser = argparse.ArgumentParser(
        description="将 DIMACS .mtx 格式数据转为项目统一 JSON 格式。")
    parser.add_argument(
        "--input", default=None,
        help="原始数据目录路径（默认: data/external/maximum_clique_raw）")
    parser.add_argument(
        "--output-base", default=None,
        help="external 数据目录路径（默认: data/external，输出将写入其子目录）")
    parser.add_argument(
        "--task-type", default="both",
        choices=["maximum_clique", "densest_subgraph", "both"],
        help="输出哪些任务类型。both 表示同时输出 MC 和 DS 两份 JSON（默认: both）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅扫描文件，不实际转换。")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    external_dir = args.output_base or os.path.join(base_dir, "data", "external")
    raw_dir = args.input or os.path.join(external_dir, "maximum_clique_raw")

    if args.task_type == "both":
        task_types = ["maximum_clique", "densest_subgraph"]
    else:
        task_types = [args.task_type]

    print(f"原始目录:    {raw_dir}")
    print(f"输出基目录:  {external_dir}")
    print(f"输出任务:    {' + '.join(task_types)}")
    print()

    count = convert_all(raw_dir, external_dir, task_types, dry_run=args.dry_run)
    if args.dry_run:
        print(f"\n(dry-run，将生成 {count} 个文件)")
    else:
        print(f"\n转换完成，共生成 {count} 个文件。")


if __name__ == "__main__":
    main()
