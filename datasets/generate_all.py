"""一键生成所有人工测试数据的入口脚本。

用法:
    python generate_all.py              # 生成所有测试实例
    python generate_all.py --dry-run    # 仅打印生成计划，不实际生成
"""

import argparse
import os
import sys
import time

# 确保可以从 datasets 目录导入生成器模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generators.planted_clique import generate_clique_batch
from generators.planted_dense import generate_dense_batch

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ============================================================
# 参数矩阵定义
# ============================================================

# 最大团配置: (节点数, 背景概率列表, 植入团大小列表, 每组实例数)
# 实验时每个实例重复运行 4 次，达到每组 5×4=20 个数据点的统计要求
CLIQUE_CONFIGS = [
    # --- 验证层 ---
    (30,   [0.1, 0.2],       [5, 8],       5),
    (50,   [0.1, 0.2, 0.3],  [5, 8, 10],   5),
    # --- 过渡层 ---
    (100,  [0.1, 0.2, 0.3],  [8, 10, 12],  5),
    # --- 主实验层 ---
    (150,  [0.1, 0.2, 0.3],  [10, 15],     5),
    (200,  [0.1, 0.2],       [12, 16, 20],  5),
    # --- 压力测试层 ---
    (300,  [0.05, 0.1],      [15, 20, 25],  5),
    (500,  [0.05, 0.1],      [20, 30],      5),
]

# 密集子图配置: (节点数, 背景概率列表, 答案大小列表, 目标密度列表, 每组实例数)
# 实验时每个实例重复运行 4 次，达到每组 5×4=20 个数据点的统计要求
DENSE_CONFIGS = [
    (30,   [0.1, 0.2],  [5, 8],     [0.6, 0.7, 0.8], 5),
    (50,   [0.1, 0.2],  [5, 8],     [0.6, 0.7, 0.8], 5),
    (100,  [0.1, 0.2],  [8, 10],    [0.6, 0.7, 0.8], 5),
    (150,  [0.1, 0.2],  [10, 15],   [0.6, 0.7, 0.8], 5),
    (200,  [0.1, 0.2],  [12, 16],   [0.6, 0.7, 0.8], 5),
]


def count_groups(configs, has_rho=False):
    """统计参数组合数和实例总数。"""
    total_groups = 0
    total_instances = 0
    for cfg in configs:
        if has_rho:
            n, p_list, k_list, rho_list, ninst = cfg
            groups = len(p_list) * len(k_list) * len(rho_list)
        else:
            n, p_list, k_list, ninst = cfg
            groups = len(p_list) * len(k_list)
        total_groups += groups
        total_instances += groups * ninst
    return total_groups, total_instances


def print_summary():
    """打印生成计划概要（不实际生成）。"""
    mc_groups, mc_instances = count_groups(CLIQUE_CONFIGS)
    ds_groups, ds_instances = count_groups(DENSE_CONFIGS, has_rho=True)

    print("=" * 60)
    print("测试数据生成计划")
    print("=" * 60)
    print()
    print("最大团问题 (MC):")
    print(f"  参数组合数: {mc_groups}")
    print(f"  实例总数:   {mc_instances}")
    print()
    for n, p_list, k_list, ninst in CLIQUE_CONFIGS:
        for p in p_list:
            for k in k_list:
                print(f"  n={n:3d}  p={p:.2f}  k={k:2d}  ×{ninst}")

    print()
    print("密集子图问题 (DS):")
    print(f"  参数组合数: {ds_groups}")
    print(f"  实例总数:   {ds_instances}")
    print()
    for n, p_list, k_list, rho_list, ninst in DENSE_CONFIGS:
        for p in p_list:
            for k in k_list:
                for rho in rho_list:
                    print(f"  n={n:3d}  p={p:.2f}  k={k:2d}  ρ={rho:.1f}  ×{ninst}")

    print()
    print(f"总计: {mc_groups + ds_groups} 组参数, "
          f"{mc_instances + ds_instances} 个实例")


def generate_all():
    """生成所有测试数据。"""
    mc_groups, mc_instances = count_groups(CLIQUE_CONFIGS)
    ds_groups, ds_instances = count_groups(DENSE_CONFIGS, has_rho=True)
    total = mc_instances + ds_instances
    generated = 0

    print(f"正在生成 {total} 个实例 "
          f"(最大团 {mc_instances} + 密集子图 {ds_instances})...")
    print()

    group_seed = 0
    t_start = time.perf_counter()

    # --- 最大团 ---
    for n, p_list, k_list, ninst in CLIQUE_CONFIGS:
        for p in p_list:
            for k in k_list:
                output_dir = os.path.join(DATA_DIR, "maximum_clique",
                                          f"mc_n{n}_p{str(p).replace('.', '')}_k{k}")
                start_seed = group_seed * 100
                generate_clique_batch(n, p, k, ninst, output_dir, start_seed)
                generated += ninst
                group_seed += 1
                print(f"  [{generated:4d}/{total}] {output_dir}  完成")

    # --- 密集子图 ---
    for n, p_list, k_list, rho_list, ninst in DENSE_CONFIGS:
        for p in p_list:
            for k in k_list:
                for rho in rho_list:
                    rho_str = str(rho).replace(".", "")
                    output_dir = os.path.join(
                        DATA_DIR, "densest_subgraph",
                        f"ds_n{n}_p{str(p).replace('.', '')}_k{k}_r{rho_str}")
                    start_seed = group_seed * 100
                    generate_dense_batch(n, p, k, rho, ninst, output_dir,
                                         start_seed)
                    generated += ninst
                    group_seed += 1
                    print(f"  [{generated:4d}/{total}] {output_dir}  完成")

    elapsed = time.perf_counter() - t_start
    print()
    print(f"全部完成。共生成 {generated} 个实例，耗时 {elapsed:.1f} 秒。")


def main():
    parser = argparse.ArgumentParser(
        description="一键生成最大团和密集子图问题的所有人工测试数据。")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印生成计划，不实际生成数据。")
    args = parser.parse_args()

    if args.dry_run:
        print_summary()
    else:
        print_summary()
        print()
        generate_all()


if __name__ == "__main__":
    main()
