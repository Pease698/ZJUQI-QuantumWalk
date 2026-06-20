"""生成 degree-misleading 最大团测试实例。

这些实例用于补充普通 planted clique 数据：部分非答案节点具有很高
degree，但不能加入答案团，以检验 CTQW 全局信号是否优于简单度数线索。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generators.planted_clique import generate_degree_misleading_clique_batch


DATA_DIR = os.path.join(
    os.path.dirname(__file__), "data", "artificial", "maximum_clique")

DEFAULT_CONFIGS = [
    (100, 0.05, 10, 5),
    (100, 0.10, 10, 5),
    (200, 0.05, 16, 5),
    (200, 0.10, 16, 5),
    (300, 0.05, 20, 5),
    (300, 0.10, 20, 5),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 degree-misleading planted clique 数据。")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划，不生成文件")
    parser.add_argument("--instances", type=int, default=None,
                        help="覆盖每组实例数")
    parser.add_argument("--decoy-prob", type=float, default=0.75,
                        help="诱饵节点连接非禁用边的概率")
    args = parser.parse_args()

    print("degree-misleading 最大团数据生成计划")
    for idx, (n, p, k, ninst_default) in enumerate(DEFAULT_CONFIGS):
        ninst = args.instances or ninst_default
        dirname = f"mc_mislead_n{n}_p{str(p).replace('.', '')}_k{k}"
        out = os.path.join(DATA_DIR, dirname)
        print(f"  {dirname}: n={n}, p={p}, k={k}, instances={ninst}")
        if args.dry_run:
            continue
        generate_degree_misleading_clique_batch(
            n=n,
            p=p,
            k=k,
            num_instances=ninst,
            output_dir=out,
            start_seed=90000 + idx * 100,
            decoy_degree_prob=args.decoy_prob,
        )


if __name__ == "__main__":
    main()
