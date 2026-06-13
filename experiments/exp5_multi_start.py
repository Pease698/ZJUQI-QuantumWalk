#!/usr/bin/env python3
"""实验五：Multi-Start + CTQW 种子选择（H4 验证）。

H4 假设：把 CTQW 用在贪心**外部**作为起点选择器（利用实验一已验证的全图
均匀叠加下的全局识别能力），能稳定超越纯 ClassicalClique 贪心。

对照（4 个，缺一不可）：
  1. ClassicalClique          — 实验二 §4.3 的强基线（团均值 7.20 / 11.70）
  2. MultiStartRandom(K)      — 随机起点；剥离"多起点本身就涨"
  3. MultiStartDegree(K)      — 度数 Top-K 起点；剥离"任何全局信号都行"
  4. MultiStartCTQW(K)        — 本方案

K ∈ {1, 3, 5, 10}，small 模式跑全扫描，medium 模式只跑 --K 指定的 K*。

诊断规则:
  CTQW > Random:  CTQW 起点有效
  CTQW ≈ Degree:  CTQW 没提供度数之外信息（H4 弱成立）
  CTQW > Degree:  CTQW 提供了度数之外的额外信息（H4 强成立）

用法:
  python experiments/exp5_multi_start.py --smoke            # 1 实例烟雾测试
  python experiments/exp5_multi_start.py --small            # n≤50 全跑 + K 扫描
  python experiments/exp5_multi_start.py --medium --K 5     # n=60~200 跑 K=5
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_utils import load_instance, load_instances_from_dir, GraphInstance
from src.candidate_set import CliqueCandidateSet
from src.scoring import ClassicalCliqueScorer
from src.algorithms.classical_greedy import ClassicalGreedy
from src.algorithms.multi_start_ctqw import (
    MultiStartCTQWGreedy, MultiStartRandomGreedy, MultiStartDegreeGreedy)
from src.metrics import mean_std
from src.config import get_data_dirs

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp5_multi_start"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPEAT_RUNS = 4
K_SWEEP = [1, 3, 5, 10]
CTQW_T = 1.0


# ============================================================
# 算法工厂
# ============================================================

def build_algorithms(K: int, seed: int = 0) -> dict:
    """构造一组 (4 算法) for given K and seed."""
    return {
        "ClassicalClique": ClassicalGreedy(
            CliqueCandidateSet(), ClassicalCliqueScorer(),
            name="ClassicalClique"),
        f"MultiStartRandom_K{K}": MultiStartRandomGreedy(K=K, seed=seed),
        f"MultiStartDegree_K{K}": MultiStartDegreeGreedy(K=K, seed=seed),
        f"MultiStartCTQW_K{K}": MultiStartCTQWGreedy(K=K, t=CTQW_T, seed=seed),
    }


# ============================================================
# 实例发现
# ============================================================

def _extract_n(dirname: str) -> int:
    import re
    m = re.search(r'_n(\d+)_', dirname)
    return int(m.group(1)) if m else 999


def discover_instances(scale: str) -> list[tuple[GraphInstance, str]]:
    """根据 scale ('small'|'medium') 返回 [(instance, label)] 列表。"""
    dirs = get_data_dirs("maximum_clique")
    if scale == "small":
        dirs = [d for d in dirs if _extract_n(d.name) <= 50]
    elif scale == "medium":
        dirs = [d for d in dirs if 50 < _extract_n(d.name) <= 200]
    elif scale == "smoke":
        dirs = [d for d in dirs if _extract_n(d.name) == 30][:1]
    else:
        raise ValueError(f"未知 scale: {scale}")

    instances = []
    for d in dirs:
        for inst in load_instances_from_dir(d):
            instances.append((inst, d.name))
    if scale == "smoke" and instances:
        instances = instances[:1]
    return instances


# ============================================================
# 单实例运行
# ============================================================

def run_one_instance(inst: GraphInstance, K: int,
                     repeat: int) -> list[dict]:
    """对一个实例跑 K 对应的 4 算法 × repeat 次。"""
    rows = []
    for run_id in range(repeat):
        algos = build_algorithms(K=K, seed=run_id)
        for algo_name, algo in algos.items():
            t0 = time.perf_counter()
            result = algo.solve(inst)
            elapsed = time.perf_counter() - t0
            row = result.to_dict()
            row["algorithm"] = algo_name  # 覆盖掉 inner 名字
            row["run_id"] = run_id
            row["K"] = K
            row["seed"] = run_id
            row["wall_time"] = elapsed
            rows.append(row)
    return rows


# ============================================================
# 批量实验
# ============================================================

def run_sweep(scale: str, Ks: list[int],
              repeat: int = REPEAT_RUNS) -> pd.DataFrame:
    """对每个 K 跑一遍全部实例。"""
    instances = discover_instances(scale)
    if not instances:
        print(f"警告: {scale} 模式找不到实例")
        return pd.DataFrame()

    total = len(instances)
    print(f"\n实验五：Multi-Start + CTQW 种子选择")
    print(f"  规模:     {scale}")
    print(f"  实例数:   {total}")
    print(f"  K 取值:   {Ks}")
    print(f"  每实例重复: {repeat}")
    print(f"  预期记录: {total * len(Ks) * 4 * repeat}")
    print("-" * 60)

    all_rows = []
    t_start = time.perf_counter()

    for K in Ks:
        print(f"\n  >> K = {K}")
        for idx, (inst, label) in enumerate(instances):
            rows = run_one_instance(inst, K=K, repeat=repeat)
            for r in rows:
                r["source_label"] = label
            all_rows.extend(rows)

            if (idx + 1) % 10 == 0 or idx == total - 1:
                elapsed = time.perf_counter() - t_start
                print(f"    [{idx + 1:4d}/{total}] K={K}, 总耗时 {elapsed:.1f}s, "
                      f"已收集 {len(all_rows)} 条")

    df = pd.DataFrame(all_rows)
    print(f"\n全部完成，共 {len(df)} 条记录，总耗时 {time.perf_counter() - t_start:.1f}s")

    csv_path = RESULTS_DIR / f"mc_{scale}_sweep.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"结果已保存至: {csv_path}")
    return df


# ============================================================
# 分析
# ============================================================

def _algo_family(algo_name: str) -> str:
    """从 algorithm 列提取算法家族（去掉 _K* 后缀）。"""
    if algo_name.startswith("MultiStartRandom"):
        return "MultiStartRandom"
    if algo_name.startswith("MultiStartDegree"):
        return "MultiStartDegree"
    if algo_name.startswith("MultiStartCTQW"):
        return "MultiStartCTQW"
    if algo_name.startswith("ClassicalClique"):
        return "ClassicalClique"
    return algo_name


def summarize(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """对每个 (family, K) 算 μ±σ，并对 ClassicalClique 做 Wilcoxon 配对检验。"""
    if df.empty:
        print("空 DataFrame，跳过分析")
        return pd.DataFrame()

    df = df.copy()
    df["family"] = df["algorithm"].apply(_algo_family)

    # ClassicalClique 不依赖 K，复制到每个 K 下用于配对
    cc_rows = df[df["family"] == "ClassicalClique"].copy()

    print(f"\n{'=' * 60}")
    print(f"实验五结果汇总 ({tag})")
    print(f"{'=' * 60}")

    Ks = sorted([int(k) for k in df["K"].unique() if not pd.isna(k)])

    # 找出 ClassicalClique 的均值（不依赖 K，所有 K 下 CC 跑的都是同一算法，
    # 但我们让它在每个 K 都跑一遍以便配对）
    cc_obj_mean, cc_obj_std = mean_std(cc_rows["objective"].tolist())
    print(f"\nClassicalClique (基线): {cc_obj_mean:.4f} ± {cc_obj_std:.4f}, "
          f"n={len(cc_rows)}")

    summary_rows = []
    summary_rows.append({
        "family": "ClassicalClique", "K": "-",
        "obj_mean": cc_obj_mean, "obj_std": cc_obj_std,
        "n": len(cc_rows), "delta_vs_CC": 0.0,
        "p_value": None, "significance": "baseline",
    })

    for fam in ["MultiStartRandom", "MultiStartDegree", "MultiStartCTQW"]:
        print(f"\n  {fam}:")
        for K in Ks:
            sub = df[(df["family"] == fam) & (df["K"] == K)]
            if sub.empty:
                continue
            obj_mean, obj_std = mean_std(sub["objective"].tolist())
            delta = obj_mean - cc_obj_mean

            # Wilcoxon 配对：(sample_id, run_id) 配对 K 下的算法 vs ClassicalClique
            cc_K = cc_rows[cc_rows["K"] == K]
            base_lookup = {
                (r["sample_id"], r["run_id"]): r["objective"]
                for _, r in cc_K.iterrows()
            }
            paired = []
            for _, r in sub.iterrows():
                key = (r["sample_id"], r["run_id"])
                if key in base_lookup:
                    paired.append((r["objective"], base_lookup[key]))
            if len(paired) >= 2:
                a_vals = np.array([p[0] for p in paired])
                b_vals = np.array([p[1] for p in paired])
                diffs = a_vals - b_vals
                if np.allclose(diffs, 0):
                    p_val = 1.0
                    sig = "≡ baseline"
                else:
                    try:
                        _, p_val = wilcoxon(a_vals, b_vals,
                                            zero_method="wilcox",
                                            alternative="two-sided")
                    except ValueError:
                        p_val = float("nan")
                    if pd.isna(p_val):
                        sig = "n/a"
                    elif p_val < 0.001:
                        sig = ("✓ p<0.001 显著优于" if delta > 0
                               else "✗ p<0.001 显著劣于") + " CC"
                    elif p_val < 0.01:
                        sig = ("✓ p<0.01 显著优于" if delta > 0
                               else "✗ p<0.01 显著劣于") + " CC"
                    elif p_val < 0.05:
                        sig = ("✓ p<0.05 显著优于" if delta > 0
                               else "✗ p<0.05 显著劣于") + " CC"
                    else:
                        sig = f"~ 无显著差异 (p={p_val:.3f})"
            else:
                p_val = float("nan")
                sig = "n/a"

            print(f"    K={K:>2d}: {obj_mean:.4f} ± {obj_std:.4f}  "
                  f"Δ={delta:+.3f}  {sig}")

            summary_rows.append({
                "family": fam, "K": K,
                "obj_mean": obj_mean, "obj_std": obj_std,
                "n": len(sub), "delta_vs_CC": delta,
                "p_value": p_val, "significance": sig,
            })

    summary_df = pd.DataFrame(summary_rows)
    out_path = RESULTS_DIR / f"mc_{tag}_summary.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n汇总已保存至: {out_path}")
    return summary_df


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="实验五：Multi-Start + CTQW 种子选择（H4 验证）")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true",
                      help="单实例烟雾测试")
    mode.add_argument("--small", action="store_true",
                      help="small (n≤50) 全跑 + K 扫描")
    mode.add_argument("--medium", action="store_true",
                      help="medium (n=60~200) 跑指定 K")
    parser.add_argument("--K", type=int, default=None,
                        help="medium 模式锁定 K（必填）；其他模式忽略")
    parser.add_argument("--csv", type=str, default=None,
                        help="直接分析已有 CSV，跳过运行")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        tag = Path(args.csv).stem
        summarize(df, tag)
        return

    if args.smoke:
        df = run_sweep("smoke", Ks=[5], repeat=2)
        summarize(df, "smoke")
        return

    if args.small:
        df = run_sweep("small", Ks=K_SWEEP, repeat=REPEAT_RUNS)
        summarize(df, "small")
        return

    if args.medium:
        if args.K is None:
            parser.error("--medium 必须配合 --K")
        df = run_sweep("medium", Ks=[args.K], repeat=REPEAT_RUNS)
        summarize(df, f"medium_K{args.K}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
