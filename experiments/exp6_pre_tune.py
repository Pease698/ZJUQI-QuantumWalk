#!/usr/bin/env python3
"""exp6 前置实验：Krylov m 与 Chebyshev d 参数调优。

目的：
  在正式大规模实验之前，确定 Krylov 子空间维数 m 和 Chebyshev 多项式阶数 d
  的合理取值。在 n=200（精确 expm 仍可行）上计算近似 vs 精确的概率误差，
  在 n=500 上测量运行时间 scaling。

实验设计：
  - 测试图：1 张 n=200 + 1 张 n=500（人工植入团图）
  - 两种初态场景：全图均匀叠加（外置方案）/ 种子集合均匀叠加（嵌入式方案）
  - 扫描参数：Krylov m ∈ {20,30,40,50,60,80,100,150}
              Chebyshev d ∈ {20,30,40,50,60,80,100,150}
  - 指标：L2 误差、L∞ 误差、Top-10 一致性、运行时间

用法:
  python3 experiments/exp6_pre_tune.py --smoke          # 烟雾测试：3个点+1次重复
  python3 experiments/exp6_pre_tune.py                  # 完整扫描（约5分钟）
  python3 experiments/exp6_pre_tune.py --timeout 120    # 自定义超时
"""

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_utils import load_instance, GraphInstance
from src.hamiltonian import construct_hamiltonian
from src.initial_state import build_initial_state
from src.ctqw_evolution import compute_ctqw_evolution
from src.timeout import run_with_timeout

# ============================================================
# 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "exp6_pre_tune"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

KRYLOV_DIMS = [20, 30, 40, 50, 60, 80, 100, 150]
CHEB_DEGREES = [20, 30, 40, 50, 60, 80, 100, 150]
N_REPEAT = 3
DEFAULT_TIMEOUT = 120

# 测试图：人工植入团数据中选 n=200 和 n=500 各一张
INSTANCE_N200_REL = "datasets/data/artificial/maximum_clique/mc_n200_p01_k12/mc_n200_p01_k12_000.json"
INSTANCE_N500_REL = "datasets/data/artificial/maximum_clique/mc_n500_p01_k20/mc_n500_p01_k20_000.json"


# ============================================================
# 辅助：创建模拟的已选集合 S（模拟嵌入式方案的中途状态）
# ============================================================

def _make_mid_s(instance: GraphInstance, n_half: int) -> set[int]:
    """从 answer_nodes 中取前 n_half 个节点作为模拟的 S。"""
    answer = instance.answer_nodes
    return set(answer[: min(n_half, len(answer))])


# ============================================================
# 核心计算
# ============================================================

def _compute_probabilities_from_parts(
    adjacency: np.ndarray,
    S: set[int],
    lam: float,
    t: float,
    init_method: str,
    evolution_method: str,
    **ev_kwargs,
) -> np.ndarray:
    """从邻接矩阵出发计算 CTQW 概率分布（不依赖 GraphInstance 的完整接口）。"""
    n = adjacency.shape[0]
    H = construct_hamiltonian(adjacency, S, lam)
    psi0 = build_initial_state(n, S, adjacency, init_method)
    psi_t = compute_ctqw_evolution(H, psi0, t, method=evolution_method, **ev_kwargs)
    return np.abs(psi_t) ** 2


def _compute_metrics(probs_approx: np.ndarray, probs_exact: np.ndarray) -> dict:
    """对比近似结果与精确结果的各项误差指标。"""
    diff = probs_approx - probs_exact
    eps2 = float(np.linalg.norm(diff))
    eps_inf = float(np.max(np.abs(diff)))

    # Top-10 一致性
    top10_exact = set(np.argsort(probs_exact)[-10:])
    top10_approx = set(np.argsort(probs_approx)[-10:])
    top10_match = len(top10_exact & top10_approx) / 10.0

    return {"eps2": eps2, "eps_inf": eps_inf, "top10_match": top10_match}


# ============================================================
# 主扫描
# ============================================================

def run_pre_experiment(smoke: bool = False, timeout: float = DEFAULT_TIMEOUT
                       ) -> pd.DataFrame:
    """运行完整的参数扫描。

    返回:
        DataFrame，每行一组参数组合的误差 + 时间数据。
    """
    # 加载测试图
    inst_200 = load_instance(str(PROJECT_ROOT / INSTANCE_N200_REL))
    inst_500 = load_instance(str(PROJECT_ROOT / INSTANCE_N500_REL))

    adj_200 = inst_200.adjacency
    adj_500 = inst_500.adjacency
    n200 = inst_200.num_nodes

    # 两种初态场景
    scenarios = [
        ("uniform_S0",   set(),              "uniform",     0.0),
        ("seeded_Smid",  _make_mid_s(inst_200, 6), "max_degree", 0.5),
    ]

    t_val = 1.0
    all_rows = []

    # ---- 先算精确基准（仅 n=200） ----
    print("计算精确基准 (n=200, scipy.linalg.expm)...")
    exact_cache: dict[str, np.ndarray] = {}
    for sc_name, S, init_m, lam_v in scenarios:
        probs = _compute_probabilities_from_parts(
            adj_200, S, lam_v, t_val, init_m, "exact")
        exact_cache[sc_name] = probs
        print(f"  {sc_name}: 完成")
    print()

    krylov_list = KRYLOV_DIMS[:3] if smoke else KRYLOV_DIMS
    cheb_list = CHEB_DEGREES[:3] if smoke else CHEB_DEGREES
    n_repeat = 1 if smoke else N_REPEAT

    # ---- 扫描 Krylov ----
    print(f"扫描 Krylov: m ∈ {krylov_list}")
    for m_val in krylov_list:
        for sc_name, S, init_m, lam_v in scenarios:
            # n=200：精度对比
            for rep in range(n_repeat):
                t0 = time.perf_counter()
                probs_200 = _compute_probabilities_from_parts(
                    adj_200, S, lam_v, t_val, init_m,
                    "krylov", krylov_dim=m_val)
                rt_200 = time.perf_counter() - t0
                metrics = _compute_metrics(probs_200, exact_cache[sc_name])

                # n=500：仅测时间（不对比精确）
                t0 = time.perf_counter()
                try:
                    _compute_probabilities_from_parts(
                        adj_500, S, lam_v, t_val, init_m,
                        "krylov", krylov_dim=m_val)
                    rt_500 = time.perf_counter() - t0
                except Exception:
                    rt_500 = float("nan")

                all_rows.append({
                    "method": "krylov", "param": m_val, "param_name": "m",
                    "scenario": sc_name, "rep": rep,
                    "n200_rt": rt_200, "n500_rt": rt_500,
                    **metrics,
                })

    # ---- 扫描 Chebyshev ----
    print(f"扫描 Chebyshev: d ∈ {cheb_list}")
    for d_val in cheb_list:
        for sc_name, S, init_m, lam_v in scenarios:
            for rep in range(n_repeat):
                t0 = time.perf_counter()
                probs_200 = _compute_probabilities_from_parts(
                    adj_200, S, lam_v, t_val, init_m,
                    "chebyshev", cheb_degree=d_val)
                rt_200 = time.perf_counter() - t0
                metrics = _compute_metrics(probs_200, exact_cache[sc_name])

                t0 = time.perf_counter()
                try:
                    _compute_probabilities_from_parts(
                        adj_500, S, lam_v, t_val, init_m,
                        "chebyshev", cheb_degree=d_val)
                    rt_500 = time.perf_counter() - t0
                except Exception:
                    rt_500 = float("nan")

                all_rows.append({
                    "method": "chebyshev", "param": d_val, "param_name": "d",
                    "scenario": sc_name, "rep": rep,
                    "n200_rt": rt_200, "n500_rt": rt_500,
                    **metrics,
                })

    return pd.DataFrame(all_rows)


# ============================================================
# 分析与推荐
# ============================================================

def _agg(df: pd.DataFrame, method: str):
    """对指定 method 按 param 聚合重复运行，取各指标均值。"""
    sub = df[df["method"] == method].copy()
    return sub.groupby("param").agg(
        eps2_mean=("eps2", "mean"),
        eps2_max=("eps2", "max"),
        eps_inf_max=("eps_inf", "max"),
        top10_min=("top10_match", "min"),
        rt200_mean=("n200_rt", "mean"),
        rt500_mean=("n500_rt", "mean"),
    ).reset_index()


def analyze_and_recommend(df: pd.DataFrame):
    """分析结果，生成收敛曲线图并输出推荐参数。"""
    print(f"\n{'=' * 65}")
    print("前置实验结果：推荐参数")
    print(f"{'=' * 65}")

    # 收敛阈值
    THRESHOLD = 1e-11

    recommendations = {}

    for method, param_label in [("krylov", "m"), ("chebyshev", "d")]:
        agg = _agg(df, method)
        if agg.empty:
            continue

        print(f"\n{'─' * 65}")
        print(f"  {method} ({param_label})")
        print(f"  {'param':>6} {'eps2_mean':>12} {'eps2_max':>12} "
              f"{'eps_inf':>12} {'top10':>8} {'rt_200':>10} {'rt_500':>10}")
        print(f"  {'─' * 62}")

        for _, row in agg.iterrows():
            p = int(row["param"])
            print(f"  {p:>6} {row['eps2_mean']:>12.2e} {row['eps2_max']:>12.2e} "
                  f"{row['eps_inf_max']:>12.2e} {row['top10_min']:>7.1%} "
                  f"{row['rt200_mean']:>9.4f}s {row['rt500_mean']:>9.4f}s")

        # 推荐：eps2_max < THRESHOLD 的最小参数值
        converged = agg[agg["eps2_max"] < THRESHOLD]
        if not converged.empty:
            best_p = int(converged["param"].min())
            best_row = agg[agg["param"] == best_p].iloc[0]
            recommendations[method] = best_p
            print(f"\n  >> 推荐 {param_label}* = {best_p} "
                  f"(eps2_max = {best_row['eps2_max']:.2e}, "
                  f"rt_200 = {best_row['rt200_mean']:.4f}s)")
        else:
            # 未达到阈值，推荐能收敛到的最好值（eps2 最小值）
            best_idx = agg["eps2_max"].idxmin()
            best_row = agg.iloc[best_idx]
            best_p = int(best_row["param"])
            recommendations[method] = best_p
            print(f"\n  >> 未达到 {THRESHOLD:.0e} 阈值，推荐 {param_label}* = {best_p} "
                  f"(eps2_max = {best_row['eps2_max']:.2e})")

    # ---- 收敛曲线图 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 左：L2 误差 vs 参数
    ax = axes[0]
    for method, marker, label in [("krylov", "o-", "Krylov (m)"),
                                   ("chebyshev", "s--", "Chebyshev (d)")]:
        agg = _agg(df, method)
        if agg.empty:
            continue
        ax.semilogy(agg["param"], agg["eps2_mean"], marker, label=label,
                    markersize=6, linewidth=1.5)
    ax.axhline(THRESHOLD, color="gray", linestyle=":", linewidth=1,
               label=f"threshold ({THRESHOLD:.0e})")
    ax.set_xlabel("Parameter (m or d)")
    ax.set_ylabel("L2 Error  ‖P_approx − P_exact‖₂")
    ax.set_title("Convergence: Error vs Parameter")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 中：运行时间 vs 参数
    ax = axes[1]
    for method, marker, label in [("krylov", "o-", "Krylov n=200"),
                                   ("krylov", "o--", "Krylov n=500"),
                                   ("chebyshev", "s-", "Chebyshev n=200"),
                                   ("chebyshev", "s--", "Chebyshev n=500")]:
        agg = _agg(df, method.split()[0])
        if agg.empty:
            continue
        col = "rt200_mean" if "200" in label else "rt500_mean"
        ls = "--" if "--" in label else "-"
        ax.plot(agg["param"], agg[col], marker[0] + ls,
                label=label, markersize=5, linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Parameter (m or d)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime vs Parameter")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 右：Top-10 一致性 vs 参数
    ax = axes[2]
    for method, marker, label in [("krylov", "o-", "Krylov"),
                                   ("chebyshev", "s--", "Chebyshev")]:
        agg = _agg(df, method)
        if agg.empty:
            continue
        ax.plot(agg["param"], agg["top10_min"], marker, label=label,
                markersize=6, linewidth=1.5)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Parameter (m or d)")
    ax.set_ylabel("Top-10 Node Overlap")
    ax.set_title("Top-10 Consistency")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("exp6 Pre-Tune: Krylov & Chebyshev Parameter Selection",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = RESULTS_DIR / "pre_tune_convergence.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n收敛曲线图已保存: {fig_path}")
    plt.close(fig)

    # ---- 保存推荐到文本文件 ----
    rec_path = RESULTS_DIR / "recommended_params.txt"
    with open(rec_path, "w", encoding="utf-8") as f:
        f.write(f"# exp6 推荐参数（基于前置实验）\n")
        f.write(f"# 收敛阈值: eps2 < {THRESHOLD}\n\n")
        for method, val in recommendations.items():
            f.write(f"{method}_param = {val}\n")
        f.write("\n# 建议在 exp6_large_scale_approx.py 的 EVOLUTION_PRESETS 中使用:\n")
        if "krylov" in recommendations:
            f.write(f"#   krylov: m = {recommendations['krylov']}\n")
        if "chebyshev" in recommendations:
            f.write(f"#   chebyshev: d = {recommendations['chebyshev']}\n")
    print(f"推荐参数已保存: {rec_path}")


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="exp6 前置实验：Krylov m 与 Chebyshev d 参数调优")
    parser.add_argument("--smoke", action="store_true",
                        help="烟雾测试：仅 3 个参数值 + 1 次重复")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help=f"单次计算的超时门限（秒），默认 {DEFAULT_TIMEOUT}")
    args = parser.parse_args()

    krylov_list = KRYLOV_DIMS[:3] if args.smoke else KRYLOV_DIMS
    cheb_list = CHEB_DEGREES[:3] if args.smoke else CHEB_DEGREES

    print("exp6 前置实验：Krylov / Chebyshev 参数调优")
    print(f"  测试图:       n=200 + n=500 各一张")
    print(f"  Krylov m:     {krylov_list}")
    print(f"  Chebyshev d:  {cheb_list}")
    print(f"  重复次数:     {1 if args.smoke else N_REPEAT}")
    print(f"  超时门限:     {args.timeout}s")
    print(f"  结果目录:     {RESULTS_DIR}")
    print()

    t_start = time.perf_counter()
    df = run_pre_experiment(smoke=args.smoke, timeout=args.timeout)
    elapsed = time.perf_counter() - t_start

    csv_path = RESULTS_DIR / "pre_tune_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n原始数据已保存: {csv_path}  ({len(df)} 条记录, 耗时 {elapsed:.0f}s)")

    if not df.empty:
        analyze_and_recommend(df)

    print(f"\n前置实验完成。")


if __name__ == "__main__":
    main()
