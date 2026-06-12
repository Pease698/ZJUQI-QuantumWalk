#!/usr/bin/env python3
"""MC medium 结果细分指标分析（D2）。

目的：
  在已有的 medium 扫描结果上，按 (n, p, k) 分组深挖：
    1. Quantum 在哪些子场景上接近 / 超过 ClassicalClique？
    2. Recall（找到的团节点占植入团的比例）
    3. 成功率（找到等于或超过植入团大小 k 的实例比例）
    4. 按图规模 / 背景密度 / 团大小分组的胜负分布

数据来源:
  results/tune_quantum_params/maximum_clique_medium_full_results.csv

输出:
  results/d2_mc_breakdown/
    - subgroup_summary.csv     按 (n, p, k) 分组的汇总
    - win_loss_matrix.csv      Quantum vs Baseline 的胜负配对
    - quantum_recall.csv       recall 指标
"""

import io
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Windows UTF-8
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 输入：medium MC 扫描结果
INPUT_CSV = (PROJECT_ROOT / "results" / "tune_quantum_params"
             / "maximum_clique_medium_full_results.csv")
OUTPUT_DIR = PROJECT_ROOT / "results" / "d2_mc_breakdown"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_sample_id(sid: str) -> dict:
    """从 sample_id 解析图参数。

    格式: mc_n{N}_p{P}_k{K}_{idx}
      mc_n100_p02_k10_000 → n=100, p=0.2, k=10, idx=0
    """
    m = re.match(r"mc_n(\d+)_p(\d+)_k(\d+)_(\d+)", sid)
    if not m:
        return {"n": np.nan, "p": np.nan, "k": np.nan, "idx": np.nan}
    n = int(m.group(1))
    p_raw = m.group(2)
    # p_raw 是去掉小数点的字符串：'01' -> 0.1, '02' -> 0.2, '005' -> 0.05, '03' -> 0.3
    # 规则：第一位是个位（通常 0），其余位是十/百分位
    if p_raw.startswith("0") and len(p_raw) >= 2:
        # '01' -> 0.1, '02' -> 0.2, '005' -> 0.05
        p = float("0." + p_raw[1:])
    else:
        p = float("0." + p_raw)
    k = int(m.group(3))
    idx = int(m.group(4))
    return {"n": n, "p": p, "k": k, "idx": idx}


def load_data() -> pd.DataFrame:
    """加载 medium 扫描数据并解析参数。"""
    if not INPUT_CSV.exists():
        print(f"错误: 找不到 {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"加载 {len(df)} 条记录从 {INPUT_CSV.name}")

    # 解析 sample_id
    parsed = df["sample_id"].apply(parse_sample_id).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    print(f"算法分布:")
    for algo, sub in df.groupby("algorithm"):
        if algo == "QuantumGuidedGreedy":
            for alpha, ssub in sub.groupby("alpha"):
                print(f"  {algo} (α={alpha}): {len(ssub)} 条")
        else:
            print(f"  {algo}: {len(sub)} 条")
    print()
    return df


# ============================================================
# 分析 1：按 (n, p, k) 分组的均值对比
# ============================================================

def subgroup_analysis(df: pd.DataFrame):
    """按 (n, p, k) 分组，对比 Quantum 与基线的均值。"""
    print("=" * 80)
    print("分析 1：按 (n, p, k) 分组的均值对比")
    print("=" * 80)

    # 基线索引
    base = df[df["algorithm"] == "ClassicalClique"]
    base_means = base.groupby(["n", "p", "k"])["objective"].mean()

    # Quantum 各 α
    rows = []
    for alpha in [0.0, 0.5, 1.0]:
        q = df[(df["algorithm"] == "QuantumGuidedGreedy") &
               (df["alpha"] == alpha)]
        q_means = q.groupby(["n", "p", "k"])["objective"].mean()
        for (n, p, k), q_val in q_means.items():
            b_val = base_means.get((n, p, k), np.nan)
            rows.append({
                "n": n, "p": p, "k": k,
                "alpha": alpha,
                "baseline_mean": b_val,
                "quantum_mean": q_val,
                "diff": q_val - b_val,
                "quantum_better": q_val > b_val,
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "subgroup_summary.csv",
                   index=False, encoding="utf-8")

    # 打印每个 alpha 下，Quantum 严格胜出的子组
    print("\n在哪些 (n, p, k) 子组上 Quantum > 基线？")
    for alpha in [0.0, 0.5, 1.0]:
        sub = summary[summary["alpha"] == alpha]
        wins = sub[sub["diff"] > 0]
        ties = sub[sub["diff"] == 0]
        losses = sub[sub["diff"] < 0]
        print(f"\n  α={alpha}:")
        print(f"    胜出子组数:   {len(wins)} / {len(sub)}")
        print(f"    持平子组数:   {len(ties)} / {len(sub)}")
        print(f"    落败子组数:   {len(losses)} / {len(sub)}")
        if not wins.empty:
            print(f"    胜出子组:")
            for _, r in wins.sort_values("diff", ascending=False).iterrows():
                print(f"      n={int(r['n'])}, p={r['p']:.2f}, k={int(r['k'])}:  "
                      f"基线 {r['baseline_mean']:.3f} → Q {r['quantum_mean']:.3f}  "
                      f"(+{r['diff']:.3f})")


# ============================================================
# 分析 2：胜率（按 sample_id × run_id 配对）
# ============================================================

def win_loss_analysis(df: pd.DataFrame):
    """逐对比较 Quantum vs Baseline，按 n 分组算胜率。"""
    print("\n" + "=" * 80)
    print("分析 2：逐对胜率")
    print("=" * 80)

    base = df[df["algorithm"] == "ClassicalClique"]
    base_lookup = {(r["sample_id"], r["run_id"]): r["objective"]
                   for _, r in base.iterrows()}

    rows = []
    for alpha in [0.0, 0.5, 1.0]:
        q = df[(df["algorithm"] == "QuantumGuidedGreedy") &
               (df["alpha"] == alpha)]

        # 按 n 分组算胜率
        for n_val, sub in q.groupby("n"):
            wins = 0
            ties = 0
            losses = 0
            for _, r in sub.iterrows():
                key = (r["sample_id"], r["run_id"])
                if key not in base_lookup:
                    continue
                b = base_lookup[key]
                if r["objective"] > b:
                    wins += 1
                elif r["objective"] == b:
                    ties += 1
                else:
                    losses += 1
            total = wins + ties + losses
            rows.append({
                "alpha": alpha,
                "n": int(n_val),
                "n_pairs": total,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_rate": wins / total if total > 0 else 0,
                "tie_rate": ties / total if total > 0 else 0,
                "loss_rate": losses / total if total > 0 else 0,
            })

    win_df = pd.DataFrame(rows)
    win_df.to_csv(OUTPUT_DIR / "win_loss_by_n.csv",
                  index=False, encoding="utf-8")

    print("\n按图规模 n 分组的胜率（Quantum vs ClassicalClique）：")
    print(f"{'α':>5} {'n':>5} {'对数':>6} {'胜':>6} {'平':>6} {'负':>6}  "
          f"{'胜率':>8} {'平局率':>8} {'败率':>8}")
    print("-" * 76)
    for _, r in win_df.iterrows():
        print(f"{r['alpha']:>5.2f} {r['n']:>5} {r['n_pairs']:>6} "
              f"{r['wins']:>6} {r['ties']:>6} {r['losses']:>6}  "
              f"{r['win_rate']:>7.1%} {r['tie_rate']:>7.1%} "
              f"{r['loss_rate']:>7.1%}")


# ============================================================
# 分析 3：Recall 与成功率
# ============================================================

def recall_analysis(df: pd.DataFrame):
    """Recall = objective / k；成功率 = (objective ≥ k) 比例。"""
    print("\n" + "=" * 80)
    print("分析 3：Recall 与成功率")
    print("=" * 80)

    # 整理：每个算法在每个 (n, p, k) 子组的 recall 均值和成功率
    df = df.copy()
    df["recall"] = df["objective"] / df["k"]
    df["found_planted"] = (df["objective"] >= df["k"]).astype(int)

    rows = []
    for algo in ["ClassicalClique"]:
        sub = df[df["algorithm"] == algo]
        rows.append({
            "label": algo,
            "alpha": np.nan,
            "recall_mean": float(sub["recall"].mean()),
            "success_rate": float(sub["found_planted"].mean()),
            "n_samples": len(sub),
        })

    for alpha in [0.0, 0.5, 1.0]:
        sub = df[(df["algorithm"] == "QuantumGuidedGreedy") &
                 (df["alpha"] == alpha)]
        rows.append({
            "label": f"QuantumGreedy (α={alpha})",
            "alpha": alpha,
            "recall_mean": float(sub["recall"].mean()),
            "success_rate": float(sub["found_planted"].mean()),
            "n_samples": len(sub),
        })

    rec_df = pd.DataFrame(rows)
    rec_df.to_csv(OUTPUT_DIR / "recall_summary.csv",
                  index=False, encoding="utf-8")

    print(f"\n{'算法':<28} {'Recall':>10} {'成功率':>10} {'样本数':>8}")
    print("-" * 60)
    for _, r in rec_df.iterrows():
        print(f"{r['label']:<28} {r['recall_mean']:>9.3f}  "
              f"{r['success_rate']:>9.1%}  {r['n_samples']:>8}")

    # 也按 n 分组看 recall 趋势
    print(f"\n按 n 分组的 recall 均值：")
    print(f"{'n':>5} {'baseline':>12} {'Q(α=0)':>12} "
          f"{'Q(α=0.5)':>12} {'Q(α=1.0)':>12}")
    print("-" * 60)
    for n_val in sorted(df["n"].unique()):
        if np.isnan(n_val):
            continue
        sub_n = df[df["n"] == n_val]
        b = sub_n[sub_n["algorithm"] == "ClassicalClique"]["recall"].mean()
        q0 = sub_n[(sub_n["algorithm"] == "QuantumGuidedGreedy") &
                   (sub_n["alpha"] == 0.0)]["recall"].mean()
        q5 = sub_n[(sub_n["algorithm"] == "QuantumGuidedGreedy") &
                   (sub_n["alpha"] == 0.5)]["recall"].mean()
        q1 = sub_n[(sub_n["algorithm"] == "QuantumGuidedGreedy") &
                   (sub_n["alpha"] == 1.0)]["recall"].mean()
        print(f"{int(n_val):>5} {b:>11.3f} {q0:>11.3f} "
              f"{q5:>11.3f} {q1:>11.3f}")


# ============================================================
# 分析 4：按 p 分组（看是否在高噪声/低噪声下表现不同）
# ============================================================

def density_analysis(df: pd.DataFrame):
    """按背景密度 p 分组对比。"""
    print("\n" + "=" * 80)
    print("分析 4：按背景密度 p 分组对比")
    print("=" * 80)

    base = df[df["algorithm"] == "ClassicalClique"]

    print(f"\n{'p':>6} {'baseline_mean':>14} ", end="")
    for alpha in [0.0, 0.5, 1.0]:
        print(f"{f'Q(α={alpha})':>12} ", end="")
    print(f"{'Q-base (α=0.5)':>16}")
    print("-" * 76)

    for p_val in sorted(df["p"].dropna().unique()):
        base_p = base[base["p"] == p_val]["objective"].mean()
        print(f"{p_val:>6.2f} {base_p:>14.3f} ", end="")
        diffs = {}
        for alpha in [0.0, 0.5, 1.0]:
            q = df[(df["algorithm"] == "QuantumGuidedGreedy") &
                   (df["alpha"] == alpha) &
                   (df["p"] == p_val)]["objective"].mean()
            print(f"{q:>12.3f} ", end="")
            diffs[alpha] = q - base_p
        print(f"{diffs.get(0.5, 0):>+16.3f}")


def main():
    print(f"D2 细分指标分析\n")
    df = load_data()

    subgroup_analysis(df)
    win_loss_analysis(df)
    recall_analysis(df)
    density_analysis(df)

    print(f"\n所有结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
