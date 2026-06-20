#!/usr/bin/env python3
"""实验六：大规模图 CTQW 近似方法对比。

分两部分独立运行（通过 --mode 切换）：

  Part A — embedded:  嵌入式量子引导贪心，对比 Krylov(m=30) 和 Chebyshev(d=50)
                      两种近似方法 + 3 种经典基线
  Part B — external:   Multi-Start 外置起点选择，3 种种子策略 × K × 2 种近似

数据范围（写死在脚本常量中，无需 CLI 指定）：
  - 人工数据：仅 n ∈ {300, 500}（由 ARTIFICIAL_N_VALUES 控制）
  - DIMACS 外部数据：5 个指定数据集（由 DIMACS_WHITELIST 控制）

用法:
  # 烟雾测试
  python3 experiments/exp6_large_scale_approx.py --mode embedded --smoke
  python3 experiments/exp6_large_scale_approx.py --mode external --smoke

  # Part A: 嵌入式方案（人工数据 n=300,500）
  python3 experiments/exp6_large_scale_approx.py --mode embedded

  # Part A: 嵌入式方案（DIMACS 外部验证）
  python3 experiments/exp6_large_scale_approx.py --mode embedded --data-source dimacs

  # Part B: 外置方案（人工数据）
  python3 experiments/exp6_large_scale_approx.py --mode external

  # Part B: 外置方案（DIMACS）
  python3 experiments/exp6_large_scale_approx.py \\
      --mode external --data-source dimacs --K-values 10 20
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_utils import load_instance, GraphInstance
from src.candidate_set import CliqueCandidateSet
from src.scoring import ClassicalCliqueScorer, ClassicalDegreeScorer
from src.algorithms.base import BaseAlgorithm
from src.algorithms.classical_greedy import ClassicalGreedy
from src.algorithms.simulated_annealing import SimulatedAnnealing
from src.algorithms.quantum_greedy import QuantumGuidedGreedy
from src.algorithms.multi_start_ctqw import (
    MultiStartCTQWGreedy, MultiStartRandomGreedy, MultiStartDegreeGreedy,
    MultiStartHybridSeedGreedy, hybrid_seed_scores)
from src.ctqw_evolution import compute_ctqw_evolution
from src.metrics import mean_std
from src.config import get_data_dirs, DATA_DIR, ensure_results_dir
from src.timeout import run_with_timeout

# ============================================================
# 配置常量
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp6_large_scale"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 大规模数据运行次数（人工每实例 2 次 = 每组 10 个数据点）
REPEAT_ARTIFICIAL = 2
REPEAT_DIMACS = 3

# 固定 CTQW 参数
FIXED_T = 1.0
FIXED_LAM = 0.5
FIXED_ALPHA = 0.5
FIXED_INIT = "max_degree"

# 演化方法预设 — 仅保留 Krylov m=30 和 Chebyshev d=50 两种近似方法
EVOLUTION_PRESETS = {
    "krylov_m30": {"method": "krylov", "krylov_dim": 30},
    "cheb_d50":   {"method": "chebyshev", "cheb_degree": 50},
}

# 人工数据：仅运行 n=300 和 n=500 的图
ARTIFICIAL_N_VALUES = {300, 500}

# DIMACS 外部数据：手选 5 个代表性数据集（3 个较小 + 2 个较大）
#
# 名称格式：填写 DIMACS 数据集目录名（即 .mtx 文件所在目录的名称）。
# 脚本内部会自动拼接为 "ext_mc_{名称}.json" 去匹配 data/external/maximum_clique/ 下的文件。
# 例如：填写 "C250-9" → 加载 ext_mc_C250-9.json
DIMACS_WHITELIST = {
    "gen200-p0-9-44",   # n=200, |E|≈1.8K
    "C250-9",            # n=250, |E|≈2.8K
    "p-hat300-3",        # n=300, |E|≈3.3K
    "C1000-9",           # n=1000, |E|≈45K
    "C2000-9",           # n=2000, |E|≈180K
}

# 颜色
COLORS = {
    "ClassicalDegree":          "#1f77b4",
    "ClassicalClique":          "#ff7f0e",
    "SimulatedAnnealing":       "#2ca02c",
    "QuantumGreedy_krylov_m30": "#9467bd",
    "QuantumGreedy_cheb_d50":   "#8c564b",
    "MS_Random":                "#7f7f7f",
    "MS_Degree":                "#bcbd22",
    "MS_CTQW_krylov_m30":       "#17becf",
    "MS_CTQW_cheb_d50":         "#e377c2",
    "MS_HybridSeed":            "#4c78a8",
}


class PrecomputedSeedMultiStartGreedy(BaseAlgorithm):
    """使用预计算 seed 排名的 Multi-Start 贪心。

    external CTQW 在同一图、同一演化方法下的概率排序与 K 无关，
    所以先算一次完整排名，再为 K=5/10/20/30 截取前 K 个 seed。
    """

    def __init__(self, seed_order: list[int], K: int,
                 method_tag: str, name: str | None = None,
                 beta: float | None = None):
        super().__init__(
            CliqueCandidateSet(), ClassicalCliqueScorer(),
            name=name or f"MS_CTQW(K={K},{method_tag})")
        self.seed_order = seed_order
        self.K = K
        self.method_tag = method_tag
        self.beta = beta

    def solve(self, instance: GraphInstance):
        t_start = time.perf_counter()
        seeds = self.seed_order[:min(self.K, len(self.seed_order))]

        best_solution: list[int] = []
        best_objective = -1.0
        per_seed_log: list[dict] = []
        total_iter = 0

        for v in seeds:
            inner = ClassicalGreedy(
                self.candidate_builder, ClassicalCliqueScorer(),
                name=f"_inner_seed{v}", start_node=v)
            result = inner.solve(instance)
            per_seed_log.append({
                "seed_node": v,
                "objective": result.objective,
                "solution_size": len(result.solution),
                "iterations": result.iterations,
            })
            total_iter += result.iterations
            if result.objective > best_objective:
                best_objective = result.objective
                best_solution = result.solution

        runtime = time.perf_counter() - t_start
        return self._build_result(
            instance=instance,
            solution=best_solution,
            objective=float(best_objective),
            runtime=runtime,
            iterations=total_iter,
            history=per_seed_log,
            extra_params={
                "K": self.K,
                "n_seeds_tried": len(seeds),
                "ctqw_seed_reused": True,
                "method_tag": self.method_tag,
                **({} if self.beta is None else {"beta": self.beta}),
            },
        )


# ============================================================
# 实例发现
# ============================================================

def _extract_n(dirname: str) -> int:
    m = re.search(r'_n(\d+)_', dirname)
    return int(m.group(1)) if m else 999


def discover_instances(
    data_source: str = "artificial",
    smoke: bool = False,
    limit: int | None = None,
    dimacs_labels: set[str] | None = None,
) -> list[tuple[GraphInstance, str]]:
    """发现大规模测试实例。

    artificial: 仅加载 n ∈ {300, 500} 的参数组（由 ARTIFICIAL_N_VALUES 控制）。
    misleading: 仅加载 degree-misleading 人工最大团样本。
    dimacs:    仅加载 DIMACS_WHITELIST 中指定的数据集。

    返回:
        [(instance, label), ...]
    """
    instances: list[tuple[GraphInstance, str]] = []

    if data_source == "artificial":
        dirs = get_data_dirs("maximum_clique")
        dirs = [d for d in dirs if _extract_n(d.name) in ARTIFICIAL_N_VALUES]
        if not dirs:
            print(f"警告: 未找到 n ∈ {ARTIFICIAL_N_VALUES} 的人工数据目录")
            return instances

        if smoke:
            dirs = dirs[:1]

        for d in dirs:
            json_files = sorted(d.glob("*.json"))
            if smoke:
                json_files = json_files[:1]
            for fpath in json_files:
                instances.append((load_instance(fpath), d.name))
                if limit is not None and len(instances) >= limit:
                    return instances

    elif data_source == "misleading":
        dirs = get_data_dirs("maximum_clique")
        dirs = [d for d in dirs if d.name.startswith("mc_mislead_")]
        if smoke:
            dirs = dirs[:1]
        for d in dirs:
            json_files = sorted(d.glob("*.json"))
            if smoke:
                json_files = json_files[:1]
            for fpath in json_files:
                instances.append((load_instance(fpath), d.name))
                if limit is not None and len(instances) >= limit:
                    return instances

    elif data_source == "dimacs":
        ext_dir = DATA_DIR / "external" / "maximum_clique"
        if not ext_dir.is_dir():
            print(f"错误: 外部数据目录不存在: {ext_dir}")
            return instances

        for fpath in sorted(ext_dir.glob("*.json")):
            inst = load_instance(fpath)
            stem = fpath.stem
            label = stem[7:] if stem.startswith("ext_mc_") else stem
            allowed = dimacs_labels if dimacs_labels is not None else DIMACS_WHITELIST
            if label not in allowed:
                continue
            if smoke:
                instances = [(inst, label)]
                break
            instances.append((inst, label))
            if limit is not None and len(instances) >= limit:
                break

    return instances


# ============================================================
# Part A: 嵌入式算法构造
# ============================================================

def build_embedded_algorithms(
    instance: GraphInstance,
    method_keys: list[str],
    seed: int = 0,
) -> dict[str, BaseAlgorithm]:
    """构造 Part A 的算法集合：经典基线 + 量子变体。"""
    builder = CliqueCandidateSet()
    algorithms: dict[str, BaseAlgorithm] = {}

    # 经典基线
    algorithms["ClassicalDegree"] = ClassicalGreedy(
        builder, ClassicalDegreeScorer(), name="ClassicalDegree")
    algorithms["ClassicalClique"] = ClassicalGreedy(
        builder, ClassicalCliqueScorer(), name="ClassicalClique")
    algorithms["SimulatedAnnealing"] = SimulatedAnnealing(
        builder, seed=seed, max_iterations=2000, name="SimulatedAnnealing")

    # 量子引导贪心 × 演化方法
    for key in method_keys:
        preset = EVOLUTION_PRESETS.get(key)
        if preset is None:
            continue
        # exact 仅在 n≤200 时启用
        if preset["method"] == "exact" and instance.num_nodes > 200:
            continue

        algo = QuantumGuidedGreedy(
            builder,
            t=FIXED_T, lam=FIXED_LAM, alpha=FIXED_ALPHA,
            init_method=FIXED_INIT, seed=seed,
            evolution_method=preset["method"],
            krylov_dim=preset.get("krylov_dim"),
            cheb_degree=preset.get("cheb_degree"),
            name=f"QuantumGreedy({key})",
        )
        algorithms[f"QuantumGreedy_{key}"] = algo

    return algorithms


# ============================================================
# Part B: 外置式算法构造
# ============================================================

def build_external_algorithms(
    instance: GraphInstance,
    method_keys: list[str],
    K_values: list[int],
    hybrid_betas: list[float] | None = None,
    seed: int = 0,
    reuse_ctqw_seeds: bool = True,
) -> dict[str, BaseAlgorithm]:
    """构造 Part B 的算法集合：Multi-Start 系列。"""
    algorithms: dict[str, BaseAlgorithm] = {}
    hybrid_betas = hybrid_betas or []

    # 单起点基线
    algorithms["ClassicalClique"] = ClassicalGreedy(
        CliqueCandidateSet(), ClassicalCliqueScorer(), name="ClassicalClique")

    # Multi-Start × K（不涉及 CTQW）
    for K in K_values:
        algorithms[f"MS_Random_K{K}"] = MultiStartRandomGreedy(
            K=K, seed=seed, name=f"MS_Random(K={K})")
        algorithms[f"MS_Degree_K{K}"] = MultiStartDegreeGreedy(
            K=K, seed=seed, name=f"MS_Degree(K={K})")

    # MultiStartCTQW × K × 演化方法。默认复用同一方法下的一次 CTQW 排名。
    seed_orders: dict[str, list[int]] = {}
    if reuse_ctqw_seeds and K_values:
        max_k = max(K_values)
        for key in method_keys:
            preset = EVOLUTION_PRESETS.get(key)
            if preset is None:
                continue
            if preset["method"] == "exact" and instance.num_nodes > 200:
                continue
            seed_orders[key] = compute_ctqw_seed_order(instance, key, max_k)

    for K in K_values:
        for key in method_keys:
            preset = EVOLUTION_PRESETS.get(key)
            if preset is None:
                continue
            if preset["method"] == "exact" and instance.num_nodes > 200:
                continue

            if key in seed_orders:
                algo = PrecomputedSeedMultiStartGreedy(
                    seed_order=seed_orders[key],
                    K=K,
                    method_tag=key,
                    name=f"MS_CTQW(K={K},{key})",
                )
            else:
                algo = MultiStartCTQWGreedy(
                    K=K, t=FIXED_T, seed=seed,
                    evolution_method=preset["method"],
                    krylov_dim=preset.get("krylov_dim"),
                    cheb_degree=preset.get("cheb_degree"),
                    name=f"MS_CTQW(K={K},{key})",
                )
            algorithms[f"MS_CTQW_K{K}_{key}"] = algo

    hybrid_seed_orders: dict[tuple[str, float], list[int]] = {}
    if reuse_ctqw_seeds and K_values and hybrid_betas:
        for key in method_keys:
            preset = EVOLUTION_PRESETS.get(key)
            if preset is None:
                continue
            if preset["method"] == "exact" and instance.num_nodes > 200:
                continue
            probs = compute_ctqw_probs(instance, key)
            degrees = np.asarray(instance.adjacency_sparse.sum(axis=1)).ravel()
            for beta in hybrid_betas:
                score = hybrid_seed_scores(probs, degrees, beta)
                hybrid_seed_orders[(key, beta)] = [
                    int(v) for v in np.argsort(score)[::-1]
                ]

    # MS_HybridSeed × K × beta × 演化方法。
    for K in K_values:
        for beta in hybrid_betas:
            for key in method_keys:
                preset = EVOLUTION_PRESETS.get(key)
                if preset is None:
                    continue
                if preset["method"] == "exact" and instance.num_nodes > 200:
                    continue

                beta_tag = _format_beta_tag(beta)
                if (key, beta) in hybrid_seed_orders:
                    algo = PrecomputedSeedMultiStartGreedy(
                        seed_order=hybrid_seed_orders[(key, beta)],
                        K=K,
                        method_tag=key,
                        beta=beta,
                        name=f"MS_HybridSeed(K={K},beta={beta:g},{key})",
                    )
                else:
                    algo = MultiStartHybridSeedGreedy(
                        K=K, beta=beta, t=FIXED_T, seed=seed,
                        evolution_method=preset["method"],
                        krylov_dim=preset.get("krylov_dim"),
                        cheb_degree=preset.get("cheb_degree"),
                        name=f"MS_HybridSeed(K={K},beta={beta:g},{key})",
                    )
                algorithms[f"MS_HybridSeed_K{K}_b{beta_tag}_{key}"] = algo

    return algorithms


def compute_ctqw_seed_order(instance: GraphInstance, method_key: str,
                            max_k: int | None = None) -> list[int]:
    """计算 external CTQW 的全图 seed 排名。

    使用稀疏 CSR 邻接矩阵做矩阵-向量乘法，避免为每个 K 重复演化。
    """
    preset = EVOLUTION_PRESETS[method_key]
    n = instance.num_nodes
    psi0 = np.ones(n, dtype=np.complex128) / np.sqrt(n)
    H = instance.adjacency_sparse
    psi_t = compute_ctqw_evolution(
        H, psi0, FIXED_T,
        method=preset["method"],
        krylov_dim=preset.get("krylov_dim"),
        cheb_degree=preset.get("cheb_degree"),
    )
    probs = np.abs(psi_t) ** 2
    order = np.argsort(probs)[::-1]
    if max_k is not None:
        order = order[:max_k]
    return [int(v) for v in order]


def compute_hybrid_seed_order(instance: GraphInstance, method_key: str,
                              beta: float,
                              max_k: int | None = None) -> list[int]:
    """计算 CTQW + Degree 融合 seed 排名。"""
    probs = compute_ctqw_probs(instance, method_key)
    degrees = np.asarray(instance.adjacency_sparse.sum(axis=1)).ravel()
    score = hybrid_seed_scores(probs, degrees, beta)
    order = np.argsort(score)[::-1]
    if max_k is not None:
        order = order[:max_k]
    return [int(v) for v in order]


def compute_ctqw_probs(instance: GraphInstance, method_key: str) -> np.ndarray:
    """计算 external CTQW 全图概率向量。"""
    preset = EVOLUTION_PRESETS[method_key]
    n = instance.num_nodes
    psi0 = np.ones(n, dtype=np.complex128) / np.sqrt(n)
    psi_t = compute_ctqw_evolution(
        instance.adjacency_sparse, psi0, FIXED_T,
        method=preset["method"],
        krylov_dim=preset.get("krylov_dim"),
        cheb_degree=preset.get("cheb_degree"),
    )
    return np.abs(psi_t) ** 2


def _format_beta_tag(beta: float) -> str:
    """把 beta 转为稳定的 algo_key 片段。"""
    return f"{beta:g}".replace(".", "p")


# ============================================================
# 实验运行
# ============================================================

def run_experiment(
    instances: list[tuple[GraphInstance, str]],
    mode: str,
    method_keys: list[str],
    K_values: list[int] | None,
    hybrid_betas: list[float] | None,
    repeat: int,
    timeout_sec: float,
    smoke: bool,
    checkpoint_path: Path | None = None,
    workers: int = 1,
    resume: bool = False,
    retry_timeouts: bool = False,
) -> pd.DataFrame:
    """运行实验并返回汇总 DataFrame。"""
    all_rows: list[dict] = []
    total_instances = len(instances)
    t_start = time.perf_counter()

    print(f"\n{'=' * 60}")
    print(f"实验六 Part {'A' if mode == 'embedded' else 'B'}: "
          f"{'嵌入式' if mode == 'embedded' else '外置式'}方案")
    print(f"  实例数:   {total_instances}")
    print(f"  方法数:   {len(method_keys)}")
    print(f"  重复:     {repeat}")
    print(f"  超时:     {timeout_sec}s")
    print(f"  workers:  {workers}")
    print(f"  resume:   {resume}")
    print(f"  retry_to: {retry_timeouts}")
    if mode == "external":
        print(f"  K 值:     {K_values}")
        print(f"  beta 值:  {hybrid_betas or []}")
    print(f"{'=' * 60}")

    completed_keys: set[tuple[str, str, int]] = set()
    if resume and checkpoint_path is not None:
        completed_keys = _load_resume_rows(
            checkpoint_path, all_rows, retry_timeouts=retry_timeouts)
        if completed_keys:
            print(f"  已载入 checkpoint: {len(completed_keys)} 条已完成记录")

    if workers > 1:
        return _run_experiment_parallel(
            instances=instances,
            mode=mode,
            method_keys=method_keys,
            K_values=K_values,
            hybrid_betas=hybrid_betas,
            repeat=repeat,
            timeout_sec=timeout_sec,
            smoke=smoke,
            checkpoint_path=checkpoint_path,
            workers=workers,
            t_start=t_start,
            initial_rows=all_rows,
            completed_keys=completed_keys,
            retry_timeouts=retry_timeouts,
        )

    for idx, (inst, label) in enumerate(instances):
        n = inst.num_nodes
        expected_keys = _expected_algo_keys(
            mode, method_keys, K_values or [5, 10], hybrid_betas or [], inst)
        expected_tasks = {
            (inst.sample_id, key, run_id)
            for run_id in range(1 if smoke else repeat)
            for key in expected_keys
        }
        if expected_tasks and expected_tasks.issubset(completed_keys):
            continue

        if mode == "embedded":
            algos = build_embedded_algorithms(inst, method_keys, seed=0)
        else:
            algos = build_external_algorithms(inst, method_keys,
                                              K_values or [5, 10],
                                              hybrid_betas=hybrid_betas,
                                              seed=0)

        for run_id in range(1 if smoke else repeat):
            seed = run_id

            for algo_name, algo_template in algos.items():
                task_key = (inst.sample_id, algo_name, run_id)
                if task_key in completed_keys:
                    continue
                # 重建算法对象（使用正确的 seed）
                algo = _rebuild_with_seed(algo_template, seed, mode)

                t0 = time.perf_counter()
                result = run_with_timeout(algo, inst, timeout_sec)
                wall = time.perf_counter() - t0

                row = result.to_dict()
                row["source_label"] = label
                row["algo_key"] = algo_name
                row["run_id"] = run_id
                row["seed"] = seed
                row["wall_time"] = wall
                row["n"] = n
                row["mode"] = mode

                # 解析标签
                row["family"] = _classify_family(algo_name)
                if mode == "embedded":
                    row["method_tag"] = _extract_ev_tag(algo_name)
                else:
                    row["method_tag"] = _extract_ev_tag(algo_name)
                    row["K"] = _extract_K(algo_name)

                all_rows.append(row)

        # 进度报告
        elapsed = time.perf_counter() - t_start
        if (idx + 1) % max(total_instances // 5, 1) == 0 or idx == total_instances - 1:
            n_timeout = sum(1 for r in all_rows if r.get("timed_out"))
            n_total = len(all_rows)
            print(f"  [{idx + 1:3d}/{total_instances}] 已完成, "
                  f"耗时 {elapsed:.0f}s, "
                  f"记录 {n_total} 条"
                  + (f", 超时 {n_timeout} 条" if n_timeout > 0 else ""))
        if checkpoint_path is not None and all_rows:
            pd.DataFrame(all_rows).to_csv(
                checkpoint_path, index=False, encoding="utf-8")

    df = pd.DataFrame(all_rows)
    elapsed_total = time.perf_counter() - t_start
    print(f"\n全部完成，总耗时 {elapsed_total:.0f}s，共 {len(df)} 条记录")
    if "timed_out" in df.columns:
        n_to = int(df["timed_out"].sum())
        if n_to > 0:
            print(f"  其中超时: {n_to} / {len(df)} ({n_to / len(df) * 100:.1f}%)")

    return df


def _expected_algo_keys(mode: str, method_keys: list[str],
                        K_values: list[int], hybrid_betas: list[float],
                        instance: GraphInstance) -> list[str]:
    """无需构造算法即可得到预期 algo_key 列表，用于 resume 快速跳过。"""
    if mode == "embedded":
        keys = ["ClassicalDegree", "ClassicalClique", "SimulatedAnnealing"]
        for key in method_keys:
            preset = EVOLUTION_PRESETS.get(key)
            if preset is None:
                continue
            if preset["method"] == "exact" and instance.num_nodes > 200:
                continue
            keys.append(f"QuantumGreedy_{key}")
        return keys

    keys = ["ClassicalClique"]
    for K in K_values:
        keys.extend([f"MS_Random_K{K}", f"MS_Degree_K{K}"])
        for key in method_keys:
            preset = EVOLUTION_PRESETS.get(key)
            if preset is None:
                continue
            if preset["method"] == "exact" and instance.num_nodes > 200:
                continue
            keys.append(f"MS_CTQW_K{K}_{key}")
        for beta in hybrid_betas:
            beta_tag = _format_beta_tag(beta)
            for key in method_keys:
                preset = EVOLUTION_PRESETS.get(key)
                if preset is None:
                    continue
                if preset["method"] == "exact" and instance.num_nodes > 200:
                    continue
                keys.append(f"MS_HybridSeed_K{K}_b{beta_tag}_{key}")
    return keys


def _task_key_from_row(row: pd.Series | dict) -> tuple[str, str, int] | None:
    try:
        return (str(row["sample_id"]), str(row["algo_key"]), int(row["run_id"]))
    except Exception:
        return None


def _load_resume_rows(checkpoint_path: Path,
                      rows_out: list[dict],
                      retry_timeouts: bool = False) -> set[tuple[str, str, int]]:
    """加载已有 full/partial CSV，并返回已完成任务键。"""
    candidates = [checkpoint_path.with_name("full_results.csv"), checkpoint_path]
    for path in candidates:
        if not path.exists():
            continue
        old = pd.read_csv(path)
        if old.empty:
            continue
        if retry_timeouts and "timed_out" in old.columns:
            timed_out = old["timed_out"].astype(str).str.lower().eq("true")
            kept = old[~timed_out].copy()
        else:
            kept = old
        rows_out.extend(kept.to_dict("records"))
        keys = set()
        for _, row in kept.iterrows():
            key = _task_key_from_row(row)
            if key is not None:
                keys.add(key)
        return keys
    return set()


def _run_experiment_parallel(
    instances: list[tuple[GraphInstance, str]],
    mode: str,
    method_keys: list[str],
    K_values: list[int] | None,
    hybrid_betas: list[float] | None,
    repeat: int,
    timeout_sec: float,
    smoke: bool,
    checkpoint_path: Path | None,
    workers: int,
    t_start: float,
    initial_rows: list[dict] | None = None,
    completed_keys: set[tuple[str, str, int]] | None = None,
    retry_timeouts: bool = False,
) -> pd.DataFrame:
    """并行运行实验。

    外层使用线程池调度 solve 任务；每个任务内部仍调用 run_with_timeout，
    由独立子进程隔离求解和超时。这样避免 Windows 下嵌套进程池的复杂性。
    """
    completed_keys = completed_keys or set()
    tasks = []
    for idx, (inst, label) in enumerate(instances):
        expected_keys = _expected_algo_keys(
            mode, method_keys, K_values or [5, 10], hybrid_betas or [], inst)
        expected_tasks = {
            (inst.sample_id, key, run_id)
            for run_id in range(1 if smoke else repeat)
            for key in expected_keys
        }
        if expected_tasks and expected_tasks.issubset(completed_keys):
            continue

        if mode == "embedded":
            algos = build_embedded_algorithms(inst, method_keys, seed=0)
        else:
            algos = build_external_algorithms(inst, method_keys,
                                              K_values or [5, 10],
                                              hybrid_betas=hybrid_betas,
                                              seed=0)

        for run_id in range(1 if smoke else repeat):
            for algo_name, algo_template in algos.items():
                task_key = (inst.sample_id, algo_name, run_id)
                if task_key in completed_keys:
                    continue
                tasks.append((idx, inst, label, algo_name, algo_template, run_id))

    total_tasks = len(tasks)
    all_rows: list[dict] = list(initial_rows or [])
    if total_tasks == 0:
        print("  所有任务均已完成，直接使用已有结果。")
        return pd.DataFrame(all_rows)
    completed_instances: set[int] = set()
    report_every = max(total_tasks // 10, 1)

    def _solve_task(task):
        idx, inst, label, algo_name, algo_template, run_id = task
        seed = run_id
        algo = _rebuild_with_seed(algo_template, seed, mode)

        t0 = time.perf_counter()
        result = run_with_timeout(algo, inst, timeout_sec)
        wall = time.perf_counter() - t0

        row = result.to_dict()
        row["source_label"] = label
        row["algo_key"] = algo_name
        row["run_id"] = run_id
        row["seed"] = seed
        row["wall_time"] = wall
        row["n"] = inst.num_nodes
        row["mode"] = mode
        row["family"] = _classify_family(algo_name)
        row["method_tag"] = _extract_ev_tag(algo_name)
        if mode == "external":
            row["K"] = _extract_K(algo_name)
        return idx, row

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_solve_task, task): task for task in tasks}
        for done_idx, future in enumerate(as_completed(future_map), 1):
            idx, row = future.result()
            all_rows.append(row)
            completed_instances.add(idx)

            if checkpoint_path is not None:
                pd.DataFrame(all_rows).to_csv(
                    checkpoint_path, index=False, encoding="utf-8")

            if done_idx % report_every == 0 or done_idx == total_tasks:
                elapsed = time.perf_counter() - t_start
                n_timeout = sum(1 for r in all_rows if r.get("timed_out"))
                print(f"  [{done_idx:4d}/{total_tasks}] task 完成, "
                      f"实例 {len(completed_instances)}/{len(instances)}, "
                      f"耗时 {elapsed:.0f}s, 记录 {len(all_rows)} 条"
                      + (f", 超时 {n_timeout} 条" if n_timeout > 0 else ""))

    df = pd.DataFrame(all_rows)
    elapsed_total = time.perf_counter() - t_start
    print(f"\n全部完成，总耗时 {elapsed_total:.0f}s，共 {len(df)} 条记录")
    if "timed_out" in df.columns:
        n_to = int(df["timed_out"].sum())
        if n_to > 0:
            print(f"  其中超时: {n_to} / {len(df)} ({n_to / len(df) * 100:.1f}%)")
    return df


def _rebuild_with_seed(template: BaseAlgorithm, seed: int,
                       mode: str) -> BaseAlgorithm:
    """根据模板创建带新 seed 的副本。"""
    name = template.name
    builder = template.candidate_builder

    if isinstance(template, ClassicalGreedy):
        return ClassicalGreedy(builder, template.scorer, name=name,
                               start_node=template.start_node)

    elif isinstance(template, SimulatedAnnealing):
        return SimulatedAnnealing(
            builder, seed=seed, T0=template.T0,
            cooling_rate=template.cooling_rate,
            max_iterations=template.max_iterations, name=name)

    elif isinstance(template, QuantumGuidedGreedy):
        return QuantumGuidedGreedy(
            builder, t=template.t, lam=template.lam,
            alpha=template.alpha, init_method=template.init_method,
            seed=seed,
            evolution_method=template.evolution_method,
            krylov_dim=template.krylov_dim,
            cheb_degree=template.cheb_degree, name=name)

    elif isinstance(template, MultiStartCTQWGreedy):
        return MultiStartCTQWGreedy(
            K=template.K, t=template.t, seed=seed,
            evolution_method=template.evolution_method,
            krylov_dim=template.krylov_dim,
            cheb_degree=template.cheb_degree, name=name)

    elif isinstance(template, MultiStartHybridSeedGreedy):
        return MultiStartHybridSeedGreedy(
            K=template.K, beta=template.beta, t=template.t, seed=seed,
            evolution_method=template.evolution_method,
            krylov_dim=template.krylov_dim,
            cheb_degree=template.cheb_degree, name=name)

    elif isinstance(template, MultiStartRandomGreedy):
        return MultiStartRandomGreedy(K=template.K, seed=seed, name=name)

    elif isinstance(template, MultiStartDegreeGreedy):
        return MultiStartDegreeGreedy(K=template.K, seed=seed, name=name)

    elif isinstance(template, PrecomputedSeedMultiStartGreedy):
        return PrecomputedSeedMultiStartGreedy(
            seed_order=template.seed_order,
            K=template.K,
            method_tag=template.method_tag,
            beta=template.beta,
            name=name)

    return template


# ============================================================
# 标签解析
# ============================================================

def _classify_family(algo_name: str) -> str:
    for fam in ["ClassicalDegree", "ClassicalClique", "SimulatedAnnealing",
                 "QuantumGreedy", "MS_Random", "MS_Degree", "MS_CTQW",
                 "MS_HybridSeed"]:
        if algo_name.startswith(fam):
            return fam
    return algo_name.split("(")[0] if "(" in algo_name else algo_name[:20]


def _extract_ev_tag(algo_name: str) -> str:
    """提取演化方法标签：exact / krylov_m30 / cheb_d60 等。"""
    for key in EVOLUTION_PRESETS:
        if key in algo_name:
            return key
    return "N/A"


def _extract_K(algo_name: str) -> int:
    m = re.search(r'K(\d+)', algo_name)
    return int(m.group(1)) if m else 0


# ============================================================
# 分析
# ============================================================

def _display_name(row: pd.Series) -> str:
    """构造区分方法的显示名。

    量子变体按演化方法区分（krylov_m30 / cheb_d50 各自独立显示），
    经典基线保持不变。
    """
    fam = row["family"]
    tag = row.get("method_tag", "N/A")
    if fam == "MS_HybridSeed":
        beta = row.get("beta", None)
        beta_part = "bNA" if pd.isna(beta) else f"b{float(beta):g}"
        if pd.notna(tag) and tag and tag != "N/A":
            return f"{fam}_{beta_part}_{tag}"
        return f"{fam}_{beta_part}"
    if pd.notna(tag) and tag and tag != "N/A":
        # 简短标签：QG_krylov_m30, MS_CTQW_cheb_d50 等
        return f"{fam}_{tag}"
    return fam


def _short_label(display: str) -> str:
    """压缩显示名以便在图表中使用。"""
    return display \
        .replace("QuantumGreedy", "QG") \
        .replace("ClassicalDegree", "DegGreedy") \
        .replace("ClassicalClique", "CliqueGreedy") \
        .replace("SimulatedAnnealing", "SimAnneal") \
        .replace("MS_HybridSeed", "Hybrid") \
        .replace("MS_", "")


def analyze_and_plot(df: pd.DataFrame, mode: str, tag: str,
                     output_dir: Path | None = None):
    """分析实验结果并生成图表和 CSV。"""
    if df.empty:
        print("DataFrame 为空，跳过分析。")
        return

    out = output_dir if output_dir is not None else RESULTS_DIR / f"exp6_{mode}_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    # 过滤有效结果，并添加 display_name 列
    df = df.copy()
    df["display_name"] = df.apply(_display_name, axis=1)
    df_valid = df[df.get("timed_out", pd.Series(False, index=df.index)) == False].copy()

    # ---- 汇总表（按 display_name 区分方法） ----
    print(f"\n{'=' * 60}")
    print(f"实验结果汇总 ({mode}, {tag})")
    print(f"{'=' * 60}")

    # 量子方法按 method_tag 分开，经典方法按 family 聚合
    display_names = sorted(df_valid["display_name"].unique(),
                           key=lambda d: df_valid[df_valid["display_name"] == d]["objective"].mean(),
                           reverse=True)

    for dname in display_names:
        sub = df_valid[df_valid["display_name"] == dname]
        obj_mean, obj_std = mean_std(sub["objective"].tolist())
        rt_mean, rt_std = mean_std(sub["wall_time"].tolist())
        n_to = int(df[df["display_name"] == dname]["timed_out"].sum()) \
            if "timed_out" in df.columns else 0
        print(f"\n  {dname}:")
        print(f"    团大小:   {obj_mean:.4f} ± {obj_std:.4f}")
        print(f"    耗时:     {rt_mean:.4f}s ± {rt_std:.4f}")
        print(f"    样本数:   {len(sub)}" + (f" (超时 {n_to})" if n_to else ""))

    # ---- 保存完整结果 ----
    df.to_csv(out / "full_results.csv", index=False, encoding="utf-8")
    _write_summary_tables(df, out, mode)

    # ---- 箱线图 ----
    if mode == "embedded":
        _plot_embedded_boxplots(df_valid, out)
    else:
        _plot_external_boxplots(df_valid, out)
        _plot_external_heatmap(df_valid, out)

    print(f"\n结果目录: {out}")


def _write_summary_tables(df: pd.DataFrame, out: Path, mode: str) -> None:
    """写出报告友好的统计表。"""
    df = df.copy()
    if "display_name" not in df.columns:
        df["display_name"] = df.apply(_display_name, axis=1)
    if "timed_out" not in df.columns:
        df["timed_out"] = False
    df["timed_out_bool"] = df["timed_out"].astype(str).str.lower().eq("true")
    df["valid_objective"] = pd.to_numeric(df["objective"], errors="coerce")

    method_summary = (
        df.groupby(["mode", "family", "display_name"], dropna=False)
        .agg(
            runs=("sample_id", "size"),
            samples=("sample_id", "nunique"),
            mean_objective=("valid_objective", "mean"),
            median_objective=("valid_objective", "median"),
            std_objective=("valid_objective", "std"),
            median_wall_time=("wall_time", "median"),
            p95_wall_time=("wall_time", lambda s: s.quantile(0.95)),
            timeout_rate=("timed_out_bool", "mean"),
        )
        .reset_index()
        .sort_values(["mean_objective", "median_wall_time"], ascending=[False, True])
    )
    method_summary.to_csv(out / "summary_by_method.csv", index=False,
                          encoding="utf-8")

    group_cols = ["source_label", "n", "display_name"]
    if mode == "external" and "K" in df.columns:
        group_cols.insert(2, "K")
    by_source = (
        df.groupby(group_cols, dropna=False)
        .agg(
            runs=("sample_id", "size"),
            mean_objective=("valid_objective", "mean"),
            max_objective=("valid_objective", "max"),
            median_wall_time=("wall_time", "median"),
            p95_wall_time=("wall_time", lambda s: s.quantile(0.95)),
            timeout_rate=("timed_out_bool", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )
    by_source.to_csv(out / "summary_by_source.csv", index=False,
                     encoding="utf-8")

    paired = _paired_difference_table(df, mode)
    if not paired.empty:
        paired.to_csv(out / "paired_differences.csv", index=False,
                      encoding="utf-8")
    print(f"  汇总表: {out / 'summary_by_method.csv'}")
    print(f"  分组表: {out / 'summary_by_source.csv'}")
    if not paired.empty:
        print(f"  配对差异表: {out / 'paired_differences.csv'}")


def _paired_difference_table(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """生成配对差异表。embedded 对 ClassicalClique，external 对同 K 的 MS_Degree。"""
    rows = []
    valid = df.copy()
    valid["objective"] = pd.to_numeric(valid["objective"], errors="coerce")
    valid = valid.dropna(subset=["objective"])
    if valid.empty:
        return pd.DataFrame()

    if mode == "embedded":
        baseline_name = "ClassicalClique"
        valid["_pair_key"] = list(zip(valid["sample_id"], valid["run_id"]))
        base = valid[valid["display_name"].eq(baseline_name)]
        base_lookup = dict(zip(base["_pair_key"], base["objective"]))
        for name in sorted(valid["display_name"].unique()):
            if name == baseline_name:
                continue
            sub = valid[valid["display_name"].eq(name)]
            diffs = [
                float(r["objective"] - base_lookup[key])
                for _, r in sub.iterrows()
                for key in [r["_pair_key"]]
                if key in base_lookup
            ]
            if diffs:
                rows.append(_diff_stats(name, baseline_name, diffs))
    else:
        valid["_pair_key"] = list(zip(valid["sample_id"], valid["run_id"], valid.get("K", 0)))
        base = valid[valid["family"].eq("MS_Degree")]
        base_lookup = dict(zip(base["_pair_key"], base["objective"]))
        for name in sorted(valid["display_name"].unique()):
            if name == "MS_Degree":
                continue
            sub = valid[valid["display_name"].eq(name)]
            diffs = [
                float(r["objective"] - base_lookup[key])
                for _, r in sub.iterrows()
                for key in [r["_pair_key"]]
                if key in base_lookup
            ]
            if diffs:
                rows.append(_diff_stats(name, "MS_Degree(same K)", diffs))
    return pd.DataFrame(rows)


def _diff_stats(name: str, baseline: str, diffs: list[float]) -> dict:
    arr = np.asarray(diffs, dtype=float)
    return {
        "method": name,
        "baseline": baseline,
        "n_pairs": len(arr),
        "mean_diff": float(arr.mean()),
        "median_diff": float(np.median(arr)),
        "wins": int((arr > 0).sum()),
        "ties": int((arr == 0).sum()),
        "losses": int((arr < 0).sum()),
        "win_rate": float((arr > 0).mean()),
    }


def _plot_embedded_boxplots(df: pd.DataFrame, out: Path):
    """Part A 箱线图：每个 display_name 独立一箱。"""
    display_names = sorted(df["display_name"].unique(),
                           key=lambda d: df[df["display_name"] == d]["objective"].mean(),
                           reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, (col, ylabel) in enumerate([("objective", "Clique Size |S|"),
                                             ("wall_time", "Runtime (s)")]):
        ax = axes[ax_idx]
        data_list = [df[df["display_name"] == d][col].dropna().tolist()
                     for d in display_names]
        short_names = [_short_label(d) for d in display_names]

        bp = ax.boxplot(data_list, tick_labels=short_names,
                        patch_artist=True, widths=0.5)
        for patch, dname in zip(bp["boxes"], display_names):
            color = COLORS.get(dname, "#999999")
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{'Objective' if ax_idx == 0 else 'Runtime'} — embedded")
        ax.grid(True, alpha=0.3, axis="y")
        plt.setp(ax.xaxis.get_ticklabels(), rotation=20, ha="right", fontsize=8)

    fig.suptitle("exp6 Part A: Embedded CTQW — Large Scale",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "embedded_boxplot.png", dpi=150, bbox_inches="tight")
    print(f"  箱线图: {out / 'embedded_boxplot.png'}")
    plt.close(fig)


def _plot_external_boxplots(df: pd.DataFrame, out: Path):
    """Part B 箱线图：按 K 分组，MS_CTQW 的两种方法各自独立显示。"""
    Ks = sorted([k for k in df["K"].unique() if k > 0])
    if not Ks:
        print("  跳过 external 箱线图：没有有效的 Multi-Start K 结果")
        return

    fig, axes = plt.subplots(1, len(Ks), figsize=(5 * len(Ks), 5),
                             squeeze=False)
    for ki, K_val in enumerate(Ks):
        ax = axes[0, ki]
        sub = df[df["K"] == K_val]

        # MS_Random, MS_Degree（不涉 CTQW，按 family 聚合），
        # MS_CTQW 按 display_name 拆分
        groups = []
        # 先加 MS_Random 和 MS_Degree（经典，不区分方法）
        if not sub[sub["family"] == "MS_Random"].empty:
            groups.append(("MS_Random", "Random"))
        if not sub[sub["family"] == "MS_Degree"].empty:
            groups.append(("MS_Degree", "Degree"))

        # MS_CTQW / MS_HybridSeed 按 display_name 拆分
        ms_ctqw_names = sorted([
            d for d in sub["display_name"].unique()
            if d.startswith("MS_CTQW") or d.startswith("MS_HybridSeed")
        ])
        for dname in ms_ctqw_names:
            # 从 display_name 提取简短标签
            label = dname.replace("MS_CTQW_", "CTQW_")
            label = label.replace("MS_HybridSeed_", "Hybrid_")
            groups.append((dname, label))

        data_list = []
        labels = []
        colors_list = []
        for key, label in groups:
            if key in ("MS_Random", "MS_Degree"):
                gsub = sub[sub["family"] == key]
            else:
                gsub = sub[sub["display_name"] == key]
            if not gsub.empty:
                data_list.append(gsub["objective"].dropna().tolist())
                labels.append(label)
                colors_list.append(COLORS.get(key, "#999999"))

        if data_list:
            bp = ax.boxplot(data_list, tick_labels=labels,
                            patch_artist=True, widths=0.5)
            for patch, c in zip(bp["boxes"], colors_list):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)

        # 叠加 ClassicalClique 基线
        cc = df[df["family"] == "ClassicalClique"]
        if not cc.empty:
            cc_mean = cc["objective"].mean()
            ax.axhline(cc_mean, color="red", linestyle="--", linewidth=1,
                       label=f"CliqueGreedy ({cc_mean:.1f})")
            ax.legend(fontsize=7)

        ax.set_title(f"K = {K_val}")
        ax.set_ylabel("Clique Size |S|")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("exp6 Part B: Multi-Start — Large Scale",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "external_boxplot.png", dpi=150, bbox_inches="tight")
    print(f"  箱线图: {out / 'external_boxplot.png'}")
    plt.close(fig)


def _plot_external_heatmap(df: pd.DataFrame, out: Path):
    """Part B 热力图：K × display_name 的团大小均值。

    MS_CTQW 按演化方法拆分多行，MS_Random / MS_Degree 各占一行。
    MS_HybridSeed 按 beta 和演化方法拆分多行。
    """
    ms_data = df[df["family"].isin([
        "MS_Random", "MS_Degree", "MS_CTQW", "MS_HybridSeed"
    ])]
    if ms_data.empty:
        print("  跳过 external 热力图：没有有效的 Multi-Start 结果")
        return
    # 用 display_name 替代 family，使 MS_CTQW 的两种方法各自成行
    agg = ms_data.groupby(["display_name", "K"])["objective"].mean().reset_index()
    pivot = agg.pivot(index="display_name", columns="K", values="objective")

    if pivot.empty:
        return

    # 重新排列行：MS_Random → MS_Degree → MS_CTQW → MS_HybridSeed
    row_order = ["MS_Random", "MS_Degree"]
    for dname in sorted(pivot.index):
        if dname.startswith("MS_CTQW"):
            row_order.append(dname)
    for dname in sorted(pivot.index):
        if dname.startswith("MS_HybridSeed"):
            row_order.append(dname)
    pivot = pivot.reindex([r for r in row_order if r in pivot.index])

    fig, ax = plt.subplots(figsize=(9, len(pivot) * 1.1 + 1.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                    origin="upper", interpolation="nearest")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if val > pivot.values.max() * 0.7 else "black")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"K={int(k)}" for k in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_short_label(r) for r in pivot.index])
    ax.set_title("Multi-Start: Mean Clique Size by K & Method",
                 fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.85, label="Clique Size |S|")
    plt.tight_layout()
    fig.savefig(out / "external_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"  热力图: {out / 'external_heatmap.png'}")
    plt.close(fig)




# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="实验六：大规模图 CTQW 近似方法对比")

    parser.add_argument("--mode", required=True,
                        choices=["embedded", "external"],
                        help="embedded: 嵌入式量子贪心; external: Multi-Start 外置")
    parser.add_argument("--smoke", action="store_true",
                        help="烟雾测试：1 实例 + 1 次重复")

    parser.add_argument("--data-source", default="artificial",
                        choices=["artificial", "misleading", "dimacs"],
                        help="数据来源（人工 n=300,500 / misleading / DIMACS 5 个指定数据集）")

    parser.add_argument("--methods", nargs="+",
                        default=["krylov_m30", "cheb_d50"],
                        help="演化方法 key 列表（默认: krylov_m30 cheb_d50）")

    parser.add_argument("--K-values", nargs="+", type=int,
                        default=[5, 10, 20, 30],
                        help="Multi-Start 起点数 K（仅 external 模式）")
    parser.add_argument("--hybrid-betas", nargs="+", type=float,
                        default=[],
                        help="启用 MS_HybridSeed 的 beta 列表，如 0.25 0.5 0.75")

    parser.add_argument("--timeout", type=float, default=300,
                        help="单次 solve 超时门限（秒），默认 300")
    parser.add_argument("--repeat", type=int, default=None,
                        help="每实例重复次数（默认: artificial=2, dimacs=3）")

    parser.add_argument("--no-plot", action="store_true", help="跳过绘图")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--limit-instances", type=int, default=None,
                        help="最多运行多少个实例；用于分批跑大规模实验")
    parser.add_argument("--dimacs-labels", nargs="+", default=None,
                        help="仅运行指定 DIMACS label，如 C250-9 p-hat300-3")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行 solve 数量；建议大图从 2 开始")
    parser.add_argument("--resume", action="store_true",
                        help="从 full_results.csv 或 full_results.partial.csv 跳过已完成任务")
    parser.add_argument("--retry-timeouts", action="store_true",
                        help="配合 --resume 使用：丢弃旧超时记录并重跑这些任务")

    args = parser.parse_args()

    # 确定超时门限
    timeout_sec = min(args.timeout, 30) if args.smoke else args.timeout

    # 确定重复次数
    if args.repeat is not None:
        repeat = args.repeat
    elif args.data_source == "dimacs":
        repeat = REPEAT_DIMACS
    else:
        repeat = REPEAT_ARTIFICIAL

    # 发现实例
    instances = discover_instances(args.data_source, smoke=args.smoke,
                                   limit=args.limit_instances,
                                   dimacs_labels=(set(args.dimacs_labels)
                                                  if args.dimacs_labels else None))
    if not instances:
        print("未找到任何测试实例，退出。")
        return

    # 人工数据的实际 n 范围
    ns = sorted(set(inst.num_nodes for inst, _ in instances))

    print(f"\n实验六：大规模图 CTQW 近似方法对比")
    print(f"  模式:       {args.mode}")
    print(f"  数据来源:   {args.data_source}")
    print(f"  节点范围:   n ∈ {ns}")
    print(f"  实例数:     {len(instances)}")
    print(f"  方法:       {args.methods}")
    print(f"  重复:       {repeat} 次/实例")
    print(f"  超时:       {timeout_sec}s")
    print(f"  workers:    {args.workers}")
    if args.mode == "external":
        print(f"  K 值:       {args.K_values}")
        print(f"  beta 值:    {args.hybrid_betas}")

    # 输出目录先创建，用作长任务 checkpoint。
    if args.data_source == "dimacs":
        data_tag = "dimacs"
    elif args.data_source == "misleading":
        data_tag = f"misleading_n{min(ns)}-{max(ns)}"
    else:
        data_tag = f"n{min(ns)}-{max(ns)}"
    if args.limit_instances is not None:
        data_tag = f"{data_tag}_first{args.limit_instances}"
    if args.dimacs_labels:
        safe_labels = "-".join(label.replace("/", "_") for label in args.dimacs_labels)
        data_tag = f"{data_tag}_{safe_labels}"
    tag = f"{data_tag}{'_smoke' if args.smoke else ''}"

    if args.output:
        out = Path(args.output)
    else:
        out = RESULTS_DIR / f"exp6_{args.mode}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "full_results.partial.csv"

    # 运行实验
    df = run_experiment(
        instances, args.mode, args.methods, args.K_values, args.hybrid_betas,
        repeat, timeout_sec, args.smoke,
        checkpoint_path=checkpoint_path,
        workers=max(1, args.workers),
        resume=args.resume,
        retry_timeouts=args.retry_timeouts)

    if df.empty:
        print("未收集到数据，退出。")
        return

    df.to_csv(out / "full_results.csv", index=False, encoding="utf-8")
    print(f"\n完整结果已保存: {out / 'full_results.csv'}")

    if not args.no_plot:
        analyze_and_plot(df, args.mode, tag, output_dir=out)

    print(f"\n实验六完成。")


if __name__ == "__main__":
    main()
