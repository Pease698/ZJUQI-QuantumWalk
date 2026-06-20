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

  # 自定义并行 worker 数
  python3 experiments/exp6_large_scale_approx.py --mode embedded --workers 3
"""

import argparse
import multiprocessing as mp
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    MultiStartCTQWGreedy, MultiStartRandomGreedy, MultiStartDegreeGreedy)
from src.metrics import mean_std
from src.config import get_data_dirs, DATA_DIR, ensure_results_dir
from src.timeout import run_with_timeout

# ============================================================
# 配置常量
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp6_large_scale"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 并行控制
MAX_WORKERS = 5                # 默认并行 worker 数
SERIAL_N_THRESHOLD = 2000      # n 超过此值自动串行处理该实例，避免 OOM

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
    # "gen200-p0-9-44",   # n=200, |E|≈1.8K
    # "C250-9",            # n=250, |E|≈2.8K
    # "p-hat300-3",        # n=300, |E|≈3.3K
    # "C1000-9",           # n=1000, |E|≈45K
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
}


# ============================================================
# 实例发现
# ============================================================

def _extract_n(dirname: str) -> int:
    m = re.search(r'_n(\d+)_', dirname)
    return int(m.group(1)) if m else 999


def discover_instances(
    data_source: str = "artificial",
    smoke: bool = False,
) -> list[tuple[GraphInstance, str]]:
    """发现大规模测试实例。

    artificial: 仅加载 n ∈ {300, 500} 的参数组（由 ARTIFICIAL_N_VALUES 控制）。
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

    elif data_source == "dimacs":
        ext_dir = DATA_DIR / "external" / "maximum_clique"
        if not ext_dir.is_dir():
            print(f"错误: 外部数据目录不存在: {ext_dir}")
            return instances

        for fpath in sorted(ext_dir.glob("*.json")):
            inst = load_instance(fpath)
            stem = fpath.stem
            label = stem[7:] if stem.startswith("ext_mc_") else stem
            if label not in DIMACS_WHITELIST:
                continue
            if smoke:
                instances = [(inst, label)]
                break
            instances.append((inst, label))

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
    seed: int = 0,
) -> dict[str, BaseAlgorithm]:
    """构造 Part B 的算法集合：Multi-Start 系列。"""
    algorithms: dict[str, BaseAlgorithm] = {}

    # 单起点基线
    algorithms["ClassicalClique"] = ClassicalGreedy(
        CliqueCandidateSet(), ClassicalCliqueScorer(), name="ClassicalClique")

    # Multi-Start × K（不涉及 CTQW）
    for K in K_values:
        algorithms[f"MS_Random_K{K}"] = MultiStartRandomGreedy(
            K=K, seed=seed, name=f"MS_Random(K={K})")
        algorithms[f"MS_Degree_K{K}"] = MultiStartDegreeGreedy(
            K=K, seed=seed, name=f"MS_Degree(K={K})")

    # MultiStartCTQW × K × 演化方法
    for K in K_values:
        for key in method_keys:
            preset = EVOLUTION_PRESETS.get(key)
            if preset is None:
                continue
            if preset["method"] == "exact" and instance.num_nodes > 200:
                continue

            algo = MultiStartCTQWGreedy(
                K=K, t=FIXED_T, seed=seed,
                evolution_method=preset["method"],
                krylov_dim=preset.get("krylov_dim"),
                cheb_degree=preset.get("cheb_degree"),
                name=f"MS_CTQW(K={K},{key})",
            )
            algorithms[f"MS_CTQW_K{K}_{key}"] = algo

    return algorithms


# ============================================================
# 单任务执行（模块级函数，供 ProcessPoolExecutor pickle）
# ============================================================

def _execute_one_task(
    algo_template: BaseAlgorithm,
    algo_name: str,
    instance: GraphInstance,
    seed: int,
    timeout_sec: float,
    mode: str,
    source_label: str,
    run_id: int,
) -> dict:
    """单个 (算法, 实例, seed) 组合的执行单元。

    必须是模块级顶层函数（非闭包），否则 spawn 上下文无法 pickle。
    内部调用 run_with_timeout 隔离子进程，确保超时安全。
    """
    algo = _rebuild_with_seed(algo_template, seed, mode)

    t0 = time.perf_counter()
    result = run_with_timeout(algo, instance, timeout_sec)
    wall = time.perf_counter() - t0

    row = result.to_dict()
    row["source_label"] = source_label
    row["algo_key"] = algo_name
    row["run_id"] = run_id
    row["seed"] = seed
    row["wall_time"] = wall
    row["n"] = instance.num_nodes
    row["mode"] = mode
    row["family"] = _classify_family(algo_name)
    row["method_tag"] = _extract_ev_tag(algo_name)
    if mode == "external":
        row["K"] = _extract_K(algo_name)
    return row


# ============================================================
# 实验运行：串行 / 并行 双模式
# ============================================================

def run_experiment(
    instances: list[tuple[GraphInstance, str]],
    mode: str,
    method_keys: list[str],
    K_values: list[int] | None,
    repeat: int,
    timeout_sec: float,
    smoke: bool,
    workers: int = MAX_WORKERS,
    serial_threshold: int = SERIAL_N_THRESHOLD,
) -> pd.DataFrame:
    """运行实验并返回汇总 DataFrame。

    烟雾测试或 workers=0 时使用串行模式；否则使用 ProcessPoolExecutor 并行。
    """
    if smoke or workers <= 0:
        return _run_serial(
            instances, mode, method_keys, K_values,
            repeat, timeout_sec, smoke)
    return _run_parallel(
        instances, mode, method_keys, K_values,
        repeat, timeout_sec, workers, serial_threshold)


def _print_header(mode: str, total_instances: int, n_methods: int,
                  repeat: int, timeout_sec: float, workers: int,
                  K_values: list[int] | None, is_parallel: bool):
    """打印实验配置头部信息。"""
    print(f"\n{'=' * 60}")
    print(f"实验六 Part {'A' if mode == 'embedded' else 'B'}: "
          f"{'嵌入式' if mode == 'embedded' else '外置式'}方案 "
          f"({'并行 × ' + str(workers) if is_parallel else '串行'})")
    print(f"  实例数:   {total_instances}")
    print(f"  方法数:   {n_methods}")
    print(f"  重复:     {repeat}")
    print(f"  超时:     {timeout_sec}s")
    if mode == "external":
        print(f"  K 值:     {K_values}")
    print(f"{'=' * 60}")


def _run_serial(
    instances: list[tuple[GraphInstance, str]],
    mode: str,
    method_keys: list[str],
    K_values: list[int] | None,
    repeat: int,
    timeout_sec: float,
    smoke: bool,
) -> pd.DataFrame:
    """串行执行：保持原有顺序，每实例逐个算法运行。"""
    all_rows: list[dict] = []
    total_instances = len(instances)
    t_start = time.perf_counter()

    _print_header(mode, total_instances, len(method_keys), repeat,
                  timeout_sec, 0, K_values, is_parallel=False)

    for idx, (inst, label) in enumerate(instances):
        if mode == "embedded":
            algos = build_embedded_algorithms(inst, method_keys, seed=0)
        else:
            algos = build_external_algorithms(inst, method_keys,
                                              K_values or [5, 10], seed=0)

        for run_id in range(1 if smoke else repeat):
            seed = run_id

            for algo_name, algo_template in algos.items():
                row = _execute_one_task(
                    algo_template, algo_name, inst, seed,
                    timeout_sec, mode, label, run_id)
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

    df = pd.DataFrame(all_rows)
    elapsed_total = time.perf_counter() - t_start
    print(f"\n全部完成，总耗时 {elapsed_total:.0f}s，共 {len(df)} 条记录")
    _print_timeout_summary(df)
    return df


def _run_parallel(
    instances: list[tuple[GraphInstance, str]],
    mode: str,
    method_keys: list[str],
    K_values: list[int] | None,
    repeat: int,
    timeout_sec: float,
    workers: int,
    serial_threshold: int,
) -> pd.DataFrame:
    """并行执行：使用 ProcessPoolExecutor 并发运行任务。

    大图实例（n > serial_threshold）自动降为串行处理，避免多进程
    同时加载大型邻接矩阵导致 OOM。
    """
    all_rows: list[dict] = []
    total_instances = len(instances)
    t_start = time.perf_counter()

    _print_header(mode, total_instances, len(method_keys), repeat,
                  timeout_sec, workers, K_values, is_parallel=True)

    # 分类实例：大图串行 / 小图并行
    serial_tasks: list[dict] = []
    parallel_tasks: list[dict] = []

    for inst, label in instances:
        n = inst.num_nodes

        if mode == "embedded":
            algos = build_embedded_algorithms(inst, method_keys, seed=0)
        else:
            algos = build_external_algorithms(inst, method_keys,
                                              K_values or [5, 10], seed=0)

        for run_id in range(repeat):
            seed = run_id
            for algo_name, algo_template in algos.items():
                task = dict(
                    algo_template=algo_template, algo_name=algo_name,
                    instance=inst, seed=seed, timeout_sec=timeout_sec,
                    mode=mode, source_label=label, run_id=run_id,
                )
                if n > serial_threshold:
                    serial_tasks.append(task)
                else:
                    parallel_tasks.append(task)

    if serial_tasks:
        big_ns = sorted(set(t["instance"].num_nodes for t in serial_tasks))
        print(f"  大图串行:   {len(serial_tasks)} 个任务 (n ∈ {big_ns} > {serial_threshold})")
    if parallel_tasks:
        print(f"  并行任务:   {len(parallel_tasks)} 个 (workers={workers})")

    n_total = len(serial_tasks) + len(parallel_tasks)

    # ---- 并行阶段 ----
    if parallel_tasks:
        ctx = mp.get_context("spawn")
        # 实际 worker 数不超过任务数
        actual_workers = min(workers, len(parallel_tasks))
        with ProcessPoolExecutor(max_workers=actual_workers,
                                 mp_context=ctx) as executor:
            futures = {}
            for task in parallel_tasks:
                future = executor.submit(
                    _execute_one_task,
                    task["algo_template"], task["algo_name"],
                    task["instance"], task["seed"], task["timeout_sec"],
                    task["mode"], task["source_label"], task["run_id"],
                )
                futures[future] = task

            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:
                    task = futures[future]
                    row = _make_error_row(
                        task["algo_name"], task["instance"],
                        task["seed"], task["timeout_sec"],
                        task["mode"], task["source_label"],
                        task["run_id"], exc)
                all_rows.append(row)

                # 周期进度
                n_done = len(all_rows)
                if n_done % max(n_total // 10, 1) == 0 or n_done == len(parallel_tasks) + len(serial_tasks):
                    elapsed = time.perf_counter() - t_start
                    n_timeout = sum(1 for r in all_rows if r.get("timed_out"))
                    print(f"  [{n_done}/{n_total}] 已完成, "
                          f"耗时 {elapsed:.0f}s"
                          + (f", 超时 {n_timeout} 条" if n_timeout > 0 else ""))

    # ---- 串行阶段（大图） ----
    for task in serial_tasks:
        try:
            row = _execute_one_task(
                task["algo_template"], task["algo_name"],
                task["instance"], task["seed"], task["timeout_sec"],
                task["mode"], task["source_label"], task["run_id"],
            )
        except Exception as exc:
            row = _make_error_row(
                task["algo_name"], task["instance"],
                task["seed"], task["timeout_sec"],
                task["mode"], task["source_label"],
                task["run_id"], exc)
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    elapsed_total = time.perf_counter() - t_start
    print(f"\n全部完成，总耗时 {elapsed_total:.0f}s，共 {len(df)} 条记录")
    _print_timeout_summary(df)
    return df


def _print_timeout_summary(df: pd.DataFrame):
    """打印超时统计。"""
    if "timed_out" in df.columns:
        n_to = int(df["timed_out"].sum())
        if n_to > 0:
            print(f"  其中超时: {n_to} / {len(df)} ({n_to / len(df) * 100:.1f}%)")


def _make_error_row(algo_name: str, instance: GraphInstance, seed: int,
                    timeout_sec: float, mode: str, source_label: str,
                    run_id: int, exc: Exception) -> dict:
    """构造任务级异常（非超时，是进程崩溃）的占位行。"""
    return {
        "algorithm": f"{algo_name}[CRASH]",
        "sample_id": instance.sample_id,
        "task_type": instance.task_type,
        "objective": float("nan"),
        "runtime": 0.0,
        "iterations": 0,
        "solution_size": 0,
        "answer_size": len(instance.answer_set),
        "recall": 0.0,
        "timed_out": False,
        "solution": [],
        "source_label": source_label,
        "algo_key": algo_name,
        "run_id": run_id,
        "seed": seed,
        "wall_time": 0.0,
        "n": instance.num_nodes,
        "mode": mode,
        "family": _classify_family(algo_name),
        "method_tag": _extract_ev_tag(algo_name),
        "K": _extract_K(algo_name) if mode == "external" else 0,
        "error": f"{type(exc).__name__}: {exc}",
    }


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

    elif isinstance(template, MultiStartRandomGreedy):
        return MultiStartRandomGreedy(K=template.K, seed=seed, name=name)

    elif isinstance(template, MultiStartDegreeGreedy):
        return MultiStartDegreeGreedy(K=template.K, seed=seed, name=name)

    return template


# ============================================================
# 标签解析
# ============================================================

def _classify_family(algo_name: str) -> str:
    for fam in ["ClassicalDegree", "ClassicalClique", "SimulatedAnnealing",
                 "QuantumGreedy", "MS_Random", "MS_Degree", "MS_CTQW"]:
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
    经典基线保持不变。仅当 method_tag 是有效字符串时才附加到显示名。
    """
    fam = row.get("family", "Unknown")
    tag = row.get("method_tag", None)
    if isinstance(tag, str) and tag != "N/A":
        return f"{fam}_{tag}"
    return fam


def _short_label(display: str) -> str:
    """压缩显示名以便在图表中使用。"""
    return display \
        .replace("QuantumGreedy", "QG") \
        .replace("ClassicalDegree", "DegGreedy") \
        .replace("ClassicalClique", "CliqueGreedy") \
        .replace("SimulatedAnnealing", "SimAnneal") \
        .replace("MS_", "")


def analyze_and_plot(df: pd.DataFrame, mode: str, tag: str):
    """分析实验结果并生成图表和 CSV。"""
    if df.empty:
        print("DataFrame 为空，跳过分析。")
        return

    out = RESULTS_DIR / f"exp6_{mode}_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    # 添加 display_name 列（原始 df 和有效子集都需要）
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

    # ---- 箱线图 ----
    if mode == "embedded":
        _plot_embedded_boxplots(df_valid, out)
    else:
        _plot_external_boxplots(df_valid, out)
        _plot_external_heatmap(df_valid, out)

    print(f"\n结果目录: {out}")


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

        # MS_CTQW 按 display_name 拆分
        ms_ctqw_names = sorted([
            d for d in sub["display_name"].unique()
            if d.startswith("MS_CTQW")
        ])
        for dname in ms_ctqw_names:
            # 从 display_name 提取简短标签
            label = dname.replace("MS_CTQW_", "CTQW_")
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
    """
    ms_data = df[df["family"].isin(["MS_Random", "MS_Degree", "MS_CTQW"])]
    # 用 display_name 替代 family，使 MS_CTQW 的两种方法各自成行
    agg = ms_data.groupby(["display_name", "K"])["objective"].mean().reset_index()
    pivot = agg.pivot(index="display_name", columns="K", values="objective")

    if pivot.empty:
        return

    # 重新排列行：MS_Random → MS_Degree → MS_CTQW 的两个变体
    row_order = ["MS_Random", "MS_Degree"]
    for dname in sorted(pivot.index):
        if dname.startswith("MS_CTQW"):
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
                        choices=["artificial", "dimacs"],
                        help="数据来源（人工 n=300,500 / DIMACS 5 个指定数据集）")

    parser.add_argument("--methods", nargs="+",
                        default=["krylov_m30", "cheb_d50"],
                        help="演化方法 key 列表（默认: krylov_m30 cheb_d50）")

    parser.add_argument("--K-values", nargs="+", type=int,
                        default=[5, 10, 20, 30],
                        help="Multi-Start 起点数 K（仅 external 模式）")

    parser.add_argument("--timeout", type=float, default=300,
                        help="单次 solve 超时门限（秒），默认 300")
    parser.add_argument("--repeat", type=int, default=None,
                        help="每实例重复次数（默认: artificial=2, dimacs=3）")

    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"并行 worker 数（默认 {MAX_WORKERS}，0=回退串行）")
    parser.add_argument("--serial-threshold", type=int,
                        default=SERIAL_N_THRESHOLD,
                        help=f"n 超过此值的大图实例改为串行处理（默认 {SERIAL_N_THRESHOLD}）")

    parser.add_argument("--no-plot", action="store_true", help="跳过绘图")
    parser.add_argument("--output", type=str, default=None, help="输出目录")

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
    instances = discover_instances(args.data_source, smoke=args.smoke)
    if not instances:
        print("未找到任何测试实例，退出。")
        return

    # 人工数据的实际 n 范围
    ns = sorted(set(inst.num_nodes for inst, _ in instances))

    print(f"\n实验六：大规模图 CTQW 近似方法对比")
    print(f"  模式:         {args.mode}")
    print(f"  数据来源:     {args.data_source}")
    print(f"  节点范围:     n ∈ {ns}")
    print(f"  实例数:       {len(instances)}")
    print(f"  方法:         {args.methods}")
    print(f"  重复:         {repeat} 次/实例")
    print(f"  超时:         {timeout_sec}s")
    print(f"  并行 worker:  {args.workers if args.workers > 0 else '串行'}"
          + (f" (大图阈值 n>{args.serial_threshold})" if args.workers > 0 else ""))
    if args.mode == "external":
        print(f"  K 值:         {args.K_values}")

    # 运行实验
    df = run_experiment(
        instances, args.mode, args.methods, args.K_values,
        repeat, timeout_sec, args.smoke,
        workers=args.workers,
        serial_threshold=args.serial_threshold)

    if df.empty:
        print("未收集到数据，退出。")
        return

    # 输出
    data_tag = "dimacs" if args.data_source == "dimacs" else f"n{min(ns)}-{max(ns)}"
    tag = f"{data_tag}{'_smoke' if args.smoke else ''}"

    if args.output:
        out = Path(args.output)
    else:
        out = RESULTS_DIR / f"exp6_{args.mode}_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "full_results.csv", index=False, encoding="utf-8")
    print(f"\n完整结果已保存: {out / 'full_results.csv'}")

    if not args.no_plot:
        analyze_and_plot(df, args.mode, tag)

    print(f"\n实验六完成。")


if __name__ == "__main__":
    main()
