"""CTQW 真实实现冒烟测试（Phase 3.1 验证脚本）。

目的：
  验证 src/scoring.py:QuantumScorer 和 src/algorithms/quantum_greedy.py 中
  新接入的 scipy.linalg.expm 真实 CTQW 计算是否在数值上正确。

检查项（5 项）：
  1. 哈密顿量 Hermitian: H = H^H  (^H 表示共轭转置)
  2. 演化算子幺正: U^H U = I  (=> 状态范数守恒)
  3. 概率守恒: sum(P_v) = 1
  4. 概率非负: P_v >= 0
  5. 理论假设一: 在 planted clique 图上 Ratio > 1

使用方法:
  cd ZJUQI-QuantumWalk
  python test_ctqw_smoke.py
"""

import io
import os
import sys

import numpy as np

# Windows 控制台 GBK 默认编码无法显示部分 Unicode 数学符号，
# 这里强制 stdout/stderr 切到 UTF-8 以便正常输出。
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

# 把项目根目录加入 import path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.graph_utils import load_instance
from src.hamiltonian import construct_hamiltonian
from src.initial_state import build_initial_state
from src.scoring import QuantumScorer
from src.algorithms.quantum_greedy import QuantumGuidedGreedy
from src.candidate_set import CliqueCandidateSet
from src.metrics import compute_ratio

# 数值容差
TOL = 1e-10

# 测试用的小图：30 节点、p=0.1、5-团
TEST_INSTANCE = os.path.join(
    PROJECT_ROOT, "datasets", "data", "artificial", "maximum_clique",
    "mc_n30_p01_k5", "mc_n30_p01_k5_000.json"
)


def _section(title: str):
    """打印分隔标题。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _check(name: str, passed: bool, detail: str = "") -> bool:
    """打印一项检查结果，返回布尔值便于累计。"""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {name}")
    if detail:
        print(f"           {detail}")
    return passed


# ============================================================
# 检查 1：哈密顿量 Hermitian
# ============================================================

def check_hamiltonian_hermitian(instance) -> bool:
    """H = A + λ Σ|u⟩⟨u| 必须是 Hermitian 矩阵。

    由于 A 是实对称的，且对角扰动项也是实对称的，所以 H 实际上是实对称。
    """
    _section("检查 1：哈密顿量 Hermitian (H = H†)")
    adj = instance.adjacency

    results = []
    # λ=0 退化情形
    H0 = construct_hamiltonian(adj, S=set(), lam=0.0)
    err0 = np.max(np.abs(H0 - H0.conj().T))
    results.append(_check(
        "λ=0, S=∅:  H == H^†",
        err0 < TOL,
        f"max|H - H^†| = {err0:.2e}"
    ))

    # 有种子扰动
    S_seed = {0, 5, 10}
    H1 = construct_hamiltonian(adj, S=S_seed, lam=1.5)
    err1 = np.max(np.abs(H1 - H1.conj().T))
    results.append(_check(
        f"λ=1.5, S={S_seed}:  H == H^†",
        err1 < TOL,
        f"max|H - H^†| = {err1:.2e}"
    ))

    return all(results)


# ============================================================
# 检查 2：演化算子幺正性
# ============================================================

def check_unitary(instance) -> bool:
    """U(t) = e^{-iHt} 必须是幺正矩阵：U^† U = I。"""
    _section("检查 2：演化算子幺正性 (U^† U = I)")
    from scipy.linalg import expm

    adj = instance.adjacency
    n = instance.num_nodes

    results = []
    for t, lam, S in [(1.0, 0.0, set()),
                      (2.0, 0.5, {0, 5}),
                      (5.0, 1.0, {3, 7, 12})]:
        H = construct_hamiltonian(adj, S=S, lam=lam)
        U = expm(-1j * H * t)
        UU = U.conj().T @ U
        err = np.max(np.abs(UU - np.eye(n)))
        results.append(_check(
            f"t={t}, λ={lam}, |S|={len(S)}:  ‖U^†U − I‖ < {TOL:.0e}",
            err < 1e-8,  # expm 数值误差比 TOL 略大
            f"max|U^†U − I| = {err:.2e}"
        ))

    return all(results)


# ============================================================
# 检查 3：概率守恒 Σ P_v = 1
# ============================================================

def check_probability_sum(instance) -> bool:
    """Σ |ψ_v(t)|² = 1（理论 §4 推论：幺正演化保概率守恒）。"""
    _section("检查 3：概率守恒 (Σ P_v = 1)")
    n = instance.num_nodes
    candidates = set(range(n))

    results = []
    for t, lam, init in [(1.0, 0.0, "max_degree"),
                         (2.0, 0.5, "uniform"),
                         (5.0, 1.0, "max_degree")]:
        # S=∅ 时按 init_method 构造初态
        qs = QuantumScorer(t=t, lam=lam, init_method=init)
        probs = qs.score_all(candidates, set(), instance)
        total = sum(probs.values())
        results.append(_check(
            f"t={t}, λ={lam}, init={init}, S=∅:  Σ P_v ≈ 1",
            abs(total - 1.0) < 1e-8,
            f"Σ P_v = {total:.10f}, |Σ-1| = {abs(total-1):.2e}"
        ))

    # S 非空的情况
    qs = QuantumScorer(t=1.0, lam=0.5, init_method="max_degree")
    probs = qs.score_all(candidates, {0, 5, 10}, instance)
    total = sum(probs.values())
    results.append(_check(
        "t=1.0, λ=0.5, S={0,5,10}:  Σ P_v ≈ 1",
        abs(total - 1.0) < 1e-8,
        f"Σ P_v = {total:.10f}"
    ))

    return all(results)


# ============================================================
# 检查 4：概率非负
# ============================================================

def check_probability_nonneg(instance) -> bool:
    """P_v = |ψ_v|² ≥ 0 必须恒成立（模平方一定非负）。"""
    _section("检查 4：概率非负 (P_v ≥ 0)")
    n = instance.num_nodes
    candidates = set(range(n))

    qs = QuantumScorer(t=2.0, lam=0.5, init_method="max_degree")
    probs = qs.score_all(candidates, {0, 5}, instance)
    values = np.array(list(probs.values()))

    min_p = float(values.min())
    n_neg = int((values < -TOL).sum())  # 容忍微小负浮点误差
    return _check(
        "所有 P_v ≥ 0",
        n_neg == 0,
        f"min P_v = {min_p:.6e}, 严格负数节点数 = {n_neg}"
    )


# ============================================================
# 检查 5：理论假设一 —— planted clique 上 Ratio > 1
# ============================================================

def check_ratio_on_clique(instance) -> bool:
    """在含 planted clique 的图上：

      Ratio = Mean(P_v, v∈S_target) / Mean(P_v, v∉S_target)

    若 CTQW 真的反映全局拓扑（理论 §3.1 假设一），Ratio 应明显 > 1。

    注意：该检查不是数值正确性必要条件——即使 CTQW 实现正确，
    Ratio 仍可能 ≤ 1（取决于 t、λ、init 选择）。这里只用作"是否合理"参考。
    """
    _section("检查 5：理论假设一 (planted clique 上 Ratio > 1)")
    n = instance.num_nodes
    answer = instance.answer_set
    print(f"  实例: {instance.sample_id}")
    print(f"  节点数 n={n}, 植入团大小 k={len(answer)}")
    print(f"  植入节点: {sorted(answer)}")

    # 扫几组参数看哪个能给出 Ratio>1
    candidates = set(range(n))
    results = []
    print(f"\n  {'t':>5} {'λ':>5} {'init':>12}  "
          f"{'Target μ':>12} {'BG μ':>12} {'Ratio':>8}")
    print("  " + "-" * 60)

    best_ratio = 0.0
    for t in [1.0, 2.0, 5.0]:
        for lam in [0.0, 0.5, 1.0]:
            for init in ["max_degree", "uniform"]:
                qs = QuantumScorer(t=t, lam=lam, init_method=init)
                probs_dict = qs.score_all(candidates, set(), instance)
                probs = np.array([probs_dict[v] for v in range(n)])
                target = [probs[v] for v in answer]
                bg = [probs[v] for v in range(n) if v not in answer]
                ratio = compute_ratio(target, bg)
                best_ratio = max(best_ratio, ratio)
                flag = " ★" if ratio > 1.0 else ""
                print(f"  {t:>5.1f} {lam:>5.2f} {init:>12}  "
                      f"{np.mean(target):>12.6f} {np.mean(bg):>12.6f} "
                      f"{ratio:>8.4f}{flag}")

    print()
    return _check(
        "至少一组参数下 Ratio > 1.0",
        best_ratio > 1.0,
        f"最大 Ratio = {best_ratio:.4f}（标 ★ 的参数组合）"
    )


# ============================================================
# 额外健全性：QuantumGuidedGreedy 端到端可跑
# ============================================================

def check_end_to_end(instance) -> bool:
    """跑一次 QuantumGuidedGreedy.solve，确保整个流程不报错。"""
    _section("检查 6：QuantumGuidedGreedy 端到端运行")
    algo = QuantumGuidedGreedy(
        CliqueCandidateSet(),
        t=1.0, lam=0.5, alpha=0.5,
        init_method="max_degree", seed=0,
        name="SmokeTest"
    )
    result = algo.solve(instance)

    # 解必须是合法团
    sol = set(result.solution)
    edges = instance.edge_set
    is_clique = True
    for u in sol:
        for v in sol:
            if u < v and (u, v) not in edges and (v, u) not in edges:
                is_clique = False
                break

    print(f"  找到的解: {sorted(sol)}")
    print(f"  团大小:   {len(sol)}（植入团大小 {len(instance.answer_set)}）")
    print(f"  迭代轮数: {result.iterations}")
    print(f"  运行时间: {result.runtime:.4f}s")

    return _check(
        "输出节点集合构成合法团",
        is_clique,
        "所有节点两两之间均有边"
    )


# ============================================================
# 主入口
# ============================================================

def main():
    print("=" * 60)
    print("  CTQW 真实实现冒烟测试 (Phase 3.1)")
    print("=" * 60)

    if not os.path.isfile(TEST_INSTANCE):
        print(f"\n错误: 找不到测试实例 {TEST_INSTANCE}")
        print("请确认已生成测试数据（cd datasets && python generate_all.py）")
        sys.exit(1)

    instance = load_instance(TEST_INSTANCE)
    print(f"\n测试实例: {instance.sample_id}")
    print(f"  节点数: {instance.num_nodes}")
    print(f"  边数:   {instance.num_edges}")
    print(f"  植入团: {sorted(instance.answer_set)}")

    # 依次执行所有检查
    results = [
        check_hamiltonian_hermitian(instance),
        check_unitary(instance),
        check_probability_sum(instance),
        check_probability_nonneg(instance),
        check_ratio_on_clique(instance),
        check_end_to_end(instance),
    ]

    # 汇总
    _section("汇总")
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n  通过项: {n_pass} / {n_total}")
    if n_pass == n_total:
        print(f"\n  ✓ 全部通过，CTQW 真实实现工作正常。")
        sys.exit(0)
    else:
        print(f"\n  ✗ 有 {n_total - n_pass} 项失败，请检查实现。")
        sys.exit(1)


if __name__ == "__main__":
    main()
