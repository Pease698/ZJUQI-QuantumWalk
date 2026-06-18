"""CTQW 演化计算模块 — 提供多种矩阵指数-向量作用近似方法。

统一接口：
    compute_ctqw_evolution(H, psi0, t, method, **kwargs) -> np.ndarray

支持的方法:
    - "exact":     scipy.linalg.expm 精确计算 (n ≤ 200 默认)
    - "krylov":    Lanczos + Krylov 子空间 (n > 200 默认，指数收敛)
    - "chebyshev": Chebyshev 多项式展开 (极低内存场景，O(3n) 额外存储)
    - "auto":      自动选择 (n≤200→exact, n>200→krylov)

设计要点:
    - H 必须是实对称矩阵（本项目哈密顿量满足此条件）
    - psi0 为归一化初态（本项目场景下为实向量）
    - Lanczos 使用实数算术全程，仅在最后 exp(-iTt) 引入复数
    - Chebyshev 需预估谱界 [λ_min, λ_max]，内部用少量 Lanczos 迭代估计
"""

import numpy as np
from scipy.linalg import expm
from scipy.special import jv as bessel_j


# ============================================================
# 统一入口
# ============================================================

def compute_ctqw_evolution(
    H: np.ndarray,
    psi0: np.ndarray,
    t: float,
    method: str = "auto",
    **kwargs,
) -> np.ndarray:
    """计算 |ψ(t)⟩ = e^{-iHt} · |ψ₀⟩。

    参数:
        H: n×n 实对称哈密顿量矩阵。
        psi0: 长度为 n 的归一化初态向量。
        t: 演化时间。
        method: "exact" | "krylov" | "chebyshev" | "auto"
        **kwargs: 传递给具体方法的超参数。
            krylov_dim (int): Krylov 子空间维数，默认 min(100, n)。
            krylov_tol (float): 早停容差，默认 1e-10。
            cheb_degree (int | None): Chebyshev 阶数，None 时自动估计。
            lambda_min / lambda_max (float | None): 谱界，None 时自动估计。

    返回:
        长度为 n 的复数向量 |ψ(t)⟩。
    """
    n = H.shape[0]

    if method == "auto":
        method = "exact" if n <= 200 else "krylov"

    if method == "exact":
        return _evolution_exact(H, psi0, t)
    elif method == "krylov":
        m = kwargs.get("krylov_dim", min(100, n))
        tol = kwargs.get("krylov_tol", 1e-10)
        return _evolution_krylov(H, psi0, t, m, tol)
    elif method == "chebyshev":
        d = kwargs.get("cheb_degree")
        lam_min = kwargs.get("lambda_min")
        lam_max = kwargs.get("lambda_max")
        return _evolution_chebyshev(H, psi0, t, d, lam_min, lam_max)
    else:
        raise ValueError(
            f"未知演化方法: {method}，支持 exact / krylov / chebyshev / auto"
        )


# ============================================================
# 精确计算
# ============================================================

def _evolution_exact(H: np.ndarray, psi0: np.ndarray, t: float) -> np.ndarray:
    """精确矩阵指数：exp(-iHt) @ psi0（O(n³)）。"""
    return expm(-1j * H * t) @ psi0


# ============================================================
# Krylov 子空间方法 (Lanczos)
# ============================================================

def _evolution_krylov(
    H: np.ndarray, psi0: np.ndarray, t: float,
    m: int, tol: float = 1e-10,
) -> np.ndarray:
    """Lanczos 算法近似 exp(-iHt)|ψ₀⟩。

    在 m 维 Krylov 子空间 K_m(H, ψ₀) 中构造标准正交基，
    将 H 投影为 m×m 三对角矩阵 T_m，在子空间内精确计算矩阵指数作用。

    复杂度: O(m·|E|) 稀疏情形，O(m·n²) 稠密情形。
    内存: O(m·n) 存储 Lanczos 向量 V_m。
    """
    n = H.shape[0]
    m = min(m, n)

    # Lanczos 迭代（实算术，因为 H 和 ψ₀ 均为实）
    V = np.zeros((n, m), dtype=np.float64)
    alpha = np.zeros(m, dtype=np.float64)
    beta = np.zeros(m - 1, dtype=np.float64)

    V[:, 0] = psi0.real.copy()

    for j in range(m):
        w = H @ V[:, j]

        alpha[j] = float(V[:, j] @ w)

        if j > 0:
            w -= beta[j - 1] * V[:, j - 1]
        w -= alpha[j] * V[:, j]

        # 完全重正交化（保证数值稳定性）
        for i in range(j + 1):
            w -= float(V[:, i] @ w) * V[:, i]

        b = float(np.linalg.norm(w))
        if j < m - 1:
            beta[j] = b
            if b < tol:
                m_eff = j + 1
                break
            V[:, j + 1] = w / b
    else:
        m_eff = m

    # 构建三对角矩阵 T_m
    T = (
        np.diag(alpha[:m_eff])
        + np.diag(beta[:m_eff - 1], k=1)
        + np.diag(beta[:m_eff - 1], k=-1)
    )

    # 在子空间内精确计算 exp(-iT_m t) @ e₁
    e1 = np.zeros(m_eff, dtype=np.complex128)
    e1[0] = 1.0
    exp_T_e1 = expm(-1j * T * t) @ e1

    # 映射回全空间
    return V[:, :m_eff] @ exp_T_e1


# ============================================================
# Chebyshev 多项式展开
# ============================================================

def _evolution_chebyshev(
    H: np.ndarray, psi0: np.ndarray, t: float,
    d: int | None = None,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
) -> np.ndarray:
    """Chebyshev 多项式展开近似 exp(-iHt)|ψ₀⟩。

    将 H 缩放至 [-1, 1] 后展开 exp(-iτx) = Σ c_k J_k(τ) T_k(x)，
    利用 T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x) 三项递推作用于向量。

    复杂度: O(d·|E|) 稀疏情形。
    内存: O(n) — 仅需 3 个长度为 n 的向量（w_{k-1}, w_k, w_{k+1}）。
    """
    n = H.shape[0]

    # 谱估计
    if lambda_min is None or lambda_max is None:
        lambda_min, lambda_max = _estimate_spectral_bounds(H, n_iter=20)

    mu = (lambda_max + lambda_min) / 2.0
    delta = (lambda_max - lambda_min) / 2.0
    tau = delta * t

    if delta < 1e-12:
        return np.exp(-1j * mu * t) * psi0

    # 自动确定截断阶数（Bessel 尾部衰减估计）
    if d is None:
        d = int(np.ceil(tau + 6.0 * tau ** (1.0 / 3.0)))
        d = max(d, 10)

    # 缩放矩阵的线性算子 H̃ = (H - μI) / δ（不显式构造矩阵）
    def _htilde_matvec(v: np.ndarray) -> np.ndarray:
        return (H @ v - mu * v) / delta

    # 三项递推
    w_old = psi0.copy()                     # T₀(H̃)|ψ₀⟩
    w_cur = _htilde_matvec(psi0)            # T₁(H̃)|ψ₀⟩

    result = np.zeros(n, dtype=np.complex128)
    result += bessel_j(0, tau) * w_old                    # k=0: c₀=1
    result += 2.0 * bessel_j(1, tau) * (-1j) * w_cur     # k=1: c₁=2

    for k in range(2, d + 1):
        w_new = 2.0 * _htilde_matvec(w_cur) - w_old      # T_k 递推
        coeff = 2.0 * bessel_j(k, tau) * ((-1j) ** k)    # 展开系数
        result += coeff * w_new
        w_old, w_cur = w_cur, w_new

    # 相位回补
    result *= np.exp(-1j * mu * t)
    return result


# ============================================================
# 谱界估计（少量 Lanczos 迭代）
# ============================================================

def _estimate_spectral_bounds(
    H: np.ndarray, n_iter: int = 20,
) -> tuple[float, float]:
    """用少量 Lanczos 迭代估计 H 的极值特征值。

    构造小规模三对角矩阵，其特征值极值可近似 H 的谱界。
    """
    n = H.shape[0]
    rng = np.random.default_rng(42)
    v = rng.normal(size=n).astype(np.float64)
    v /= np.linalg.norm(v)

    alphas = []
    betas = []
    v_old = np.zeros(n)

    for _ in range(n_iter):
        w = H @ v
        alpha = float(v @ w)
        alphas.append(alpha)

        w = w - alpha * v
        if betas:
            w -= betas[-1] * v_old

        b = float(np.linalg.norm(w))
        if b < 1e-14:
            break
        betas.append(b)
        v_old = v
        v = w / b

    m_eff = len(alphas)
    if m_eff <= 1:
        return -1.0, 1.0

    T = (
        np.diag(alphas)
        + np.diag(betas[:m_eff - 1], k=1)
        + np.diag(betas[:m_eff - 1], k=-1)
    )
    eigvals = np.linalg.eigvalsh(T)

    # 用 Gershgorin 圆盘定理扩展边界以确保安全
    margin = betas[-1] if betas else 0.5
    return float(eigvals[0] - margin), float(eigvals[-1] + margin)
