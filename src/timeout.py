"""算法运行超时控制 — 供 exp6 系列大规模实验使用。

使用 multiprocessing.Process 将算法求解隔离到独立子进程：
  - 子进程正常完成 → 通过 Pipe 传回 AlgorithmResult
  - 子进程超时 → terminate() 安全终止，返回 timed_out=True 的标记结果

设计要点:
  - 子进程被 kill 不会破坏父进程的 numpy/scipy 状态
  - 使用 "spawn" 上下文避免 fork 带来的资源继承问题
  - 不在 runner.py / config.py 中集成，仅由 exp6_*.py 显式调用
"""

import multiprocessing as mp

from .algorithms.base import BaseAlgorithm, AlgorithmResult
from .graph_utils import GraphInstance


def run_with_timeout(
    algo: BaseAlgorithm,
    instance: GraphInstance,
    timeout_sec: float,
) -> AlgorithmResult:
    """在子进程中运行 algo.solve(instance)，超时返回标记结果。

    参数:
        algo: 算法对象（必须可 pickle）。
        instance: 图实例（必须可 pickle）。
        timeout_sec: 超时门限（秒）。None 或 ≤0 表示不设超时，主进程直接运行。

    返回:
        AlgorithmResult。若超时，result.timed_out = True，
        result.objective = NaN，result.solution = []。
    """
    if timeout_sec is None or timeout_sec <= 0:
        return algo.solve(instance)

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    p = ctx.Process(
        target=_solve_subprocess,
        args=(child_conn, algo, instance),
    )
    p.start()
    child_conn.close()

    if parent_conn.poll(timeout_sec):
        status, payload = parent_conn.recv()
        p.join()
        parent_conn.close()

        if status == "ok":
            return payload
        return _make_result(
            algo, instance, timeout_sec,
            timed_out=False, extra_params={"error": payload},
        )
    else:
        p.terminate()
        p.join()
        parent_conn.close()
        return _make_result(algo, instance, timeout_sec, timed_out=True)


def _solve_subprocess(conn, algo: BaseAlgorithm, instance: GraphInstance) -> None:
    """子进程入口：运行算法并通过 Pipe 传回 (status, payload)。"""
    try:
        result = algo.solve(instance)
        conn.send(("ok", result))
    except Exception as exc:
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def _make_result(
    algo: BaseAlgorithm,
    instance: GraphInstance,
    timeout_sec: float,
    timed_out: bool,
    extra_params: dict | None = None,
) -> AlgorithmResult:
    """构造超时/错误标记结果。"""
    params = {
        "timed_out": timed_out,
        "timeout_limit": timeout_sec,
    }
    if extra_params:
        params.update(extra_params)

    label = "[TIMEOUT]" if timed_out else "[ERROR]"
    return AlgorithmResult(
        algorithm=f"{algo.name}{label}",
        sample_id=instance.sample_id,
        task_type=instance.task_type,
        solution=[],
        objective=float("nan"),
        runtime=timeout_sec,
        iterations=0,
        timed_out=timed_out,
        params=params,
    )
