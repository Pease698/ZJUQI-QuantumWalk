"""配置管理：参数定义、路径管理、实验预设。

所有实验参数集中由此管理，避免在脚本中散落魔法数字。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---- 项目根路径 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---- 实验参数预设 ----
# 这些值来自理论文档 §13.1 实验设置总表


@dataclass
class ExperimentConfig:
    """单组实验的完整配置。

    对应理论文档 §13.1 中的实验因素，每个字段都有一个"建议取值"集合。
    """

    # ---- 图参数 ----
    n: int                           # 节点总数
    p: float                         # 背景边概率
    k: int                           # 目标子图大小
    rho: float | None = None        # 目标边密度 (None 表示团任务)

    # ---- CTQW 参数 ----
    t: float = 1.0                   # 演化时间
    lam: float = 0.0                 # 扰动强度 λ
    init_method: str = "max_degree"  # 初态方法: uniform | max_degree | random

    # ---- 混合参数 ----
    alpha: float = 0.5               # 量子权重 α ∈ [0, 1]

    # ---- 实验参数 ----
    repeat_runs: int = 4             # 每个实例重复运行次数
    seed: int = 0                    # 基础随机种子

    def __post_init__(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha 必须在 [0, 1] 内，当前为 {self.alpha}")
        if self.init_method not in ("uniform", "max_degree", "random"):
            raise ValueError(f"不支持的 init_method: {self.init_method}")


# ---- CTQW 参数扫描预设（理论 §13.5）----
T_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0]
LAMBDA_VALUES = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

# ---- 算法名称常量 ----
ALGO_CLASSICAL_DEGREE = "ClassicalDegree"
ALGO_CLASSICAL_CLIQUE = "ClassicalClique"
ALGO_CLASSICAL_DENSE = "ClassicalDense"
ALGO_SIMULATED_ANNEALING = "SimulatedAnnealing"
ALGO_QUANTUM_GREEDY = "QuantumGuidedGreedy"


def ensure_results_dir() -> Path:
    """创建并返回 results 目录路径。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def get_data_dirs(task_type: str) -> list[Path]:
    """返回指定任务类型下所有参数组合的数据目录。

    参数:
        task_type: "maximum_clique" 或 "densest_subgraph"

    返回:
        所有子目录的 Path 列表，每个子目录包含该参数组合的 JSON 实例。
    """
    artificial_dir = DATA_DIR / "artificial" / task_type
    external_dir = DATA_DIR / "external" / task_type

    dirs = []
    for base in [artificial_dir, external_dir]:
        if base.is_dir():
            for entry in sorted(base.iterdir()):
                if entry.is_dir():
                    dirs.append(entry)
    return dirs
