# src — 量子引导贪心算法核心库
#
# 模块层级：
#   config          — 配置管理（参数定义、路径、预设）
#   graph_utils     — 图加载与邻接矩阵构建
#   candidate_set   — 候选集合构造（理论 §7）
#   scoring         — 评分函数（理论 §8）：经典评分 + 量子评分 + 混合评分
#   hamiltonian     — 哈密顿量构造（理论 §5）
#   initial_state   — 初始态构造（理论 §6）
#   algorithms/     — 算法实现：经典贪心、模拟退火、量子引导贪心
#   runner          — 实验运行器：批量执行 + 结果收集
#   metrics         — 评价指标
#
# 外部使用者应通过 src 包级别的 import 使用：
#   from src import ClassicalGreedy, CliqueCandidateSet, run_experiment

from .graph_utils import load_instance, build_adjacency, GraphInstance
from .candidate_set import (
    CandidateSetBuilder,
    CliqueCandidateSet,
    DenseCandidateSet,
)
from .scoring import (
    Scorer,
    ClassicalCliqueScorer,
    ClassicalDenseScorer,
    ClassicalDegreeScorer,
    HybridScorer,
    QuantumScorer,
)
from .algorithms import (
    AlgorithmResult,
    BaseAlgorithm,
    ClassicalGreedy,
    SimulatedAnnealing,
    QuantumGuidedGreedy,
)
from .runner import run_experiment, run_single_instance
from .metrics import (
    compute_ratio,
    success_rate,
    mean_std,
    evaluate_solution,
)
from .config import ExperimentConfig, DATA_DIR, RESULTS_DIR
