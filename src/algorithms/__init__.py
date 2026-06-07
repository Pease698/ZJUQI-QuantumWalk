"""算法实现模块。

包含：
  base                — 算法抽象基类 + 结果数据结构
  classical_greedy    — 经典贪心算法（多种变体通过组合 Scorer 实现）
  simulated_annealing — 模拟退火算法
  quantum_greedy      — 量子引导贪心算法（CTQW 计算模块占位）
"""

from .base import BaseAlgorithm, AlgorithmResult
from .classical_greedy import ClassicalGreedy
from .simulated_annealing import SimulatedAnnealing
from .quantum_greedy import QuantumGuidedGreedy
