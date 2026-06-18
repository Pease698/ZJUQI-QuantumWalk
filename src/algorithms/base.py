"""算法抽象基类与结果数据结构。

所有算法（经典贪心、模拟退火、量子引导贪心）都实现 BaseAlgorithm 接口，
返回统一的 AlgorithmResult 数据结构。
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..graph_utils import GraphInstance
from ..candidate_set import CandidateSetBuilder


@dataclass
class AlgorithmResult:
    """算法运行结果的标准数据结构。

    所有算法必须返回此结构的实例，以便 runner 统一收集和比较。
    """

    algorithm: str                        # 算法名称标识
    sample_id: str                        # 测试实例 ID
    task_type: str                        # "maximum_clique" | "densest_subgraph"
    solution: list[int]                   # 算法选出的节点列表
    objective: float                      # 目标函数值（团大小 或 密度）
    runtime: float                        # 运行时间（秒）
    iterations: int                       # 迭代轮数
    history: list[dict] = field(          # 每轮状态记录
        default_factory=list,
        repr=False,
    )
    # 可选：与 ground truth 的对比信息
    answer_size: int = 0
    recall: float = 0.0
    timed_out: bool = False             # 是否因超时被终止（exp6 使用）
    params: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        """转为扁平字典，便于写入 DataFrame。"""
        return {
            "algorithm": self.algorithm,
            "sample_id": self.sample_id,
            "task_type": self.task_type,
            "objective": self.objective,
            "runtime": self.runtime,
            "iterations": self.iterations,
            "solution_size": len(self.solution),
            "answer_size": self.answer_size,
            "recall": self.recall,
            "timed_out": self.timed_out,
            "solution": self.solution,
            **self.params,
        }


class BaseAlgorithm(ABC):
    """所有算法的抽象基类。

    子类只需实现 solve(instance) -> AlgorithmResult。

    通过构造函数注入 CandidateSetBuilder 和 Scorer，
    不同算法变体通过组合产生，而非继承。
    """

    def __init__(self, candidate_builder: CandidateSetBuilder,
                 scorer=None, name: str = "BaseAlgorithm"):
        self.candidate_builder = candidate_builder
        self.scorer = scorer
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def solve(self, instance: GraphInstance) -> AlgorithmResult:
        """在给定图上运行算法，返回结果。"""
        ...

    def _build_result(self, instance: GraphInstance, solution: list[int],
                      objective: float, runtime: float, iterations: int,
                      history: list[dict] | None = None,
                      extra_params: dict | None = None) -> AlgorithmResult:
        """构造 AlgorithmResult 的辅助方法。"""
        answer_set = instance.answer_set
        recall = 0.0
        if answer_set:
            found = sum(1 for v in solution if v in answer_set)
            recall = found / len(answer_set)

        return AlgorithmResult(
            algorithm=self.name,
            sample_id=instance.sample_id,
            task_type=instance.task_type,
            solution=solution,
            objective=objective,
            runtime=runtime,
            iterations=iterations,
            history=history or [],
            answer_size=len(answer_set),
            recall=recall,
            params=extra_params or {},
        )
