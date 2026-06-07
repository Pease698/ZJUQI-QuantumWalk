# src — 量子引导贪心算法核心库

## 概述

`src/` 是本项目的算法核心库，实现了量子信息理论文档中定义的全套算法框架。设计理念为**模块化组合**——算法行为通过注入不同的 `CandidateSetBuilder`（候选集合构造器）和 `Scorer`（评分函数）来控制，而非通过大量子类继承。每一模块的理论依据均标注在对应文件中。

## 架构总览

```
src/
├── __init__.py                 # 包入口，导出所有公开接口
├── README.md                   # 本文件
│
├── config.py                   # 配置管理：参数定义、路径、实验预设
├── graph_utils.py              # 图加载：JSON→GraphInstance→邻接矩阵/NetworkX
│
├── candidate_set.py            # 候选集合构造（理论 §7）
│   ├── CandidateSetBuilder     #   抽象基类
│   ├── CliqueCandidateSet      #   最大团：C(S) = {v | ∀u∈S, (u,v)∈E}
│   └── DenseCandidateSet       #   密集子图：C(S) = {v | ρ(S∪{v}) ≥ θ}
│
├── scoring.py                  # 评分函数（理论 §8）★ 最关键的抽象层
│   ├── Scorer                  #   评分函数统一接口
│   ├── ClassicalCliqueScorer   #   最大团经典评分：deg_{C(S)}(v)
│   ├── ClassicalDenseScorer    #   密集子图经典评分：Δρ(v,S)
│   ├── ClassicalDegreeScorer   #   纯度数评分（基线对照）
│   ├── QuantumScorer           #   量子概率评分：[占位] 待 CTQW 模块完成
│   └── HybridScorer            #   混合评分组合器：α·Q_norm + (1-α)·R_norm
│
├── hamiltonian.py              # 哈密顿量构造（理论 §5）
│   └── construct_hamiltonian() #   H = A + λ·Σ|u⟩⟨u|
│
├── initial_state.py            # 初始态构造（理论 §6）
│   └── build_initial_state()   #   |ψ₀⟩，支持 uniform/max_degree/random
│
├── algorithms/                 # 算法实现
│   ├── __init__.py
│   ├── base.py                 #   BaseAlgorithm 抽象基类 + AlgorithmResult
│   ├── classical_greedy.py     #   经典贪心算法（多种变体通过组合实现）
│   ├── simulated_annealing.py  #   模拟退火算法（经典启发式对照）
│   └── quantum_greedy.py       #   量子引导贪心（算法主流程，CTQW 占位）
│
├── runner.py                   # 实验运行器：批量执行 + 结果收集 → DataFrame
└── metrics.py                  # 评价指标：Ratio, success_rate, mean_std 等
```

## 数据流

```
JSON 文件
  │
  ├─→ graph_utils.load_instance()  ──→  GraphInstance
  │      ├── .adjacency   (np.ndarray, n×n)
  │      ├── .nx_graph    (nx.Graph)
  │      ├── .edge_set    (set, O(1) lookup)
  │      └── .answer_set  (ground truth)
  │
  ├─→ candidate_set.CandidateSetBuilder.build(adj, edges, S, all_nodes)
  │      └──→ candidates: set[int]
  │
  ├─→ scoring.Scorer.score_all(candidates, S, instance)
  │      └──→ {v: score} 映射
  │
  ├─→ algorithms.BaseAlgorithm.solve(instance)
  │      └──→ AlgorithmResult
  │
  └─→ metrics.evaluate_solution(instance, solution)
         └──→ objective: float
```

## 模块详细说明

### config.py — 配置管理

集中管理所有实验参数，避免散落魔法数字。

**核心类型**：
- `ExperimentConfig` — 单组实验的完整参数集合（图参数、CTQW 参数、混合参数、实验参数）
- `T_VALUES`, `LAMBDA_VALUES`, `ALPHA_VALUES` — 参数扫描预设（对应理论 §13.5）
- `get_data_dirs(task_type)` — 获取所有参数组合的数据目录

### graph_utils.py — 图加载与邻接矩阵

从统一 JSON 格式加载测试实例，自动构建两种图表示：

**核心类型**：
- `GraphInstance` — 内存中的图数据对象
  - `adjacency` (property): lazy 构建的 NumPy 邻接矩阵
  - `nx_graph` (property): lazy 构建的 NetworkX 图
  - `edge_set` (property): 无序边集合，用于 O(1) 邻接查询
  - `answer_set` (property): ground truth 节点集合

**核心函数**：
- `load_instance(filepath)` — 从 JSON 加载单个实例
- `load_instances_from_dir(directory)` — 批量加载目录中所有实例
- `build_adjacency(n, edges)` — 从边列表构建邻接矩阵

### candidate_set.py — 候选集合构造（理论 §7）

候选集合是"可行性投影"——在量子概率分布和组合优化约束之间的桥梁。

**设计原则**：先由约束筛选合法候选，再由评分函数排序。这样 CTQW 不需要知道最大团或密集子图的具体约束。

**核心类**：
- `CandidateSetBuilder` — 抽象基类，定义 `build(adjacency, edge_set, S, all_nodes) → set[int]`
- `CliqueCandidateSet` — 最大团候选集合，要求候选与 S 中所有节点相连
- `DenseCandidateSet(theta)` — 密集子图候选集合，可配置密度阈值 θ

### scoring.py — 评分函数（理论 §8）★

**这是整个架构最关键的抽象层。** 评分函数完全独立于算法逻辑，支持以下实验需求：

| 实验 | 评分组合 | α | 目的 |
|------|---------|---|------|
| 实验二：对比 | ClassicalCliqueScorer / ClassicalDegreeScorer | 0 | 经典基线 |
| 实验二：对比 | QuantumScorer | 1 | 纯量子评分 |
| 实验三：消融 | ClassicalCliqueScorer only | 0 | 无量子 |
| 实验三：消融 | QuantumScorer only | 1 | 无经典 |
| 实验三：消融 | HybridScorer(Quantum, Classical, α=0.5) | 0.5 | 完整混合 |
| 实验四：敏感性 | HybridScorer(Quantum, Classical, α) | 扫描 [0,1] | α 分析 |

**核心类**：
- `Scorer` — 抽象基类，定义 `score(v, S, instance)` 和 `score_all(candidates, S, instance)`
- `ClassicalCliqueScorer` — 最大团经典评分 `deg_{C(S)}(v)`，衡量后续扩展能力
- `ClassicalDenseScorer` — 密集子图经典评分 `ρ(S∪{v}) - ρ(S)`，直接对应目标函数
- `ClassicalDegreeScorer` — 纯度数评分 `deg(v)`，作为最简基线
- `QuantumScorer` — 量子概率评分 `P_v(t)`，当前为占位实现，返回均匀随机概率
- `HybridScorer(scorer_a, scorer_b, alpha)` — 组合任意两个评分器，自动 min-max 归一化

**HybridScorer 的关键设计**：
- 不硬编码"量子 + 经典"——传入任意两个 Scorer 对象
- `alpha=0` 退化为纯 `scorer_b`，`alpha=1` 退化为纯 `scorer_a`
- 归一化仅在候选集合上计算（理论 §8.3 要求）

**当前状态**：`QuantumScorer` 为占位实现，所有经典评分函数已可用。CTQW 模块完成后，只需替换 `QuantumScorer` 的实现即可接入。

### hamiltonian.py — 哈密顿量构造（理论 §5）

构造量子演化所需的哈密顿量矩阵：

```
H = A + λ · Σ_{u∈S} |u⟩⟨u|
```

其中 `|u⟩⟨u|` 等价于在第 u 个对角位置加 1。当前 `λ=0` 时直接返回邻接矩阵 A，对应纯图结构演化。

### initial_state.py — 初始态构造（理论 §6）

构造量子游走的初始态 `|ψ₀⟩`，支持三种初始化方式：

| 方法 | 行为 | 适用场景 |
|------|------|---------|
| `uniform` | 所有节点等权 1/√n | 无先验偏置 |
| `max_degree` | 选度数最高节点 | 团和密集子图任务 |
| `random` | 随机单节点（固定 seed） | 对照实验 |

当 S 非空时，自动使用种子集合均匀初态 `(1/√|S|)·Σ|u⟩`。

### algorithms/ — 算法实现

#### base.py — 算法基类

- `BaseAlgorithm` — 抽象基类，子类只需实现 `solve(instance) → AlgorithmResult`
- `AlgorithmResult` — 统一的结果数据结构，包含算法名、解、目标值、运行时间、历史记录等

#### classical_greedy.py — 经典贪心算法

实现理论 §12 中贪心框架的经典版本。每轮选择当前候选集合中评分最高的节点。

**不同变体通过组合实现**，无需子类化：
```python
# 纯度数贪心（基线）
algo = ClassicalGreedy(CliqueCandidateSet(), ClassicalDegreeScorer())

# 最大团经典贪心
algo = ClassicalGreedy(CliqueCandidateSet(), ClassicalCliqueScorer())

# 密集子图经典贪心
algo = ClassicalGreedy(DenseCandidateSet(), ClassicalDenseScorer())
```

#### simulated_annealing.py — 模拟退火算法

经典启发式全局搜索方法，作为量子引导算法的对照（理论 §13.3）。

**核心参数**：
- `T0` — 初始温度（默认 1.0）
- `cooling_rate` — 温度衰减因子（默认 0.995）
- `max_iterations` — 最大迭代次数（默认 5000）

邻域操作：50% 概率加入随机合法节点，50% 概率移除随机节点。接受准则为标准 Metropolis 准则。

#### quantum_greedy.py — 量子引导贪心算法

完整实现了理论 §12 的算法流程（CTQW 演化部分为占位）。待 CTQW 模块完成后，只需替换 `_placeholder_ctqw()` 方法即可。

### runner.py — 实验运行器

批量执行实验的核心入口：

- `run_experiment(data_dir, algorithms, config)` — 对单目录运行实验
- `run_single_instance(instance, algorithms, config)` — 对单实例运行所有算法
- `run_all_datasets(task_type, algorithms, config)` — 遍历所有参数组合

所有结果收集为 pandas DataFrame，可直接用于统计分析和可视化。

### metrics.py — 评价指标

| 函数 | 公式 | 对应假设 |
|------|------|---------|
| `compute_ratio` | Mean(P_target) / Mean(P_bg) | 假设一：CTQW 概率反映全局结构 |
| `evaluate_solution` | |S| 或 ρ(S) | 假设二：量子评分改善质量 |
| `success_rate` | count(≥threshold) / total | 统计显著性 |
| `mean_std` | μ, σ | 多次运行稳定性 |
| `aggregate_results` | 汇总统计 | 实验报告 |

## 使用示例

### 快速开始：对单个实例运行实验

```python
from src import (
    ClassicalGreedy, SimulatedAnnealing, QuantumGuidedGreedy,
    CliqueCandidateSet, DenseCandidateSet,
    ClassicalCliqueScorer, ClassicalDenseScorer, ClassicalDegreeScorer,
    load_instance, run_single_instance, ExperimentConfig,
)

# 加载测试数据
instance = load_instance("datasets/data/artificial/maximum_clique/"
                         "mc_n100_p02_k10/mc_n100_p02_k10_000.json")

# 定义算法集合
algorithms = {
    "DegreeGreedy": ClassicalGreedy(
        CliqueCandidateSet(), ClassicalDegreeScorer()),
    "CliqueGreedy": ClassicalGreedy(
        CliqueCandidateSet(), ClassicalCliqueScorer()),
    "SimulatedAnnealing": SimulatedAnnealing(
        CliqueCandidateSet()),
    "QuantumGuided": QuantumGuidedGreedy(
        CliqueCandidateSet(), t=1.0, lam=0.5, alpha=0.5),
}

# 运行实验
config = ExperimentConfig(n=100, p=0.2, k=10, repeat_runs=4)
results = run_single_instance(instance, algorithms, config)

for row in results:
    print(f"{row['algorithm']}: obj={row['objective']}, "
          f"time={row['runtime']:.3f}s")
```

### 批量实验

```python
from src import run_all_datasets, ExperimentConfig
from src.algorithms.classical_greedy import ClassicalGreedy
from src.candidate_set import CliqueCandidateSet
from src.scoring import ClassicalCliqueScorer

config = ExperimentConfig(n=100, p=0.2, k=10, repeat_runs=4)
algorithms = {
    "ClassicalClique": ClassicalGreedy(
        CliqueCandidateSet(), ClassicalCliqueScorer()),
}

df = run_all_datasets("maximum_clique", algorithms, config,
                      output_dir="results/maximum_clique_baseline")
```

### 运行预设实验

项目在 `experiments/` 目录下提供了四组预设实验脚本和一个通用运行器，对应理论文档 §13 的实验设计：

| 脚本 | 理论 | 目的 | 适用数据 | 状态 |
|------|------|------|---------|------|
| `exp1_ctqw_visualization.py` | §13.2 | CTQW 概率可视化 | 仅人工 | CTQW 占位 |
| `exp2_algorithm_comparison.py` | §13.3 | 算法对比 ★ 核心 | 人工 + 外部 | 经典算法可运行 |
| `exp3_ablation.py` | §13.4 | 消融实验 | 仅人工 | CTQW 占位 |
| `exp4_sensitivity.py` | §13.5 | 参数敏感性 | 仅人工 | CTQW 占位 |
| `run_experiment.py` | 通用 | 自由组合 | 人工 + 外部 | 可运行 |

重复策略：
- **人工数据**：5 实例/组 × 4 次/实例 = 20 数据点（§13.1 要求）
- **外部数据**：1 实例/数据集 × 10 次 = 10 数据点（无跨图方差，增多运行内统计）

#### 实验一：CTQW 概率分布可视化（§13.2）

```bash
# 参数: --task --n --k [--p] [--t] [--lam] [--init] [--instance] [--save]
python3 experiments/exp1_ctqw_visualization.py --task maximum_clique --n 30 --k 5
```
产出：概率分布节点图（目标高亮）、度数 vs CTQW 对比图、Ratio 指标 JSON。

#### 实验二：算法对比（§13.3）★ 核心实验

```bash
# 参数: --task --data-source [--small|--full] [--csv]
# 人工数据（5实例/组 × 4次重复 = 20数据点）
python3 experiments/exp2_algorithm_comparison.py --task maximum_clique --small

# 外部 DIMACS 数据（1实例/数据集 × 10次重复）
python3 experiments/exp2_algorithm_comparison.py \
    --task maximum_clique --data-source external
```
对照方法：ClassicalDegree、 ClassicalClique、 SimulatedAnnealing、 QuantumGuidedGreedy。
产出：箱线图、分组柱状图、汇总统计 CSV。

#### 实验三：消融实验（§13.4）

```bash
python3 experiments/exp3_ablation.py --task maximum_clique --small
```
消融设置：A(纯经典) / B(量子,λ=0) / C(量子,λ>0) / D(完整混合)。产出：四组箱线图、提升图、CSV。

#### 实验四：参数敏感性（§13.5）

```bash
# 参数: --task [--instance] [--instance-idx]
python3 experiments/exp4_sensitivity.py --task maximum_clique
```
扫描 t×λ×α。产出：三张热力图、敏感性曲线、完整扫描 CSV。

#### 通用实验运行器

```bash
# 自由组合数据源、算法、评分方式和参数
python3 experiments/run_experiment.py \
    --task maximum_clique --data-source artificial --small \
    --algorithms ClassicalDegree ClassicalClique SimulatedAnnealing

# 外部数据
python3 experiments/run_experiment.py \
    --task maximum_clique --data-source external \
    --algorithms ClassicalDegree ClassicalClique SimulatedAnnealing

# 单文件 + 自定义参数
python3 experiments/run_experiment.py \
    --task maximum_clique --data-source file \
    --data-path path/to/instance.json \
    --algorithms QuantumGuidedGreedy --scorer hybrid --alpha 0.7
```
完整参数见 `python3 experiments/run_experiment.py --help`。

#### 数据加载路径

实验脚本通过两层发现定位测试数据：

1. **参数组发现** (`get_data_dirs`)：遍历 `data/artificial/{task}/` 下的子目录（每个子目录 = 一个参数组，含 5 个 JSON）
2. **实例加载** (`load_instances_from_dir`)：加载目录内所有 `*.json`
3. **外部数据** (`--data-source external`)：直接遍历 `data/external/{task}/*.json`（平铺，每个数据集 1 个 JSON）

所有结果输出到 `results/` 下按实验名称命名的子目录。

## CTQW 模块接入指南

待 CTQW 计算模块完成后，按以下三步接入：

### 步骤 1：实现 `QuantumScorer.score_all()`

在 [scoring.py](scoring.py) 的 `QuantumScorer` 类中，将占位实现替换为：

```python
def score_all(self, candidates, S, instance):
    from ..hamiltonian import construct_hamiltonian
    from ..initial_state import build_initial_state
    from scipy.linalg import expm

    n = instance.num_nodes
    H = construct_hamiltonian(instance.adjacency, S, self.lam)
    psi0 = build_initial_state(n, S, instance.adjacency, self.init_method)

    U = expm(-1j * H * self.t)
    psi_t = U @ psi0
    probs = np.abs(psi_t) ** 2

    return {v: probs[v] for v in candidates}
```

### 步骤 2：替换 `QuantumGuidedGreedy` 中的占位

在 [algorithms/quantum_greedy.py](algorithms/quantum_greedy.py) 中，将 `_placeholder_ctqw()` 调用替换为直接使用 `QuantumScorer`。

### 步骤 3：验证

```python
# 确保 QuantumScorer 的评分与 QuantumGuidedGreedy 内部一致
q_scorer = QuantumScorer(t=1.0, lam=0.5)
algo = QuantumGuidedGreedy(builder, t=1.0, lam=0.5)
```

接入完成后，所有实验脚本和运行器无需任何修改即可使用真实 CTQW 评分。

## 模块间依赖关系

```
                        config.py
                            ↑
                     runner.py ──→ metrics.py
                       ↑    ↑
         graph_utils.py    algorithms/
              ↑              ↑     ↑
       candidate_set.py    base.py scoring.py
              ↑              ↑        ↑
       hamiltonian.py    classical_greedy.py
              ↑          simulated_annealing.py
       initial_state.py  quantum_greedy.py
```

依赖规则：
- 上层模块依赖下层，但评分函数 (`scoring.py`) 可独立被算法层、运行器层和指标层使用
- `algorithms/` 子模块互相独立，仅依赖 `base.py` 和外部接口
- `runner.py` 是唯一协调所有模块的顶层入口

## 设计决策记录

1. **评分函数独立于算法**：因为消融实验需要精确控制评分组合（理论 §13.4），如果评分嵌入在算法内会产生组合爆炸。
2. **HybridScorer 是通用组合器而非特定量子+经典的组合**：允许任意两个评分器混合，适应未来可能的扩展（如加入 PageRank 评分）。
3. **候选集合与评分函数解耦**：候选集合做"可行性投影"，评分函数做"质量排序"——对应理论文档中"先保证合法，再比较优劣"的设计理念。
4. **数据层使用统一 JSON 格式**：`GraphInstance` 封装了邻接矩阵、NetworkX 图和边集合的 lazy 构建，算法代码无需关心数据来源。
5. **CTQW 模块占位但接口就绪**：`QuantumScorer` 和 `QuantumGuidedGreedy` 的接口已就位，后续只需替换内部实现。
