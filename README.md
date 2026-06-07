# ZJUQI-QuantumWalk

浙江大学量子信息课程的量子引导贪心算法课题。

## 项目概述

本项目的核心目标是验证一种明确的算法假设：

> 连续时间量子游走（CTQW）产生的节点概率分布，能够比传统局部指标更好地反映图的全局拓扑结构，并可作为贪心算法的节点评分依据，从而提升图组合优化问题的求解质量。

项目整体采用 **"量子全局视野 + 经典局部贪心"** 的双层架构。CTQW 负责生成包含全局拓扑信息的节点概率分布，经典贪心策略负责在候选集合中完成节点选择。

## 项目结构

```
ZJUQI-QuantumWalk/
├── src/                         # 算法核心库（详见 src/README.md）
│   ├── config.py                #   配置管理
│   ├── graph_utils.py           #   图加载与邻接矩阵构建
│   ├── candidate_set.py         #   候选集合构造（理论 §7）
│   ├── scoring.py               #   评分函数 ★ 独立模块（理论 §8）
│   ├── hamiltonian.py           #   哈密顿量构造（理论 §5）
│   ├── initial_state.py         #   初始态构造（理论 §6）
│   ├── algorithms/              #   算法实现
│   │   ├── base.py              #     算法基类 + 结果数据结构
│   │   ├── classical_greedy.py  #     经典贪心算法
│   │   ├── simulated_annealing.py #   模拟退火算法
│   │   └── quantum_greedy.py    #     量子引导贪心 [CTQW 占位]
│   ├── runner.py                #   实验运行器
│   ├── metrics.py               #   评价指标
│   └── README.md                #   src 模块详细文档
│
├── datasets/                    # 测试数据集
│   ├── generators/              #   人工图生成器
│   ├── converters/              #   外部数据格式转换器
│   ├── data/                    #   生成的 JSON 数据
│   │   ├── artificial/          #     人工植入数据（答案已知，5实例/组）
│   │   └── external/            #     外部 DIMACS 数据（答案仅知上界）
│   ├── generate_all.py          #   一键生成入口
│   ├── visualize.py             #   可视化脚本
│   └── README.md                #   数据集详细文档
│
├── experiments/                 # 实验脚本（对应理论 §13 四组实验）
│   ├── exp1_ctqw_visualization.py    #   实验一：CTQW 概率分布可视化
│   ├── exp2_algorithm_comparison.py  #   实验二：与经典贪心算法对比（支持外部数据）
│   ├── exp3_ablation.py              #   实验三：消融实验
│   ├── exp4_sensitivity.py           #   实验四：参数敏感性分析
│   └── run_experiment.py             #   通用实验运行器（自由组合）
├── results/                     # 实验结果输出（CSV + 图表）
├── image/                       # 理论文档插图
├── 量子信息理论部分.md           # 理论文档
├── 量子信息理论部分.pdf          # 理论文档（PDF）
├── requirements.txt             # 项目依赖
└── README.md                    # 本文件
```

## 快速开始

### 环境配置

```bash
pip install -r requirements.txt
```

### 生成测试数据

```bash
cd datasets
python generate_all.py              # 生成全部人工测试实例
python generate_all.py --dry-run     # 预览生成计划

# 外部数据格式转换（DIMACS等 → 统一JSON）
python -m converters.convert_dimacs
```

## 实验脚本

五个实验脚本位于 `experiments/` 目录下，对应理论文档 §13 的实验设计和灵活的探索性实验：

| 脚本 | 理论依据 | 目的 | 适用数据 | CTQW 依赖 |
|------|---------|------|---------|-----------|
| `exp1_ctqw_visualization.py` | §13.2 | CTQW 概率分布可视化 | 仅人工 | 占位 |
| `exp2_algorithm_comparison.py` | §13.3 | 算法对比 ★ 核心实验 | **人工 + 外部** | 占位 |
| `exp3_ablation.py` | §13.4 | 消融实验 | 仅人工 | 占位 |
| `exp4_sensitivity.py` | §13.5 | 参数敏感性分析 | 仅人工 | 占位 |
| `run_experiment.py` | 通用 | 自由组合数据/算法/参数 | **人工 + 外部** | 占位 |

实验脚本的重复策略：
- **人工数据**：每组 5 个实例 × 每实例 4 次 = 20 数据点（符合 §13.1 "每组至少 20 次重复"）
- **外部数据**：每数据集 1 个实例 × 每实例 10 次 = 10 数据点（无跨图方差，需更多运行内统计）

> **关于实验图表**：`experiments/` 目录下四个专有实验脚本（exp1~exp4）中的图表生成逻辑为初步版本，
> 仅用于快速验证算法行为和获取初步数值结果。后续可根据论文需求调整图表样式、颜色方案、
> 标注方式等。所有绘图逻辑集中在各脚本的 `analyze_and_plot` / `plot_*` 函数中，
> 修改时无需改动算法运行逻辑。`run_experiment.py` 的绘图逻辑同样可独立修改。

### 实验一：CTQW 概率分布可视化（§13.2）

验证 CTQW 是否能在团或密集子图附近形成概率集中。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务类型 | `maximum_clique` |
| `--instance` | 指定 JSON 文件路径 | 自动搜索 |
| `--n` | 节点数 | 30 |
| `--p` | 背景边概率 | 自动匹配 |
| `--k` | 目标子图大小 | 自动匹配 |
| `--t` | CTQW 演化时间 | 1.0 |
| `--lam` | 扰动强度 λ | 0.0 |
| `--init` | 初态方式: uniform / max_degree / random | `max_degree` |

产出：
- 概率分布节点图（目标区域高亮）
- 度数 vs CTQW 概率散点图 + 排名对比图
- Ratio 指标 JSON

```bash
python3 experiments/exp1_ctqw_visualization.py --task maximum_clique --n 30 --k 5
```

### 实验二：算法对比（§13.3）★ 核心实验

验证量子引导评分是否提升求解质量。**这是唯一同时支持人工数据和外部数据的固定流程实验。**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务类型 | `maximum_clique` |
| `--data-source` | `artificial`（人工）或 `external`（DIMACS） | `artificial` |
| `--small` | 仅 n≤50（仅对 artificial 有效） | 开启 |
| `--full` | 使用全部数据（覆盖 --small） | — |
| `--csv` | 直接分析已有 CSV，跳过运行 | — |

对照方法：ClassicalDegree、 ClassicalClique/ClassicalDense、 SimulatedAnnealing、 QuantumGuidedGreedy

产出：
- 箱线图（目标值 + 运行时间）
- 按图规模分组的柱状图（带误差棒）
- 汇总统计 CSV

```bash
# 小规模人工数据快速验证
python3 experiments/exp2_algorithm_comparison.py --task maximum_clique --small

# 全部人工数据（耗时较长）
python3 experiments/exp2_algorithm_comparison.py --task maximum_clique --full

# 外部 DIMACS 数据（每实例重复 10 次）
python3 experiments/exp2_algorithm_comparison.py \
    --task maximum_clique --data-source external
```

**外部数据说明**：外部数据来自 DIMACS 最大团基准测试集（brock、C、gen、p-hat 系列，共 10 张图，200~2000 节点）。由于 `answer_nodes` 为空（确切答案未知），评测时将算法找到的团大小与 `best_known_clique_size` 比较。大图上的模拟退火算法耗时较长，建议单独运行。

### 实验三：消融实验（§13.4）

验证量子概率、经典评分、种子扰动各自对性能的贡献。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务类型 | `maximum_clique` |
| `--small` | 仅 n≤50 | 开启 |
| `--full` | 全部人工数据 | — |
| `--csv` | 分析已有 CSV | — |

消融设置：
- A 组：纯经典基线（Q=否, R=是, λ=否）
- B 组：原始 CTQW 概率（Q=是, R=否, λ=否）
- C 组：CTQW + 种子扰动（Q=是, R=否, λ=是）
- D 组：完整算法（Q=是, R=是, λ=是）

产出：
- 四组箱线图对比
- 相对基线提升柱状图
- 汇总统计 CSV

```bash
python3 experiments/exp3_ablation.py --task maximum_clique --small
```

### 实验四：参数敏感性分析（§13.5）

验证算法在不同参数下的稳定性。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务类型 | `maximum_clique` |
| `--instance` | 指定 JSON 文件 | 自动选择第一个小图 |
| `--instance-idx` | 使用第 N 个实例 | 0 |

扫描范围：
- t ∈ {0.5, 1, 2, 5, 10}
- λ ∈ {0, 0.1, 0.5, 1, 2, 5}
- α ∈ {0, 0.25, 0.5, 0.75, 1}

产出：
- t-λ、α-λ、t-α 三张热力图
- α 敏感性曲线（带误差带）
- 完整扫描 CSV

```bash
python3 experiments/exp4_sensitivity.py --task maximum_clique
```

### 通用实验运行器

自由组合数据源、算法、评分方法和参数，用于探索性分析和外部数据测试。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务类型（必选） | — |
| `--data-source` | artificial / external / dir / file（必选） | — |
| `--data-path` | dir/file 模式下的路径 | — |
| `--small` | 仅 artificial + n≤50 | False |
| `--algorithms` | 算法列表（必选，可多选） | — |
| `--scorer` | 量子评分方式: classical / degree / quantum / hybrid | hybrid |
| `--alpha` | 混合权重 α∈[0,1] | 0.5 |
| `--t` | CTQW 演化时间 | 1.0 |
| `--lam` | 种子扰动强度 λ | 0.5 |
| `--init` | 初态方式: uniform / max_degree / random | max_degree |
| `--repeat` | 每实例重复次数（auto: artificial=4, external=10） | auto |
| `--seed` | 基础随机种子 | 0 |
| `--output` | 输出目录 | 自动命名 |
| `--no-plot` | 仅输出 CSV | False |
| `--verbose` | 详细输出 | False |

可选算法：`ClassicalDegree` `ClassicalClique` `ClassicalDense` `SimulatedAnnealing` `QuantumGuidedGreedy`

```bash
# 人工数据小规模实验
python3 experiments/run_experiment.py \
    --task maximum_clique --data-source artificial --small \
    --algorithms ClassicalDegree ClassicalClique SimulatedAnnealing

# 外部数据对比实验
python3 experiments/run_experiment.py \
    --task maximum_clique --data-source external \
    --algorithms ClassicalDegree ClassicalClique SimulatedAnnealing QuantumGuidedGreedy

# 指定目录
python3 experiments/run_experiment.py \
    --task maximum_clique --data-source dir \
    --data-path datasets/data/artificial/maximum_clique/mc_n100_p02_k10 \
    --algorithms ClassicalClique

# 单文件 + 自定义量子参数
python3 experiments/run_experiment.py \
    --task maximum_clique --data-source file \
    --data-path datasets/data/external/maximum_clique/ext_mc_C250-9.json \
    --algorithms QuantumGuidedGreedy --scorer hybrid --alpha 0.7 --t 2.0
```

## 外部数据与人工数据

| 属性 | 人工数据 | 外部数据（DIMACS） |
|------|---------|-------------------|
| 来源 | 植入模型生成 | 公开图算法基准 |
| 图结构 | ER 随机背景 | 真实图结构（社区、power-law 度分布） |
| ground truth | 精确已知（节点级） | 仅知道团大小上界 |
| 实例组织 | 参数组子目录（5 个/组） | 平铺 JSON（1 个/数据集） |
| 评价方式 | 比较解与答案的 recall | 比较团大小与 best_known |
| 重复策略 | 5×4=20 数据点/组 | 1×10=10 数据点/数据集 |
| 用途 | 验证算法假设（已知目标结构） | 验证泛化能力（真实图结构） |

## 在代码中使用

```python
from src import (
    ClassicalGreedy, SimulatedAnnealing,
    CliqueCandidateSet, ClassicalCliqueScorer,
    load_instance, run_single_instance, ExperimentConfig,
)

# 加载测试实例
instance = load_instance(
    "datasets/data/artificial/maximum_clique/"
    "mc_n100_p02_k10/mc_n100_p02_k10_000.json"
)

# 定义算法
algorithms = {
    "CliqueGreedy": ClassicalGreedy(
        CliqueCandidateSet(), ClassicalCliqueScorer()),
}

# 运行实验
config = ExperimentConfig(n=100, p=0.2, k=10, repeat_runs=4)
results = run_single_instance(instance, algorithms, config)
```

## 待完成工作

> **CTQW 演化计算模块**（`scipy.linalg.expm` 实现 `e^{-iHt}`）是当前唯一的阻塞项。
> 该模块完成后，以下所有任务均可立即生效，**无需修改任何其他代码**。

| 优先级 | 任务 | 阻塞项 | 涉及文件 |
|--------|------|--------|---------|
| **P0** | 实现 CTQW 矩阵指数计算 | — | `src/scoring.py` (`QuantumScorer.score_all`) |
| P1 | 补全实验一：真实 CTQW 概率分布可视化 | P0 | `experiments/exp1_ctqw_visualization.py` |
| P1 | 补全实验三：真实消融实验结果 | P0 | `experiments/exp3_ablation.py` |
| P1 | 补全实验四：真实参数敏感性热力图 | P0 | `experiments/exp4_sensitivity.py` |
| P2 | 大规模图的 Krylov 子空间近似方法 | P0 | `src/hamiltonian.py` |
| P2 | 外部数据消融实验（验证泛化能力） | P0 | 在 `experiments/run_experiment.py` 上手动组合 |
| P3 | 实验图表美化（论文出版级质量） | — | 各 `experiments/*.py` 的 `analyze_and_plot` / `plot_*` 函数 |
| P3 | 密集子图任务的外部数据收集与转换 | — | `datasets/data/external/densest_subgraph_raw/` |

## 开发状态

| 模块 | 状态 |
|------|------|
| 数据集生成 | 已完成 |
| 对照实验框架 | 已完成 |
| 经典贪心算法 | 已完成 |
| 模拟退火算法 | 已完成 |
| 量子引导贪心算法框架 | 框架就绪，CTQW 计算模块占位 |
| CTQW 演化计算 | **待实现（P0 阻塞项）** |
| 实验一: CTQW 可视化 | 框架就绪，CTQW 占位 |
| 实验二: 算法对比 | 已完成（人工+外部数据） |
| 实验三: 消融实验 | 框架就绪，CTQW 占位 |
| 实验四: 参数敏感性 | 框架就绪，CTQW 占位 |
| 通用实验运行器 | 已完成（人工+外部数据） |
| `datasets/visualize.py` | 已完成（基础模式 + CTQW 概率着色） |

## 依赖

- Python 3.10+
- NumPy ≥ 1.24 — 矩阵运算
- SciPy ≥ 1.11 — 矩阵指数 `expm`（CTQW 演化）
- NetworkX ≥ 3.1 — 图构建与经典算法
- Matplotlib ≥ 3.7 — 可视化
- pandas — 实验结果管理

## 参考文献

参见 [量子信息理论部分.md](量子信息理论部分.md) 第 13 节的完整文献列表。
