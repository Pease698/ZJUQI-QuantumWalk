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
│   ├── ctqw_evolution.py        #   CTQW 演化计算（exact / Krylov / Chebyshev）
│   ├── timeout.py               #   超时控制（exp6 使用，multiprocessing 隔离）
│   ├── algorithms/              #   算法实现
│   │   ├── base.py              #     算法基类 + 结果数据结构
│   │   ├── classical_greedy.py  #     经典贪心算法
│   │   ├── simulated_annealing.py #   模拟退火算法
│   │   ├── quantum_greedy.py    #     量子引导贪心（嵌入式方案）
│   │   └── multi_start_ctqw.py  #     Multi-Start CTQW（外部起点选择器）
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
│   ├── visualize.py             #   可视化脚本（支持 CTQW 概率着色模式）
│   └── README.md                #   数据集详细文档
│
├── experiments/                 # 实验脚本（理论 §13 + 新增实验五/六）
│   ├── exp1_ctqw_visualization.py    #   实验一：CTQW 概率分布可视化
│   ├── exp2_algorithm_comparison.py  #   实验二：与经典贪心算法对比（支持外部数据）
│   ├── exp3_ablation.py              #   实验三：消融实验
│   ├── exp4_sensitivity.py           #   实验四：参数敏感性分析
│   ├── exp5_multi_start.py           #   实验五：Multi-Start CTQW 种子选择（H4 验证）
│   ├── exp6_pre_tune.py              #   实验六前置：Krylov m / Chebyshev d 参数调优
│   ├── exp6_large_scale_approx.py    #   实验六：大规模图 CTQW 近似方法对比
│   ├── tune_quantum_params.py        #   量子参数调优（init × α 扫描）
│   ├── d2_mc_breakdown.py            #   细分指标分析（按 n/p/k 分组深挖）
│   └── run_experiment.py             #   通用实验运行器（自由组合）
├── results/                     # 实验结果输出（CSV + 图表）
├── report/                      # 实验报告与演示材料
│   ├── 实验报告.md               #   正式实验报告
│   ├── 汇报讲稿.md               #   汇报讲稿
│   └── ppt/                     #   Beamer 幻灯片
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

实验脚本位于 `experiments/` 目录下，包含理论文档 §13 的四组固定实验、新发现的 Multi-Start 方案（实验五），以及大规模图近似方法对比（实验六）：

| 脚本 | 理论依据 | 目的 | 适用数据 | CTQW 方法 |
|------|---------|------|---------|-----------|
| `exp1_ctqw_visualization.py` | §13.2 | CTQW 概率分布可视化 | 仅人工 | exact |
| `exp2_algorithm_comparison.py` | §13.3 | 算法对比 ★ 核心实验 | **人工 + 外部** | auto（自动选择） |
| `exp3_ablation.py` | §13.4 | 消融实验 | 仅人工 | auto |
| `exp4_sensitivity.py` | §13.5 | 参数敏感性分析 | 仅人工 | auto |
| `exp5_multi_start.py` | 新增 H4 | Multi-Start CTQW 种子选择 | 仅人工 | auto |
| `exp6_pre_tune.py` | 新增 | Krylov m / Chebyshev d 参数调优 | n=200+500 单图 | exact + krylov + chebyshev |
| `exp6_large_scale_approx.py` | 新增 | 大规模图近似方法对比 | **人工(n≥100) + DIMACS** | exact / krylov / chebyshev |
| `tune_quantum_params.py` | — | init × α 参数扫描调优 | 仅人工 | auto |
| `d2_mc_breakdown.py` | — | 按 n/p/k 分组细分分析 | 读取 CSV | 无 |
| `run_experiment.py` | 通用 | 自由组合数据/算法/参数 | **人工 + 外部** | auto |

实验脚本的重复策略：
- **人工数据（小规模）**：每组 5 个实例 × 每实例 4 次 = 20 数据点（符合 §13.1）
- **人工数据（大规模 exp6）**：每组 5 个实例 × 每实例 2 次 = 10 数据点（控制总耗时）
- **外部数据（小规模）**：每数据集 1 个实例 × 每实例 10 次 = 10 数据点
- **外部数据（大规模 exp6）**：每数据集 1 个实例 × 每实例 3 次 = 3 数据点

> **关于实验图表**：`experiments/` 目录下各实验脚本中的图表生成逻辑为初步版本，
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

### 实验五：Multi-Start CTQW 种子选择（新增 H4 验证）

基于实验二的机理诊断——嵌入式 CTQW 在贪心内部与经典 R 评分方向重合但精度更低——将 CTQW 外置于贪心算法之前，用全图均匀叠加初态做起点选择器。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--smoke` | 单实例烟雾测试 | — |
| `--small` | n≤50 全跑 + K 扫描 | — |
| `--medium --K <N>` | n=60~200 跑指定 K | — |
| `--csv` | 直接分析已有 CSV | — |

K 扫描范围：{1, 3, 5, 10}

对照算法（4 个，逐一剥离贡献源）：
1. ClassicalClique — 强经典基线
2. MultiStartRandom(K) — 随机起点；剥离"多起点本身就涨"
3. MultiStartDegree(K) — 度数 Top-K 起点；剥离"任何全局信号都行"
4. MultiStartCTQW(K) — 本方案

诊断规则：
- CTQW > Random → CTQW 起点选择有效
- CTQW > Degree → CTQW 提供了度数之外的额外信息（H4 强成立）

产出：K 扫描汇总 CSV、Wilcoxon 配对检验表

```bash
python3 experiments/exp5_multi_start.py --smoke             # 秒级烟雾测试
python3 experiments/exp5_multi_start.py --small             # K 全扫描 ~6s
python3 experiments/exp5_multi_start.py --medium --K 10     # medium 最优 K ~89s
```

### 实验六前置：Krylov m / Chebyshev d 参数调优

在正式大规模实验之前，对比不同参数值的精度与效率，确定推荐值。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--smoke` | 烟雾测试：3 个参数值 + 1 次重复 | — |
| `--timeout` | 单次计算超时门限（秒） | 120 |

扫描范围：
- Krylov: m ∈ {20, 30, 40, 50, 60, 80, 100, 150}
- Chebyshev: d ∈ {20, 30, 40, 50, 60, 80, 100, 150}

测试图：n=200（精确 expm 对照）+ n=500（时间 scaling）

产出：
- 收敛曲线三子图（L2 误差、运行时间、Top-10 一致性）
- 推荐参数文本 `recommended_params.txt`
- 完整扫描 CSV

**评价指标说明**：

| 指标 | 公式 | 含义 | 判断标准 |
|------|------|------|---------|
| $\epsilon_2$ (L2 误差) | $\|P_{\text{approx}} - P_{\text{exact}}\|_2 = \sqrt{\sum_v (P_v^{\text{approx}} - P_v^{\text{exact}})^2}$ | 近似概率向量与精确结果的**整体欧氏距离**。对全部 $n$ 个节点的误差做平方和再开方，反映概率分布的全局偏离程度 | $\epsilon_2 < 10^{-6}$ 视为收敛 |
| $\epsilon_\infty$ (L∞ 误差) | $\max_v \|P_v^{\text{approx}} - P_v^{\text{exact}}\|$ | 近似概率向量与精确结果在**单个节点上的最大偏差**。最坏情况下的逐节点概率误差，用于检查是否存在局部异常偏离 | $\epsilon_\infty < 10^{-6}$ 视为收敛 |
| Top-10 一致性 | $\frac{\|\text{Top10}_{\text{approx}} \cap \text{Top10}_{\text{exact}}\|}{10}$ | 按概率降序排列的**前 10 个节点中，近似与精确结果重叠的比例**。1.0 表示完全一致，0.3 表示仅 3 个节点重合。直接衡量近似是否改变了概率排名的头部结构 | =1 为完美匹配 |
| 运行时间 (rt) | 秒 | n=200 上单次 CTQW 演化的 wall time，用于比较不同参数的成本；n=500 上仅测时间 scaling，不与精确 expm 对比（n=500 时 expm 已不可行） | — |

**四个指标之间的关系**：

- $\epsilon_2$ 和 $\epsilon_\infty$ 通常高度相关，但 $\epsilon_2$ 更能容忍"大量节点有微小误差"的场景（求平方和均值后影响分散），$\epsilon_\infty$ 会暴露个别节点的异常偏离
- Top-10 一致性是最实用的指标——Multi-Start CTQW 的核心操作是**对概率做 Top-K 选择**，Top-10 能否正确匹配直接决定起点选择是否正确
- 运行时间随 $m$ / $d$ 近似线性增长，最佳参数是**精度刚好收敛（$\epsilon_2 < 10^{-6}$ 且 Top-10=1.0）时的最小 $m$ / $d$**

**推荐参数选择逻辑**：在满足 $\epsilon_2^{\max} < 10^{-6}$ 的参数值中取最小值，这样既保证精度又最小化计算开销。

```bash
python3 experiments/exp6_pre_tune.py --smoke           # 10s 烟雾测试
python3 experiments/exp6_pre_tune.py                   # 完整扫描（约 5 分钟）
python3 experiments/exp6_pre_tune.py --timeout 120     # 自定义超时
```

### 实验六：大规模图 CTQW 近似方法对比

分两部分独立运行，使用 `--timeout` 控制单次运行上限。

数据范围（写死在脚本常量中）：
- **人工数据**：仅 n ∈ {300, 500}（10 个参数组，50 个实例）
- **DIMACS 外部数据**：5 个指定数据集（gen200, C250-9, p-hat300-3, C1000-9, C2000-9）

演化方法（仅两种近似）：
- `krylov_m30`：Krylov 子空间 m=30
- `cheb_d50`：Chebyshev 多项式 d=50

**Part A — 嵌入式方案**（`--mode embedded`）：

| 算法 | 演化方法 | 数量 |
|------|---------|------|
| ClassicalDegree | — | 1 |
| ClassicalClique | — | 1（强基线） |
| SimulatedAnnealing | — | 1 |
| QuantumGuidedGreedy × krylov_m30 | Krylov m=30 | 1 |
| QuantumGuidedGreedy × cheb_d50 | Chebyshev d=50 | 1 |
| **每实例合计** | | **5 算法** |

**Part B — 外置式方案**（`--mode external`，K ∈ {5,10,20,30}）：

| 算法 | K 值 | 演化方法 | 数量 |
|------|------|---------|------|
| ClassicalClique | — | — | 1 |
| MS_Random × K | 5,10,20,30 | — | 4 |
| MS_Degree × K | 5,10,20,30 | — | 4 |
| MS_CTQW × K × krylov_m30 | 5,10,20,30 | Krylov m=30 | 4 |
| MS_CTQW × K × cheb_d50 | 5,10,20,30 | Chebyshev d=50 | 4 |
| **每实例合计** | | | **17 算法** |

**运行次数**：人工数据每实例 2 次，DIMACS 每实例 3 次。

**运行总量**：

| 场景 | 实例 | 算法/实例 | 重复 | 总运行 |
|---|---|---|---|---|
| Part A artificial | 50 | 5 | 2 | **500** |
| Part A dimacs | 5 | 5 | 3 | **75** |
| Part B artificial | 50 | 17 | 2 | **1,700** |
| Part B dimacs | 5 | 17 | 3 | **255** |
| **合计** | | | | **2,530** |

```bash
# 烟雾测试（串行模式，方便调试）
python3 experiments/exp6_large_scale_approx.py --mode embedded --smoke
python3 experiments/exp6_large_scale_approx.py --mode external --smoke

# Part A — 嵌入式方案（默认 5 线程并行）
python3 experiments/exp6_large_scale_approx.py --mode embedded

# Part A — 嵌入式方案（DIMACS，降为 3 线程以防大图 OOM）
python3 experiments/exp6_large_scale_approx.py --mode embedded --data-source dimacs --workers 3

# Part B — 外置方案（人工数据）
python3 experiments/exp6_large_scale_approx.py --mode external

# Part B — 外置方案（DIMACS，减少 K + 降线程）
python3 experiments/exp6_large_scale_approx.py \
    --mode external --data-source dimacs --K-values 10 20 --workers 3

# 串行模式（调试/低内存环境）
python3 experiments/exp6_large_scale_approx.py --mode embedded --workers 0

# 自定义超时和重复次数
python3 experiments/exp6_large_scale_approx.py \
    --mode embedded --timeout 600 --repeat 3
```

**并行执行说明**：

默认使用 `ProcessPoolExecutor`（最多 5 个 worker，`--workers` 可调整）。进程结构为两层：

```
主进程
  ├── Pool Worker 1 ──→ run_with_timeout() ──→ 孙进程 (algo.solve)
  ├── Pool Worker 2 ──→ run_with_timeout() ──→ 孙进程 (algo.solve)
  └── ...
```

- **内层**（`run_with_timeout`）：孙进程隔离执行，超时由 Worker 监控并安全 kill，不影响其他 Worker
- **外层**（`ProcessPoolExecutor`）：`spawn` 上下文，避免 fork 的 numpy 状态问题
- **大图保护**：n > `--serial-threshold`（默认 1000）的实例自动降为串行，防止多进程同时加载大型邻接矩阵导致 OOM
- **烟雾测试**（`--smoke`）始终串行运行
- `--workers 0` 回退为完全串行模式

### 量子参数调优脚本

扫描 `init_method × α` 组合（固定 t=1.0, λ=0.5），用于实验二的参数调优前置。

```bash
python3 experiments/tune_quantum_params.py --task maximum_clique
python3 experiments/tune_quantum_params.py --task maximum_clique --quick  # 仅 α∈{0,0.5,1.0}
```

### 细分指标分析脚本

读取已有实验 CSV，按 (n, p, k) 分组深挖：子组均值对比、胜率配对、recall/成功率、背景密度分组。

```bash
python3 experiments/d2_mc_breakdown.py
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
    ClassicalGreedy, SimulatedAnnealing, QuantumGuidedGreedy,
    CliqueCandidateSet, DenseCandidateSet,
    ClassicalCliqueScorer, ClassicalDenseScorer, ClassicalDegreeScorer,
    load_instance, run_single_instance, ExperimentConfig,
)
from src.algorithms.multi_start_ctqw import (
    MultiStartCTQWGreedy, MultiStartRandomGreedy, MultiStartDegreeGreedy,
)

# 加载测试实例
instance = load_instance(
    "datasets/data/artificial/maximum_clique/"
    "mc_n100_p02_k10/mc_n100_p02_k10_000.json"
)

# 定义算法集合 —— 支持多种 CTQW 演化方法
algorithms = {
    "DegreeGreedy": ClassicalGreedy(
        CliqueCandidateSet(), ClassicalDegreeScorer()),
    "CliqueGreedy": ClassicalGreedy(
        CliqueCandidateSet(), ClassicalCliqueScorer()),
    "SimulatedAnnealing": SimulatedAnnealing(
        CliqueCandidateSet(), max_iterations=2000),
    # 量子引导贪心 —— 自动选择演化方法（n≤200 用 exact，否则用 krylov）
    "QuantumGreedy": QuantumGuidedGreedy(
        CliqueCandidateSet(), t=1.0, lam=0.5, alpha=0.5),
    # 指定 Krylov 子空间维数 m=60
    "QuantumGreedy_Krylov60": QuantumGuidedGreedy(
        CliqueCandidateSet(), t=1.0, lam=0.5, alpha=0.5,
        evolution_method="krylov", krylov_dim=60),
    # Multi-Start CTQW 外置起点选择
    "MultiStartCTQW": MultiStartCTQWGreedy(K=5, t=1.0),
    # 大图场景：Multi-Start + Krylov 近似
    "MultiStartCTQW_Krylov": MultiStartCTQWGreedy(
        K=10, t=1.0, evolution_method="krylov", krylov_dim=60),
}

# 运行实验
config = ExperimentConfig(n=100, p=0.2, k=10, repeat_runs=4)
results = run_single_instance(instance, algorithms, config)
for row in results:
    print(f"{row['algorithm']}: obj={row['objective']}, "
          f"time={row['runtime']:.3f}s")
```

## 待完成工作

CTQW 矩阵指数计算（P0）和 Krylov/Chebyshev 大规模近似方法已完成。以下为后续可探索方向：

### 短期优化（改动局部，风险低）

| 任务 | 涉及文件 | 说明 |
|------|---------|------|
| 去均匀基线的 Q 评分 | `src/scoring.py` | `P_v(t) → P_v(t) − 1/n`，提升概率对比度 |
| Q 作为二级排序（打破 R 平局） | `src/algorithms/quantum_greedy.py` | 先按 R 排，R 相同时用 Q 决胜 |
| 多时间平均 Q 评分 | `src/scoring.py` | 多 t 值概率求平均，抹平单 t 相位抖动 |
| 解耦初态（每轮全图均匀叠加 + λ 传 S 信息） | `src/algorithms/quantum_greedy.py` | 将实验一有效的全局初态搬进嵌入式贪心 |

### 中长期探索（涉及新实验或对外部数据的扩展）

| 任务 | 涉及文件 | 说明 |
|------|---------|------|
| 运行 exp6 前置实验 + 大规模对比 | `experiments/exp6_*.py` | 确定最优 Krylov/Chebyshev 参数并验证大图性能 |
| DIMACS 真实图族 CTQW vs Degree 解耦实验 | `experiments/exp5_multi_start.py` | 在团节点不一定高度数的图族上验证 CTQW 独立信息 |
| 密集子图任务外部数据收集与转换 | `datasets/converters/` | — |
| 实验图表美化（论文出版级质量） | 各 `experiments/*.py` 绘图函数 | — |
| DS 任务完整实验（small/medium 全规模） | `experiments/exp2~exp4` | 目前主要聚焦 MC 任务 |

## 开发状态

| 模块 | 状态 |
|------|------|
| 数据集生成 | 已完成 |
| 对照实验框架 | 已完成 |
| 经典贪心算法 | 已完成（支持 start_node 种子注入） |
| 模拟退火算法 | 已完成 |
| 量子引导贪心算法（嵌入式） | 已完成（基于 scipy.linalg.expm） |
| Multi-Start CTQW（外置起点选择器） | 已完成 |
| CTQW 演化方法：exact / Krylov / Chebyshev | 已完成（`src/ctqw_evolution.py`） |
| 算法运行超时控制 | 已完成（`src/timeout.py`，exp6 使用） |
| CTQW 冒烟测试 | 已完成（6/6 项数值检查通过） |
| 实验一：CTQW 可视化 | 已完成（Ratio=7.08） |
| 实验二：算法对比 | 已完成（人工 + 外部数据） |
| 实验三：消融实验 | 已完成 |
| 实验四：参数敏感性 | 已完成 |
| 实验五：Multi-Start | 已完成（首次 p<0.001 显著超越强基线） |
| 实验六前置：参数调优 | 脚本就绪，待运行 |
| 实验六：大规模近似对比 | 脚本就绪（支持 5 线程并行 + 超时），待运行 |
| 通用实验运行器 | 已完成（人工 + 外部数据） |
| `datasets/visualize.py` | 已完成（基础模式 + CTQW 概率着色） |

## 核心发现

本项目累计运行约 **18,300 次**算法实验（≈40 分钟计算），围绕三组实验得出以下结论：

| 实验 | 验证内容 | 结论 | 关键证据 |
|---|---|---|---|
| **实验一** | CTQW 能否识别图的全局团结构？ | **✓ 成立** | 团节点平均概率 / 背景 = 7.08 |
| **实验二** | 嵌入式 CTQW 评分是否更稳健？ | **⚠ 部分成立** | 混合 > 纯量子 (+43%)，但混合 < 纯经典 (−6.8%) |
| **实验三** | 外置 CTQW 起点选择器能否超越强基线？ | **✓ 成立** | small Δ=+0.12, medium Δ=+0.71，均 p<0.001 |

**最关键发现**：CTQW 的有效性高度依赖于使用方式。

- **外置于贪心之前**（全图均匀叠加初态）→ 统计显著超越 ClassicalClique 强基线（实验三）
- **嵌入于贪心内部**（种子集合均匀叠加初态）→ 与经典 R 评分方向重合且精度更低（实验二）

两种用法对应 CTQW 的两种物理模式：**全局拓扑识别** vs **局部种子扩散**。原始理论文档 §6 的嵌入式设计未区分这一差异，实验三的外置方案是项目发现的最有效用法。

## 依赖

- Python 3.10+
- NumPy ≥ 1.24 — 矩阵运算
- SciPy ≥ 1.11 — 矩阵指数 `expm`（CTQW 演化）
- NetworkX ≥ 3.1 — 图构建与经典算法
- Matplotlib ≥ 3.7 — 可视化
- pandas — 实验结果管理

## 参考文献

理论背景与算法设计参见 [量子信息理论部分.md](量子信息理论部分.md) 第 13 节。

实验过程与结果分析参见 [report/实验报告.md](report/实验报告.md)。
