# 测试数据集 —— 量子引导贪心算法

## 概述

本目录包含用于图组合优化算法基准测试的人工测试数据，覆盖两类任务：**最大团问题（Maximum Clique, MC）** 和 **密集子图发现问题（Densest Subgraph, DS）**。

所有数据采用植入（planted）模型生成。对于最大团问题，先在背景图中随机选取 k 个节点构成完全子图（团），再按背景边概率 p 生成其余边。对于密集子图问题，先在背景图中随机选取 k 个节点并以概率 ρ 生成内部边，再按背景边概率 p 生成其余边。这样生成的测试数据天然带有已知的答案（ground truth），便于后续实验中的精确评估。

与直接从网上获取的真实网络数据集相比，人工生成数据的优势在于：
- 答案完全已知，无需人工标注或猜测；
- 参数可控，可系统性地改变图规模、噪声强度和目标结构大小；
- 支持统计显著性检验，每组参数可生成多个独立实例。

### 实验策略

**每组参数生成 5 个不同的图实例，实验时每个实例重复运行 4 次**（使用不同随机初始化），合计 5×4=20 个数据点。此设计的优势：

- **区分两类随机性**：不同实例反映图结构的随机性（背景边、植入位置），多次运行反映算法初始化的随机性，二者分开统计可分别计算图间方差和运行间方差
- **满足统计要求**：20 个数据点满足理论文档中"每组至少 20 次重复"的要求
- **控制数据量**：每组仅 5 个 JSON 文件，相比每组 20 个文件减少了 75% 的存储和加载开销

## 目录结构

```
datasets/
├── generators/              # 人工图生成器
│   ├── __init__.py          # 模块导出，汇总所有公开接口
│   ├── base.py              # 公共工具：JSON 保存、样本ID 生成、团/密度验证
│   ├── planted_clique.py    # 植入团生成器（最大团问题）
│   └── planted_dense.py     # 植入密集子图生成器（密集子图问题）
├── data/
│   ├── maximum_clique/      # 最大团测试用例，按参数组合分子目录
│   └── densest_subgraph/    # 密集子图测试用例，按参数组合分子目录
├── generate_all.py          # 一键生成所有测试数据的入口脚本
├── visualize.py             # 可视化脚本：将单个 JSON 绘制为图
└── README.md                # 本说明文档
```

## 统一 JSON 格式

每个测试实例保存为一个独立的 JSON 文件，字段含义如下：

```json
{
  "sample_id": "mc_n100_p02_k10_000",
  "num_nodes": 100,
  "num_edges": 345,
  "edges": [[0, 1], [3, 7], ...],
  "task_type": "maximum_clique",
  "is_artificial": true,
  "parameters": {
    "num_nodes": 100,
    "bg_edge_prob": 0.2,
    "answer_size": 10,
    "answer_edge_density": 1.0
  },
  "answer_nodes": [3, 15, 22, 41, 56, 63, 71, 84, 90, 97],
  "answer_edges": [[3, 15], [3, 22], ...]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_id` | string | 唯一标识符，编码了所有生成参数。命名规则见下文 |
| `num_nodes` | int | 图中节点总数，节点编号从 0 到 num_nodes-1 |
| `num_edges` | int | 图中边总数（含背景边和答案边） |
| `edges` | list[list[int,int]] | 所有边的列表，每条边为 `[u, v]` 且满足 `u < v`（无向简单图） |
| `task_type` | string | 任务类型：`"maximum_clique"` 或 `"densest_subgraph"` |
| `is_artificial` | bool | 是否为人工生成数据。本目录中所有数据均为 `true`，后续加入真实数据集时用于区分 |
| `parameters.num_nodes` | int | 生成参数：节点总数 |
| `parameters.bg_edge_prob` | float | 生成参数：背景边概率 p |
| `parameters.answer_size` | int | 生成参数：植入答案的节点数 k |
| `parameters.answer_edge_density` | float | 生成参数：答案子图的边密度。最大团恒为 1.0；密集子图为实际密度值（可能与目标 ρ 有微小差异） |
| `answer_nodes` | list[int] | 植入答案包含的节点列表（已排序），即 ground truth |
| `answer_edges` | list[list[int,int]] | 植入答案节点之间的边，可以理解为"已知的最优解包含的边" |

### sample_id 命名规则

格式：`<任务前缀>_n<节点数>_p<背景概率>_k<答案大小>[_r<答案密度>]_<实例序号>.json`

- 任务前缀：`mc` 表示最大团，`ds` 表示密集子图
- 背景概率 p：`p01` 表示 0.1，`p005` 表示 0.05（去掉小数点，保留有效数字）
- 答案密度 r（仅密集子图）：`r06` 表示 0.6，`r07` 表示 0.7
- 实例序号：三位零填充，从 `000` 到 `004`（每组 5 个实例）

示例：
- `mc_n100_p02_k10_000.json` —— 最大团，100 节点，p=0.2，k=10，第 0 号实例
- `ds_n50_p01_k8_r06_004.json` —— 密集子图，50 节点，p=0.1，k=8，ρ=0.6，第 4 号实例

## 参数矩阵

### 最大团问题（Maximum Clique）

人工生成参数按实验目的分为四个层级，每级逐步增大图规模和答案挑战。

每组生成 5 个实例，实验时每个实例重复 4 次运行，合计 20 个数据点：

| 层级 | 节点数 n | 背景边概率 p | 植入团大小 k | 每组实例数 | 说明 |
|---|---|---|---|---|---|
| 验证层 | 30 | 0.1, 0.2 | 5, 8 | 5 | 小规模快速验证，用于调试算法正确性和可视化 CTQW 概率分布 |
| 验证层 | 50 | 0.1, 0.2, 0.3 | 5, 8, 10 | 5 | 中等规模验证，覆盖更多背景密度和答案规模组合 |
| 过渡层 | 100 | 0.1, 0.2, 0.3 | 8, 10, 12 | 5 | 向主实验规模过渡，算法行为应在此处与验证层趋势一致 |
| 主实验层 | 150 | 0.1, 0.2, 0.3 | 10, 15 | 5 | 消融实验和参数敏感性扫描的主要数据来源 |
| 主实验层 | 200 | 0.1, 0.2 | 12, 16, 20 | 5 | 较大规模下的算法性能评估，p=0.3 时背景过密故跳过 |
| 压力测试层 | 300 | 0.05, 0.1 | 15, 20, 25 | 5 | 大图压力测试，降低背景概率以保持植入团的可区分性 |
| 压力测试层 | 500 | 0.05, 0.1 | 20, 30 | 5 | 最大规模压力测试，仅用于最优参数下的最终对比，不做全参数扫描 |

**设计考量**：
- 植入团大小 k 需满足 k > 2log₂(n)/log₂(1/p)，确保植入团在统计上可辨识
- 每组 5 个不同图实例 × 4 次重复运行 = 20 个数据点，满足"每组至少 20 次重复"的统计要求
- 区分图间方差（不同实例）和运行间方差（同一实例不同初始化），实验报告中应分别报告

### 密集子图问题（Densest Subgraph）

所有密集子图数据限制在 n ≤ 200，每组 5 个实例，实验时每个实例重复 4 次运行：

| 节点数 n | 背景边概率 p | 答案大小 k | 目标密度 ρ | 每组实例数 |
|---|---|---|---|---|
| 30 | 0.1, 0.2 | 5, 8 | 0.6, 0.7, 0.8 | 5 |
| 50 | 0.1, 0.2 | 5, 8 | 0.6, 0.7, 0.8 | 5 |
| 100 | 0.1, 0.2 | 8, 10 | 0.6, 0.7, 0.8 | 5 |
| 150 | 0.1, 0.2 | 10, 15 | 0.6, 0.7, 0.8 | 5 |
| 200 | 0.1, 0.2 | 12, 16 | 0.6, 0.7, 0.8 | 5 |

**与最大团的区别**：
- 密度 ρ < 1.0 表示植入答案并非完全子图，允许少量缺边，更接近真实网络中社区或功能模块的形态
- 不对 k 做统计可辨识性约束——即使随机背景也可能存在更密集的子图，这正反映了该问题的实际难度
- 每组的实际密度（`answer_edge_density` 字段）可能与目标 ρ 有微小偏差，因为边数是整数值

### 总规模汇总

| 任务 | 参数组合数 | 每组实例数 | JSON 文件总数 |
|---|---|---|---|
| 最大团 | 44 组 | 5 | 220 |
| 密集子图 | 60 组 | 5 | 300 |
| **合计** | **104 组** | — | **520** |

实验时实际数据点：520 个实例 × 4 次重复运行 = 2,080 个数据点。

## 生成算法

### 植入团生成算法（planted clique）

1. 随机选择 k 个节点作为植入团 S_answer
2. 在 S_answer 内部添加所有 k(k-1)/2 条边（确保是合法团）
3. 对于其他节点对（至少一端不在 S_answer 中），以概率 p 添加背景边
4. 验证 S_answer 确为合法团（自动断言检查）

### 植入密集子图生成算法（planted dense subgraph）

1. 随机选择 k 个节点作为植入密集子图 S_answer
2. 计算 S_answer 内部所有可能的 k(k-1)/2 条边
3. 从中随机选取 round(ρ × k(k-1)/2) 条边作为植入边
4. 对于其他节点对，以概率 p 添加背景边
5. 计算并记录 S_answer 的实际密度

### 随机种子管理

所有实例使用确定性随机种子保证可复现性：

```
种子 = 参数组序号 × 100 + 实例序号
```

每个参数组合的组序号互不相同，因此全局无种子冲突。同一组内的 5 个实例使用连续种子（如 0, 1, ..., 4），不同组之间种子至少间隔 100。

## 使用方法

### 生成全部数据

```bash
cd datasets
python generate_all.py              # 生成所有 520 个测试实例（JSON 文件）
python generate_all.py --dry-run     # 仅打印生成计划，不实际生成
```

### 可视化某个实例

将测试数据 JSON 文件绘制为图，答案子图以红色高亮显示：

```bash
# 交互式显示
python visualize.py data/maximum_clique/mc_n100_p02_k10/mc_n100_p02_k10_000.json

# 保存为图片文件
python visualize.py data/densest_subgraph/ds_n50_p01_k8_r06/ds_n50_p01_k8_r06_004.json --save fig.png

# 大图隐藏节点标签，避免画面拥挤
python visualize.py <json_path> --no-labels

# 自定义图片尺寸
python visualize.py <json_path> --figsize 20,15
```

**可视化要素**：
- 浅蓝色节点 = 背景节点，红色节点 = 答案节点
- 浅灰色细边 = 背景边，红色粗边 = 答案边
- 节点大小按度缩放：度数越高，节点越大
- 图例标注答案节点数、背景节点等信息
- 标题包含 sample_id 和所有生成参数

### 在代码中加载

```python
import json

# 加载单个测试实例
with open("data/maximum_clique/mc_n100_p02_k10/mc_n100_p02_k10_000.json") as f:
    data = json.load(f)

edges = data["edges"]              # list[list[int,int]]，所有边
answer = data["answer_nodes"]      # list[int]，答案节点
params = data["parameters"]        # dict，生成参数
task = data["task_type"]           # "maximum_clique" 或 "densest_subgraph"

# 按参数过滤所有实例（示例：遍历某一参数组合的所有文件）
import os, glob
pattern = "data/maximum_clique/mc_n100_p02_k10/*.json"
for filepath in sorted(glob.glob(pattern)):
    with open(filepath) as f:
        instance = json.load(f)
    # 在此处理每个实例

# 实验运行示例：对同一实例多次重复运行
RUNS_PER_INSTANCE = 4
for filepath in sorted(glob.glob(pattern)):
    with open(filepath) as f:
        instance = json.load(f)
    for run in range(RUNS_PER_INSTANCE):
        result = your_algorithm(instance, seed=run)  # 不同随机初始化
        # 记录 result（解质量、运行时间等）
```

## 依赖环境

生成和可视化需要以下 Python 包：

- **Python 3.10+**（`float | None` 类型注解需要）
- `networkx` —— 图结构构建与布局计算
- `matplotlib` —— 图形渲染

安装命令：

```bash
pip install networkx matplotlib
```

仅运行 `generate_all.py` 生成数据（不使用可视化功能）则无需额外依赖，仅需 Python 标准库。
