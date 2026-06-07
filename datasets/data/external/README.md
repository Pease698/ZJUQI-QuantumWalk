# 外部数据集

## 说明

本目录存放从网络获取的真实数据集，经转换后采用与人工数据集相同的统一 JSON 格式。

## 目录结构

```
external/
├── maximum_clique/          # 转换后的最大团 JSON（供算法直接读取）
├── densest_subgraph/        # 转换后的密集子图 JSON
├── maximum_clique_raw/      # 最大团原始数据（从 DIMACS 下载的 .mtx）
└── README.md
```

## 数据复用策略

DIMACS 数据集目前仅下载了最大团测试图，但其中的植入团（密度 1.0）同时也是密度最高的密集子图，因此同一份 DIMACS 数据可以同时用于两类任务的评测：

- `maximum_clique/` 中的 `ext_mc_*.json` 和 `densest_subgraph/` 中的 `ext_ds_*.json` 的图结构完全相同
- 两者由 `convert_dimacs.py` 默认同时生成，无需手动复制

## JSON 格式要求

外部数据集 JSON 文件必须与人工数据集遵循相同的格式规范。与人工数据的主要区别：

| 字段 | 外部数据取值 |
|---|---|
| `is_artificial` | `false` |
| `answer_nodes` | `[]`（确切答案未知） |
| `answer_edges` | `[]`（确切答案未知） |
| `parameters.source` | 数据来源，如 `"DIMACS"` |
| `parameters.dataset` | 原始数据集名称 |
| `parameters.best_known_clique_size` | 目前已知最佳团大小（int 或 null） |

**最大团示例**（转换自 DIMACS）：

```json
{
  "sample_id": "ext_mc_brock400-4",
  "num_nodes": 400,
  "num_edges": 59765,
  "edges": [[0, 1], [0, 2], ...],
  "task_type": "maximum_clique",
  "is_artificial": false,
  "parameters": {
    "source": "DIMACS",
    "dataset": "brock400-4",
    "best_known_clique_size": 22
  },
  "answer_nodes": [],
  "answer_edges": []
}
```

**密集子图示例**（与上例共享图结构，仅 `sample_id` 和 `task_type` 不同）：

```json
{
  "sample_id": "ext_ds_brock400-4",
  "num_nodes": 400,
  "num_edges": 59765,
  "edges": [[0, 1], [0, 2], ...],
  "task_type": "densest_subgraph",
  "is_artificial": false,
  "parameters": {
    "source": "DIMACS",
    "dataset": "brock400-4",
    "best_known_clique_size": 22
  },
  "answer_nodes": [],
  "answer_edges": []
}
```

## 算法评测说明

由于外部数据不存在标准答案（`answer_nodes` 为空），评测策略与人工数据不同：

- **有答案（人工数据）**：验证算法找到的解是否包含或超过植入答案，可计算精确召回率
- **无答案（外部数据）**：将算法输出与 `best_known_clique_size` 比较，报告是否达到或超过已知最佳值

## 目前已下载的数据集

DIMACS 最大团数据位于 `maximum_clique_raw/`，共 10 个测试图（brock 系列、C 系列、gen 系列、p-hat 系列），全部同时用于 MC 和 DS 评测。

## 原始数据目录

`maximum_clique_raw/` 存放从网络下载的原始格式数据（.mtx、.graph 等）。这些文件**不直接用于实验**，需先通过 `converters/` 下的脚本转为统一 JSON。

## 数据转换脚本

转换脚本位于 `datasets/converters/`：

```bash
cd datasets

# 默认: 同时输出 MC 和 DS 两份 JSON
python -m converters.convert_dimacs

# 仅输出最大团
python -m converters.convert_dimacs --task-type maximum_clique

# 仅输出密集子图
python -m converters.convert_dimacs --task-type densest_subgraph

# 预览
python -m converters.convert_dimacs --dry-run
```
