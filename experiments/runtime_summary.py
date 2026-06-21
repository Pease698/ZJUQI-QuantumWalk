"""
统计实验四各算法在四种图上的平均运行时间。

用法：
    python experiments/runtime_summary.py

输出：
    控制台表格 + results/exp6_runtime_summary.csv
"""
import os
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, "results", "exp6_large_scale")

DATASETS = [
    ("内置式人工图", "exp6_embedded_n300-500"),
    ("内置式外部图", "exp6_embedded_dimacs"),
    ("外置式人工图", "exp6_external_n300-500"),
    ("外置式外部图", "exp6_external_dimacs"),
]

rows = []
for label, subdir in DATASETS:
    path = os.path.join(BASE, subdir, "full_results.csv")
    df = pd.read_csv(path)
    means = df.groupby("display_name")["runtime"].mean()
    for algo, rt in means.items():
        rows.append((algo, label, rt))

summary = pd.DataFrame(rows, columns=["algorithm", "dataset", "mean_runtime_s"])
pivot = summary.pivot_table(
    index="algorithm", columns="dataset", values="mean_runtime_s", aggfunc="first"
)

# 按外置式人工图的 runtime 降序排列
pivot = pivot.sort_values("外置式人工图", ascending=False)

print("各算法平均运行时间 (秒)")
print("=" * 80)
print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))
print()

out = os.path.join(REPO, "results", "exp6_runtime_summary.csv")
pivot.round(2).to_csv(out, encoding="utf-8-sig")
print(f"已保存: {out}")
