"""
生成报告级图表：实验四 §6.3 大规模对比图。

读取 results/exp6_large_scale/ 下 external 模式两类数据集的 full_results.csv，
按起点选择策略汇总平均团大小，输出左右两子图的分组条形图。

用法：
    python experiments/make_report_figures.py

输出：
    report/figures/exp6_large_scale_comparison.png  (300 dpi)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

# ---- 中文字体（Windows: SimHei / Microsoft YaHei 均可）----
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, "results", "exp6_large_scale")
OUT_DIR = os.path.join(REPO, "report", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# 方法显示顺序（display_name -> 中文标签, 颜色, 是否高亮）
METHODS = [
    ("ClassicalClique",    "经典强基线",      "#9e9e9e", False),
    ("MS_Random",          "多起点·随机",     "#bdbdbd", False),
    ("MS_Degree",          "多起点·度数",     "#1f77b4", False),
    ("MS_CTQW_krylov_m30", "多起点·CTQW(Krylov)",   "#ff7f0e", True),
    ("MS_CTQW_cheb_d50",   "多起点·CTQW(Chebyshev)", "#d62728", True),
]

PANELS = [
    ("exp6_external_dimacs",    "(a) DIMACS 真实图 (n 至 1000+)"),
    ("exp6_external_n300-500",  "(b) 人工大图 n=300/500"),
]


def load_means(subdir):
    df = pd.read_csv(os.path.join(BASE, subdir, "full_results.csv"))
    g = df.groupby("display_name")["objective"].mean()
    return g


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

for ax, (subdir, title) in zip(axes, PANELS):
    means = load_means(subdir)
    labels, vals, colors = [], [], []
    for key, zh, color, _hl in METHODS:
        if key in means.index:
            labels.append(zh)
            vals.append(means[key])
            colors.append(color)
    bars = ax.bar(range(len(vals)), vals, color=colors,
                  edgecolor="black", linewidth=0.6, width=0.66)
    # 高亮 CTQW：加粗描边
    for bar, (key, _zh, _c, hl) in zip(bars, [m for m in METHODS if m[0] in means.index]):
        if hl:
            bar.set_linewidth(1.8)
    # 数值标签
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10)
    # 强基线参考线
    if "ClassicalClique" in means.index:
        ax.axhline(means["ClassicalClique"], color="#9e9e9e",
                   linestyle="--", linewidth=1.0, zorder=0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("平均最大团大小", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

fig.suptitle("实验四 §6.3：大规模图上各起点选择策略对比（external 模式）",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])

out = os.path.join(OUT_DIR, "exp6_large_scale_comparison.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved:", out)
