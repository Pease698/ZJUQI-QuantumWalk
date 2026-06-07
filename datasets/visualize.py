"""测试数据可视化脚本 —— 将单个 JSON 测试文件绘制为图。

用法:
    python visualize.py data/maximum_clique/mc_n100_p02_k10/mc_n100_p02_k10_000.json
    python visualize.py data/densest_subgraph/ds_n50_p01_k8_r06/ds_n50_p01_k8_r06_005.json
    python visualize.py <json_path> --save output.png   # 保存为图片文件
    python visualize.py <json_path> --no-labels         # 隐藏节点标签（大图适用）
    python visualize.py <json_path> --figsize 20,15     # 自定义图片尺寸宽,高
"""

import argparse
import json
import os
import sys

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ---- 跨平台中文字体设置 ----
_CJK_CANDIDATES = [
    # Windows
    'Microsoft YaHei',          # 微软雅黑（Win 默认）
    'SimHei',                   # 黑体
    'SimSun',                   # 宋体
    'KaiTi',                    # 楷体
    # macOS
    'PingFang SC',              # 苹方（macOS 默认简体中文）
    'Heiti SC',                 # 黑体-简
    'STHeiti',                  # 华文黑体
    # Linux (常见发行版)
    'WenQuanYi Micro Hei',     # 文泉驿微米黑（apt: fonts-wqy-microhei）
    'WenQuanYi Zen Hei',       # 文泉驿正黑
    'Noto Sans CJK SC',        # Google Noto（apt: fonts-noto-cjk）
    'Noto Sans SC',            # Google Noto Sans SC
    'Source Han Sans SC',      # 思源黑体（Adobe）
    'AR PL UMing CN',          # AR PL 明体
    # Android / ChromeOS
    'Droid Sans Fallback',
    # fallback
    'sans-serif',
]


def _setup_chinese_fonts() -> str | None:
    """扫描系统可用字体，设置第一个可用的中文字体。

    返回找到的字体名，若未找到则返回 None 并打印安装提示。
    """
    available = {f.name for f in fm.fontManager.ttflist}

    found = []
    for name in _CJK_CANDIDATES:
        if name in available:
            found.append(name)

    if found:
        matplotlib.rcParams['font.sans-serif'] = found
        matplotlib.rcParams['axes.unicode_minus'] = False
        return found[0]

    # 无 CJK 字体：使用默认配置，给出安装提示
    print("警告: 未在系统中找到中文字体，图表中的中文可能无法正常显示。")
    print("  安装中文字体：")
    print("    Ubuntu/Debian:  sudo apt install fonts-wqy-microhei")
    print("    CentOS/RHEL:    sudo yum install wqy-microhei-fonts")
    print("    Arch:            sudo pacman -S wqy-microhei")
    print("    macOS:           无需额外安装（已内置 PingFang SC）")
    print("    Windows:         无需额外安装（已内置微软雅黑）")
    return None


_ = _setup_chinese_fonts()


def load_data(json_path: str) -> dict:
    """从 JSON 文件加载测试数据。"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph(data: dict) -> tuple[nx.Graph, set, set]:
    """从测试数据构造 NetworkX 图对象。

    返回:
        (图对象, 答案节点集合, 答案边集合)
    """
    G = nx.Graph()
    G.add_nodes_from(range(data["num_nodes"]))
    G.add_edges_from(data["edges"])

    answer_nodes = set(data["answer_nodes"])
    answer_edges = set()
    for u, v in data["answer_edges"]:
        answer_edges.add((u, v))
        answer_edges.add((v, u))

    return G, answer_nodes, answer_edges


def visualize(json_path: str, save_path: str | None = None,
              show_labels: bool = True, figsize: tuple = (14, 10)):
    """绘制图结构，高亮显示答案子图。

    答案节点以红色显示，答案边以红色粗线显示；
    背景节点以浅蓝色显示，背景边以浅灰色细线显示；
    节点大小按度等比缩放。
    """
    data = load_data(json_path)
    G, answer_nodes, answer_edges = build_graph(data)

    # 使用弹簧布局，k 控制节点间距
    pos = nx.spring_layout(G, seed=42, k=3.0 / (G.number_of_nodes() ** 0.5),
                           iterations=50)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    # 按度等比缩放节点大小
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    min_deg = min(degrees.values()) if degrees else 0
    if max_deg == min_deg:
        node_sizes = {v: 200 for v in G.nodes()}
    else:
        node_sizes = {v: 80 + 400 * (degrees[v] - min_deg) / (max_deg - min_deg)
                      for v in G.nodes()}

    # 分离答案节点与背景节点
    bg_nodes = [v for v in G.nodes() if v not in answer_nodes]
    ans_nodes_list = list(answer_nodes)

    # 分离答案边与背景边
    bg_edge_list = []
    ans_edge_list = []
    for u, v in G.edges():
        if (u, v) in answer_edges:
            ans_edge_list.append((u, v))
        else:
            bg_edge_list.append((u, v))

    # 绘制背景边（浅灰色细线）
    nx.draw_networkx_edges(G, pos, edgelist=bg_edge_list, ax=ax,
                           edge_color="#d0d0d0", alpha=0.5, width=0.5)

    # 绘制答案边（红色粗线）
    nx.draw_networkx_edges(G, pos, edgelist=ans_edge_list, ax=ax,
                           edge_color="#e74c3c", alpha=0.9, width=2.5)

    # 绘制背景节点（浅蓝色）
    if bg_nodes:
        bg_sizes = [node_sizes[v] for v in bg_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=bg_nodes, ax=ax,
                               node_color="#a0c4e8", node_size=bg_sizes,
                               edgecolors="#7fa8cc", linewidths=0.5)

    # 绘制答案节点（红色）
    ans_sizes = [node_sizes[v] for v in ans_nodes_list]
    nx.draw_networkx_nodes(G, pos, nodelist=ans_nodes_list, ax=ax,
                           node_color="#e74c3c", node_size=ans_sizes,
                           edgecolors="#c0392b", linewidths=1.5)

    # 绘制节点标签
    if show_labels:
        labels = {v: str(v) for v in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7,
                                font_color="#333333")

    # 构造标题
    params = data["parameters"]
    title_parts = [
        f"样本: {data['sample_id']}",
        f"n={data['num_nodes']}  |E|={data['num_edges']}",
        f"p_bg={params['bg_edge_prob']}",
        f"k_answer={params['answer_size']}",
    ]
    if data["task_type"] == "densest_subgraph":
        title_parts.append(f"ρ_answer={params['answer_edge_density']:.3f}")
    title = "  |  ".join(title_parts)

    ax.set_title(title, fontsize=11, pad=12)

    # 图例
    legend_handles = [
        mpatches.Patch(color="#e74c3c", label=f"答案节点 (k={params['answer_size']})"),
        mpatches.Patch(color="#a0c4e8", label="背景节点"),
        plt.Line2D([0], [0], color="#e74c3c", linewidth=2.5, label="答案边"),
        plt.Line2D([0], [0], color="#d0d0d0", linewidth=0.5, label="背景边"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
              framealpha=0.9)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存至: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="将测试数据 JSON 文件绘制为图，高亮显示答案子图。")
    parser.add_argument("json_path", help="测试数据 JSON 文件的路径。")
    parser.add_argument("--save", default=None,
                        help="保存图片到指定文件路径，而非交互式显示。")
    parser.add_argument("--no-labels", action="store_true",
                        help="隐藏节点标签（大图或节点密集时建议使用）。")
    parser.add_argument("--figsize", default="14,10",
                        help="图片尺寸，格式为 宽,高（默认: 14,10）。")
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"错误: 文件不存在: {args.json_path}")
        sys.exit(1)

    w, h = map(int, args.figsize.split(","))
    visualize(args.json_path, save_path=args.save,
              show_labels=not args.no_labels, figsize=(w, h))


if __name__ == "__main__":
    main()
