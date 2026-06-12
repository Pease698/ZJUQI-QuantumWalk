"""测试数据可视化脚本 —— 将单个 JSON 测试文件绘制为图。

用法:
    # 基础模式：高亮显示答案子图
    python visualize.py data/maximum_clique/mc_n100_p02_k10/mc_n100_p02_k10_000.json
    python visualize.py <json_path> --save output.png
    python visualize.py <json_path> --no-labels
    python visualize.py <json_path> --figsize 20,15

    # CTQW 概率着色模式：双图对比（答案高亮 + CTQW 概率热力图）
    python visualize.py <json_path> --ctqw
    python visualize.py <json_path> --ctqw --ctqw-t 2.0 --ctqw-lam 0.5
    python visualize.py <json_path> --ctqw --ctqw-init uniform
    python visualize.py <json_path> --ctqw --save ctqw_fig.png --no-labels
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
import numpy as np

# 确保可以从项目根目录导入 src 模块
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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
    node_sizes = _compute_degree_sizes(G)

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
    _set_title(ax, data)

    # 图例
    legend_handles = [
        mpatches.Patch(color="#e74c3c", label=f"答案节点 (k={data['parameters']['answer_size']})"),
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


# ============================================================
# CTQW 概率着色模式
# ============================================================

_CTQW_HEATMAP = plt.cm.YlOrRd       # 浅黄(低概率) → 深红(高概率)
_COLOR_TARGET_OUTLINE = "#e74c3c"   # 答案节点边框色
_COLOR_BG_EDGE = "#e0e0e0"          # 背景边颜色


def visualize_with_ctqw(json_path: str,
                         t: float = 1.0,
                         lam: float = 0.0,
                         init_method: str = "max_degree",
                         save_path: str | None = None,
                         show_labels: bool = True,
                         figsize: tuple = (20, 10)):
    """双图对比模式：左侧答案高亮 + 右侧 CTQW 概率热力图。

    右侧图中，节点颜色深浅表示 CTQW 概率 P_v(t)：
      - 颜色越深（红）= 概率越高
      - 颜色越浅（黄）= 概率越低
    答案节点额外添加边框高亮。

    控制台输出：
      - Ratio = Mean(P_target) / Mean(P_background)
      - 以及目标/背景区域的详细概率统计

    参数:
        json_path: JSON 测试数据文件路径。
        t: CTQW 演化时间。
        lam: 种子扰动强度 λ。
        init_method: 初态方式 "uniform" / "max_degree" / "random"。
        save_path: 保存图片路径。
        show_labels: 是否显示节点标签。
        figsize: 图片尺寸 (宽, 高)。
    """
    data = load_data(json_path)
    G, answer_nodes, answer_edges = build_graph(data)
    n = data["num_nodes"]
    answer_set = answer_nodes

    # ---- 计算 CTQW 概率 ----
    print(f"计算 CTQW 概率分布 (t={t}, λ={lam}, init={init_method})...")
    probs = _compute_ctqw_probs(data, G, t, lam, init_method)

    # ---- 计算 Ratio ----
    target_probs = [float(probs[v]) for v in answer_set]
    bg_probs = [float(probs[v]) for v in range(n) if v not in answer_set]
    target_mean = float(np.mean(target_probs)) if target_probs else 0.0
    bg_mean = float(np.mean(bg_probs)) if bg_probs else 0.0
    ratio = target_mean / bg_mean if bg_mean > 0 else float("inf")

    print(f"  目标节点平均概率: {target_mean:.6f}")
    print(f"  背景节点平均概率: {bg_mean:.6f}")
    print(f"  Ratio = {ratio:.4f}  {'> 1 ✓' if ratio > 1 else '< 1'}")

    # ---- 布局 ----
    pos = nx.spring_layout(G, seed=42, k=3.0 / (G.number_of_nodes() ** 0.5),
                           iterations=50)
    node_sizes = _compute_degree_sizes(G)
    bg_nodes = [v for v in G.nodes() if v not in answer_set]
    ans_nodes_list = list(answer_set)

    # 分离边
    bg_edge_list = []
    ans_edge_list = []
    for u, v in G.edges():
        if (u, v) in answer_edges:
            ans_edge_list.append((u, v))
        else:
            bg_edge_list.append((u, v))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ---- 左图：答案高亮（复用现有逻辑） ----
    ax = axes[0]
    ax.set_axis_off()

    nx.draw_networkx_edges(G, pos, edgelist=bg_edge_list, ax=ax,
                           edge_color="#d0d0d0", alpha=0.5, width=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=ans_edge_list, ax=ax,
                           edge_color="#e74c3c", alpha=0.9, width=2.5)

    if bg_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=bg_nodes, ax=ax,
                               node_color="#a0c4e8",
                               node_size=[node_sizes[v] for v in bg_nodes],
                               edgecolors="#7fa8cc", linewidths=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=ans_nodes_list, ax=ax,
                           node_color="#e74c3c",
                           node_size=[node_sizes[v] for v in ans_nodes_list],
                           edgecolors="#c0392b", linewidths=1.5)

    if show_labels:
        labels = {v: str(v) for v in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7,
                                font_color="#333333")

    _set_title(ax, data, prefix="答案高亮")
    ax.legend(handles=[
        mpatches.Patch(color="#e74c3c", label=f"答案节点 (k={len(answer_set)})"),
        mpatches.Patch(color="#a0c4e8", label="背景节点"),
    ], loc="upper right", fontsize=8, framealpha=0.9)

    # ---- 右图：CTQW 概率热力图 ----
    ax = axes[1]
    ax.set_axis_off()

    # 归一化概率到 [0,1] 用于 colormap
    p_min, p_max = float(probs.min()), float(probs.max())
    if p_max - p_min < 1e-12:
        p_norm = np.zeros_like(probs)
    else:
        p_norm = (probs - p_min) / (p_max - p_min)

    # 背景边统一浅灰
    all_edges = list(G.edges())
    nx.draw_networkx_edges(G, pos, edgelist=all_edges, ax=ax,
                           edge_color=_COLOR_BG_EDGE, alpha=0.4, width=0.5)

    # 所有节点按 CTQW 概率着色
    for node in G.nodes():
        color = _CTQW_HEATMAP(p_norm[node])
        ax.scatter(*pos[node],
                   s=node_sizes[node] * 0.8,
                   c=[color], edgecolors="#888888", linewidths=0.3,
                   zorder=2)

    # 答案节点加红色边框高亮
    if answer_set:
        for v in answer_set:
            if v in pos:
                ax.scatter(*pos[v],
                           s=node_sizes[v] * 0.8,
                           c="none", edgecolors=_COLOR_TARGET_OUTLINE,
                           linewidths=2.5, zorder=3)

    if show_labels and n <= 100:
        for v in G.nodes():
            ax.annotate(str(v), pos[v], fontsize=6,
                        ha="center", va="center", color="#333333",
                        fontweight="bold" if v in answer_set else "normal")

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=_CTQW_HEATMAP,
                                norm=plt.Normalize(vmin=p_min, vmax=p_max))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("CTQW Probability P_v(t)", fontsize=9)

    title = (f"CTQW Probability (t={t}, λ={lam})\n"
             f"Ratio = {ratio:.3f}  |  "
             f"Target mean = {target_mean:.4f}  |  "
             f"Background mean = {bg_mean:.4f}")
    ax.set_title(title, fontsize=10)

    # 整体标题
    params = data["parameters"]
    suptitle_parts = [
        f"CTQW: {data['sample_id']}",
        f"n={data['num_nodes']}  |E|={data['num_edges']}",
        f"t={t}, λ={lam}",
    ]
    fig.suptitle("  |  ".join(suptitle_parts), fontsize=12, fontweight="bold")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存至: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def _compute_ctqw_probs(data: dict, G: nx.Graph,
                         t: float, lam: float,
                         init_method: str) -> np.ndarray:
    """计算所有节点的 CTQW 概率 P_v(t)。

    通过 QuantumScorer 调用 scipy.linalg.expm 计算真实的矩阵指数演化。
    """
    from src.scoring import QuantumScorer
    from src.graph_utils import GraphInstance

    # 构造 GraphInstance（评分器需要的接口）
    edges = [tuple(e) for e in data["edges"]]
    answer_edges = [tuple(e) for e in data.get("answer_edges", [])]

    instance = GraphInstance(
        sample_id=data["sample_id"],
        num_nodes=data["num_nodes"],
        num_edges=data["num_edges"],
        edges=edges,
        task_type=data["task_type"],
        is_artificial=data.get("is_artificial", True),
        answer_nodes=data.get("answer_nodes", []),
        answer_edges=answer_edges,
        parameters=data.get("parameters", {}),
    )

    scorer = QuantumScorer(t=t, lam=lam, init_method=init_method, seed=42)
    candidates = set(range(data["num_nodes"]))
    scores = scorer.score_all(candidates, set(), instance)

    probs = np.zeros(data["num_nodes"], dtype=np.float64)
    for v, p in scores.items():
        probs[v] = float(p)
    return probs


# ============================================================
# 辅助函数
# ============================================================

def _compute_degree_sizes(G: nx.Graph) -> dict:
    """按度等比缩放节点大小。"""
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    min_deg = min(degrees.values()) if degrees else 0
    if max_deg == min_deg:
        return {v: 200 for v in G.nodes()}
    return {v: 80 + 400 * (degrees[v] - min_deg) / (max_deg - min_deg)
            for v in G.nodes()}


def _set_title(ax_or_suptitle, data: dict, prefix: str = ""):
    """设置图表标题，包含实例参数信息。"""
    params = data["parameters"]
    title_parts = [
        f"{prefix}样本: {data['sample_id']}",
        f"n={data['num_nodes']}  |E|={data['num_edges']}",
        f"p_bg={params['bg_edge_prob']}",
        f"k_answer={params['answer_size']}",
    ]
    if data["task_type"] == "densest_subgraph":
        title_parts.append(f"ρ_answer={params['answer_edge_density']:.3f}")
    title = "  |  ".join(title_parts)
    ax_or_suptitle.set_title(title, fontsize=11, pad=12)


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="将测试数据 JSON 文件绘制为图，支持答案高亮和 CTQW 概率着色。")
    parser.add_argument("json_path", help="测试数据 JSON 文件的路径。")
    parser.add_argument("--save", default=None,
                        help="保存图片到指定文件路径，而非交互式显示。")
    parser.add_argument("--no-labels", action="store_true",
                        help="隐藏节点标签（大图或节点密集时建议使用）。")
    parser.add_argument("--figsize", default="14,10",
                        help="图片尺寸，格式为 宽,高（基础模式默认 14,10；"
                             "CTQW 模式默认 20,10）。")

    # CTQW 参数组
    parser.add_argument("--ctqw", action="store_true",
                        help="启用 CTQW 概率着色模式（双图对比布局）。")
    parser.add_argument("--ctqw-t", type=float, default=1.0,
                        help="CTQW 演化时间（默认 1.0）。")
    parser.add_argument("--ctqw-lam", type=float, default=0.0,
                        help="种子扰动强度 λ（默认 0.0）。")
    parser.add_argument("--ctqw-init", type=str, default="max_degree",
                        choices=["uniform", "max_degree", "random"],
                        help="初态初始化方式（默认 max_degree）。")

    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"错误: 文件不存在: {args.json_path}")
        sys.exit(1)

    w, h = map(int, args.figsize.split(","))

    if args.ctqw:
        # CTQW 模式默认更宽的画布
        if args.figsize == "14,10":
            w, h = 20, 10
        visualize_with_ctqw(
            args.json_path,
            t=args.ctqw_t,
            lam=args.ctqw_lam,
            init_method=args.ctqw_init,
            save_path=args.save,
            show_labels=not args.no_labels,
            figsize=(w, h),
        )
    else:
        visualize(args.json_path, save_path=args.save,
                  show_labels=not args.no_labels, figsize=(w, h))


if __name__ == "__main__":
    main()
