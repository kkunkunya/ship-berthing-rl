"""
Paper-style baseline comparison figure (优化版)

3个面板：
- A: 成功率对比（带95%置信区间）
- B: 最近距离分布（箱线图+散点）
- C: 最佳尝试距离曲线

Input: 3个评估JSON文件
Output: 高质量PNG/PDF图表
Pos: 基线对比可视化系统

优化改进：
1. B面板设置合理的y轴范围，避免数据点超出
2. C面板显示完整时间范围
3. 图例位置优化，避免遮挡数据
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SKILL_ROOT = Path(
    "C:/Users/sxk27/.claude/plugins/marketplaces/claude-scientific-skills/"
    "scientific-skills/scientific-visualization"
)
warnings.filterwarnings("ignore", message="The figure layout has changed to tight")
try:
    sys.path.insert(0, str(SKILL_ROOT / "scripts"))
    from style_presets import apply_publication_style, set_color_palette
    from figure_export import save_publication_figure
    HAS_SCI_STYLE = True
except Exception:
    HAS_SCI_STYLE = False


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _success_ci(success_rate, n):
    if n <= 0:
        return 0.0
    return 1.96 * np.sqrt(success_rate * (1 - success_rate) / n)


def main():
    parser = argparse.ArgumentParser(description="Paper-style baseline comparison figure")
    parser.add_argument("--sac", type=str, required=True)
    parser.add_argument("--ppo", type=str, required=True)
    parser.add_argument("--td3", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="assets/results/demo_delivery/figures/baseline_comparison_paper",
    )

    args = parser.parse_args()

    if HAS_SCI_STYLE:
        apply_publication_style("nature")
        set_color_palette("okabe_ito")

    data = {
        "SAC": _load_json(args.sac),
        "PPO": _load_json(args.ppo),
        "TD3": _load_json(args.td3),
    }

    labels = ["SAC", "PPO", "TD3"]
    colors = ["#0072B2", "#E69F00", "#D55E00"]
    success_rates = [data[name]["success_rate"] for name in labels]
    episodes = [data[name]["episodes"] for name in labels]
    ci = [_success_ci(rate, n) for rate, n in zip(success_rates, episodes)]

    # 增加图表高度，留出更多空间
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=False)
    rng = np.random.default_rng(42)

    # ========== Panel A: Success rate with 95% CI ==========
    ax = axes[0]
    x = np.arange(len(labels))
    bars = ax.bar(x, success_rates, yerr=ci, capsize=5, color=colors,
                  edgecolor='black', linewidth=1.5)

    # 在柱子上方添加数值标签
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + ci[i] + 0.02,
                f'{rate*100:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylabel("Success Rate", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.15)  # 留出空间显示标签
    ax.set_title("A  Success Rate (95% CI)", fontsize=13, fontweight='bold', pad=10)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=11)

    # ========== Panel B: Final distance distribution ==========
    ax = axes[1]
    min_distances = [np.array([r["min_distance"] for r in data[name]["episode_records"]]) for name in labels]

    # 计算合适的y轴范围
    all_dist = np.concatenate(min_distances)
    y_max = max(all_dist) * 1.15

    # 绘制箱线图
    bp = ax.boxplot(min_distances, positions=x, widths=0.5, patch_artist=True, showfliers=False)

    # 设置箱线图颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='black', linewidth=1.5)

    # 添加散点
    for idx, (dist, color) in enumerate(zip(min_distances, colors)):
        jitter = (rng.random(len(dist)) - 0.5) * 0.2
        ax.scatter(np.full_like(dist, idx, dtype=float) + jitter, dist,
                   s=25, alpha=0.6, color=color, edgecolors='black', linewidths=0.5, zorder=5)

    # 成功阈值线
    pos_tolerance = data["SAC"]["tolerances"]["position"]
    ax.axhline(pos_tolerance, color="black", linestyle="--", linewidth=2,
               label=f"Success Threshold ({pos_tolerance}m)", zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylabel("Minimum Distance to Goal (m)", fontsize=12, fontweight='bold')
    ax.set_ylim(0, y_max)
    ax.set_title("B  Closest Approach Distribution", fontsize=13, fontweight='bold', pad=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.tick_params(labelsize=11)

    # 添加统计信息
    for idx, (name, dist) in enumerate(zip(labels, min_distances)):
        mean_val = np.mean(dist)
        ax.annotate(f'Mean: {mean_val:.1f}m', xy=(idx, mean_val),
                    xytext=(idx + 0.3, mean_val + y_max * 0.05),
                    fontsize=9, ha='left', color=colors[idx],
                    arrowprops=dict(arrowstyle='->', color=colors[idx], lw=1))

    # ========== Panel C: Best attempt distance trace ==========
    ax = axes[2]

    # 计算最大时间跨度
    max_time = 0
    traces = {}
    for name in labels:
        best = data[name].get("best_attempt")
        if best:
            dist = np.array(best["distance_trace"])
            dt = best.get("dt", 0.1)
            t = np.arange(dist.shape[0]) * dt
            traces[name] = (t, dist)
            max_time = max(max_time, t[-1])

    # 绘制曲线
    for name, color in zip(labels, colors):
        if name in traces:
            t, dist = traces[name]
            ax.plot(t, dist, label=name, color=color, linewidth=2.5, zorder=3)
            # 标记终点
            ax.scatter(t[-1], dist[-1], color=color, s=80, marker='o',
                       edgecolors='black', linewidths=1.5, zorder=5)
            # 在终点添加标签
            ax.annotate(f'{name}: {dist[-1]:.1f}m', xy=(t[-1], dist[-1]),
                        xytext=(t[-1] - max_time*0.15, dist[-1] + 5),
                        fontsize=10, ha='right', color=color, fontweight='bold')

    # 成功阈值线
    ax.axhline(pos_tolerance, color="black", linestyle="--", linewidth=2,
               label=f"Success Threshold ({pos_tolerance}m)", zorder=10)

    ax.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Distance to Goal (m)", fontsize=12, fontweight='bold')
    ax.set_xlim(0, max_time * 1.05)

    # 设置y轴范围
    all_trace_dist = np.concatenate([traces[name][1] for name in traces])
    ax.set_ylim(0, max(all_trace_dist) * 1.1)

    ax.set_title("C  Best Attempt Distance Trace", fontsize=13, fontweight='bold', pad=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.tick_params(labelsize=11)

    # 调整布局
    fig.tight_layout(pad=2.0, h_pad=2.5)

    # 保存图表
    output_base = Path(args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    if HAS_SCI_STYLE:
        save_publication_figure(fig, output_base, formats=["pdf", "png"], dpi=600)
        print(f"图表已保存（科学出版物格式）：{output_base}.pdf 和 {output_base}.png")
    else:
        fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches='tight')
        print(f"图表已保存：{output_base}.png")

    plt.close(fig)

    # 打印摘要
    print("\n" + "="*50)
    print("性能摘要")
    print("="*50)
    for name in labels:
        print(f"{name}: 成功率={data[name]['success_rate']*100:.0f}%, "
              f"平均最小距离={np.mean(min_distances[labels.index(name)]):.2f}m")
    print("="*50)


if __name__ == "__main__":
    main()
