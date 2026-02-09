"""
统计分布对比图：SAC vs PPO vs TD3（优化版）

从多个维度展示算法性能差异：
- 最小距离分布（对数尺度）
- 最终速度分布
- 最终航向误差分布
- 完成步数分布

Input: 3个评估JSON文件（sac_evaluation.json, ppo_evaluation.json, td3_evaluation.json）
Output: 高质量PNG图表（300 dpi）
Pos: 基线对比可视化系统

优化改进：
1. 显著性标记位置自动适应坐标范围
2. 图例位置优化，避免遮挡
3. 添加更多边距防止元素溢出
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# 尝试加载科学出版物样式
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
    """加载JSON文件"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _compute_significance(data1, data2):
    """
    计算两组数据的统计显著性（Mann-Whitney U检验）

    返回：
    - p值
    - 显著性标记（*** p<0.001, ** p<0.01, * p<0.05, ns 不显著）
    """
    if len(data1) == 0 or len(data2) == 0:
        return 1.0, "ns"

    # 使用 Mann-Whitney U 检验（非参数检验，适用于非正态分布）
    _, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')

    if p_value < 0.001:
        return p_value, "***"
    elif p_value < 0.01:
        return p_value, "**"
    elif p_value < 0.05:
        return p_value, "*"
    else:
        return p_value, "ns"


def _add_significance_bracket(ax, x1, x2, y, text, h=0.02):
    """
    添加显著性标记（带括号）

    Args:
        ax: matplotlib axis
        x1, x2: 括号的x坐标
        y: 括号的y坐标（相对于数据范围的比例）
        text: 显著性标记文本
        h: 括号高度（相对比例）
    """
    # 在坐标范围内添加标记
    ax.annotate('', xy=(x1, y), xytext=(x2, y),
                arrowprops=dict(arrowstyle='-', color='black', lw=1.5))
    ax.annotate('', xy=(x1, y), xytext=(x1, y - h * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                arrowprops=dict(arrowstyle='-', color='black', lw=1.5))
    ax.annotate('', xy=(x2, y), xytext=(x2, y - h * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                arrowprops=dict(arrowstyle='-', color='black', lw=1.5))
    ax.text((x1 + x2) / 2, y, text, ha='center', va='bottom', fontsize=12, fontweight='bold')


def main():
    parser = argparse.ArgumentParser(description="统计分布对比图")
    parser.add_argument("--sac", type=str, required=True, help="SAC评估JSON文件路径")
    parser.add_argument("--ppo", type=str, required=True, help="PPO评估JSON文件路径")
    parser.add_argument("--td3", type=str, required=True, help="TD3评估JSON文件路径")
    parser.add_argument(
        "--output",
        type=str,
        default="assets/results/demo_delivery/figures/statistical_comparison",
        help="输出文件路径（不含扩展名）"
    )

    args = parser.parse_args()

    # 应用科学出版物样式
    if HAS_SCI_STYLE:
        apply_publication_style("nature")
        set_color_palette("okabe_ito")

    # 加载数据
    data = {
        "SAC": _load_json(args.sac),
        "PPO": _load_json(args.ppo),
        "TD3": _load_json(args.td3),
    }

    labels = ["SAC", "PPO", "TD3"]
    colors = ["#0072B2", "#E69F00", "#D55E00"]  # 蓝色、橙色、红橙色
    success_colors = ["#2ca02c", "#d62728", "#d62728"]  # 绿色（成功）、红色（失败）

    # 提取数据
    min_distances = {}
    final_velocities = {}
    final_headings = {}
    steps = {}

    for name in labels:
        records = data[name]["episode_records"]
        min_distances[name] = np.array([r["min_distance"] for r in records])
        final_velocities[name] = np.array([r["final_velocity"] for r in records])
        final_headings[name] = np.array([abs(r["final_heading_deg"]) for r in records])
        steps[name] = np.array([r["steps"] for r in records])

    # 获取成功阈值
    pos_tolerance = data["SAC"]["tolerances"]["position"]
    vel_tolerance = data["SAC"]["tolerances"]["velocity"]
    heading_tolerance_deg = np.rad2deg(data["SAC"]["tolerances"]["heading_rad"])

    # 创建 2x2 子图（增加图表尺寸和边距）
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    rng = np.random.default_rng(42)

    # ========== 左上：最小距离分布（对数尺度） ==========
    ax = axes[0, 0]
    positions = [1, 2, 3]

    # 绘制 violin plot
    parts = ax.violinplot(
        [min_distances[name] for name in labels],
        positions=positions,
        showmeans=False,
        showmedians=False,
        widths=0.6
    )

    # 设置 violin plot 颜色
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)

    # 绘制 box plot（叠加在 violin plot 上）
    bp = ax.boxplot(
        [min_distances[name] for name in labels],
        positions=positions,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5)
    )

    # 添加散点（带抖动）
    for idx, (name, color) in enumerate(zip(labels, success_colors), start=1):
        dist = min_distances[name]
        jitter = (rng.random(len(dist)) - 0.5) * 0.15
        ax.scatter(
            np.full_like(dist, idx, dtype=float) + jitter,
            dist,
            s=20,
            alpha=0.6,
            color=color,
            edgecolors='black',
            linewidths=0.5
        )

    # 添加成功阈值线
    ax.axhline(pos_tolerance, color='black', linestyle='--', linewidth=2,
               label=f'Success Threshold ({pos_tolerance}m)', zorder=10)

    # 设置对数尺度和坐标范围
    ax.set_yscale('log')
    all_dist = np.concatenate([min_distances[name] for name in labels])
    y_min = max(0.1, min(all_dist) * 0.5)
    y_max = max(all_dist) * 2.5  # 留出更多顶部空间
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylabel('Minimum Distance to Goal (m)', fontsize=12, fontweight='bold')
    ax.set_title('A  Minimum Distance Distribution (Log Scale)', fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, which='both')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    # 添加显著性标记（在标题下方）
    p_sac_ppo, sig_sac_ppo = _compute_significance(min_distances["SAC"], min_distances["PPO"])
    sig_y = y_max * 0.7  # 在对数坐标中的位置
    ax.text(1.5, sig_y, sig_sac_ppo, ha='center', va='bottom', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    # ========== 右上：最终速度分布 ==========
    ax = axes[0, 1]

    # 绘制 violin plot
    parts = ax.violinplot(
        [final_velocities[name] for name in labels],
        positions=positions,
        showmeans=False,
        showmedians=False,
        widths=0.6
    )

    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)

    # 绘制 box plot
    bp = ax.boxplot(
        [final_velocities[name] for name in labels],
        positions=positions,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5)
    )

    # 添加散点
    for idx, (name, color) in enumerate(zip(labels, success_colors), start=1):
        vel = final_velocities[name]
        jitter = (rng.random(len(vel)) - 0.5) * 0.15
        ax.scatter(
            np.full_like(vel, idx, dtype=float) + jitter,
            vel,
            s=20,
            alpha=0.6,
            color=color,
            edgecolors='black',
            linewidths=0.5
        )

    # 添加速度阈值线
    ax.axhline(vel_tolerance, color='black', linestyle='--', linewidth=2,
               label=f'Success Threshold ({vel_tolerance}m/s)', zorder=10)

    # 设置坐标范围（留出顶部空间）
    all_vel = np.concatenate([final_velocities[name] for name in labels])
    y_max_vel = max(all_vel) * 1.3
    ax.set_ylim(0, y_max_vel)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Velocity (m/s)', fontsize=12, fontweight='bold')
    ax.set_title('B  Final Velocity Distribution', fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    # 添加显著性标记
    p_sac_ppo, sig_sac_ppo = _compute_significance(final_velocities["SAC"], final_velocities["PPO"])
    sig_y = y_max_vel * 0.92
    ax.text(1.5, sig_y, sig_sac_ppo, ha='center', va='bottom', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    # ========== 左下：最终航向误差分布 ==========
    ax = axes[1, 0]

    # 绘制 violin plot
    parts = ax.violinplot(
        [final_headings[name] for name in labels],
        positions=positions,
        showmeans=False,
        showmedians=False,
        widths=0.6
    )

    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)

    # 绘制 box plot
    bp = ax.boxplot(
        [final_headings[name] for name in labels],
        positions=positions,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5)
    )

    # 添加散点
    for idx, (name, color) in enumerate(zip(labels, success_colors), start=1):
        heading = final_headings[name]
        jitter = (rng.random(len(heading)) - 0.5) * 0.15
        ax.scatter(
            np.full_like(heading, idx, dtype=float) + jitter,
            heading,
            s=20,
            alpha=0.6,
            color=color,
            edgecolors='black',
            linewidths=0.5
        )

    # 添加航向阈值线
    ax.axhline(heading_tolerance_deg, color='black', linestyle='--', linewidth=2,
               label=f'Success Threshold ({heading_tolerance_deg:.1f} deg)', zorder=10)

    # 设置坐标范围（留出顶部空间）
    all_heading = np.concatenate([final_headings[name] for name in labels])
    y_max_head = max(all_heading) * 1.25
    ax.set_ylim(0, y_max_head)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Heading Error (degrees)', fontsize=12, fontweight='bold')
    ax.set_title('C  Final Heading Error Distribution', fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    # 添加显著性标记
    p_sac_ppo, sig_sac_ppo = _compute_significance(final_headings["SAC"], final_headings["PPO"])
    sig_y = y_max_head * 0.92
    ax.text(1.5, sig_y, sig_sac_ppo, ha='center', va='bottom', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    # ========== 右下：完成步数分布 ==========
    ax = axes[1, 1]

    # 绘制 violin plot
    parts = ax.violinplot(
        [steps[name] for name in labels],
        positions=positions,
        showmeans=False,
        showmedians=False,
        widths=0.6
    )

    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)

    # 绘制 box plot
    bp = ax.boxplot(
        [steps[name] for name in labels],
        positions=positions,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5)
    )

    # 添加散点
    for idx, (name, color) in enumerate(zip(labels, success_colors), start=1):
        step = steps[name]
        jitter = (rng.random(len(step)) - 0.5) * 0.15
        ax.scatter(
            np.full_like(step, idx, dtype=float) + jitter,
            step,
            s=20,
            alpha=0.6,
            color=color,
            edgecolors='black',
            linewidths=0.5
        )

    # 设置坐标范围（留出顶部空间）
    all_steps = np.concatenate([steps[name] for name in labels])
    y_max_steps = max(all_steps) * 1.15
    ax.set_ylim(0, y_max_steps)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylabel('Episode Steps', fontsize=12, fontweight='bold')
    ax.set_title('D  Episode Duration Distribution', fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)

    # 添加显著性标记
    p_sac_ppo, sig_sac_ppo = _compute_significance(steps["SAC"], steps["PPO"])
    sig_y = y_max_steps * 0.92
    ax.text(1.5, sig_y, sig_sac_ppo, ha='center', va='bottom', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    # 调整布局（增加子图间距）
    fig.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.5)

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

    # 打印统计摘要
    print("\n" + "="*60)
    print("统计摘要")
    print("="*60)
    for name in labels:
        print(f"\n{name}:")
        print(f"  最小距离: {np.mean(min_distances[name]):.2f} +/- {np.std(min_distances[name]):.2f} m")
        print(f"  最终速度: {np.mean(final_velocities[name]):.3f} +/- {np.std(final_velocities[name]):.3f} m/s")
        print(f"  航向误差: {np.mean(final_headings[name]):.2f} +/- {np.std(final_headings[name]):.2f} deg")
        print(f"  完成步数: {np.mean(steps[name]):.0f} +/- {np.std(steps[name]):.0f}")

    print("\n" + "="*60)
    print("显著性检验（SAC vs PPO）")
    print("="*60)
    p_dist, sig_dist = _compute_significance(min_distances["SAC"], min_distances["PPO"])
    p_vel, sig_vel = _compute_significance(final_velocities["SAC"], final_velocities["PPO"])
    p_head, sig_head = _compute_significance(final_headings["SAC"], final_headings["PPO"])
    p_step, sig_step = _compute_significance(steps["SAC"], steps["PPO"])

    print(f"最小距离: p={p_dist:.2e}, {sig_dist}")
    print(f"最终速度: p={p_vel:.2e}, {sig_vel}")
    print(f"航向误差: p={p_head:.2e}, {sig_head}")
    print(f"完成步数: p={p_step:.2e}, {sig_step}")
    print("="*60)


if __name__ == "__main__":
    main()
