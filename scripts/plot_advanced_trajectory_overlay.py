"""
高级多轨迹叠加可视化（使用已采集的轨迹数据）

展示 SAC 的多条成功轨迹叠加效果，突出其稳定性和一致性。

Input: SAC 轨迹 JSON 文件
Output: 高质量 PNG/PDF 图表（600 dpi）
Pos: 高级轨迹可视化系统
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

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


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def generate_arc_reference(num_points=100):
    """生成弧形参考轨迹"""
    arc_radius = 100.0
    arc_length = 131.68
    arc_angle = arc_length / arc_radius

    center_x, center_y = 0.0, arc_radius
    start_angle = -np.pi / 2
    angles = np.linspace(start_angle, start_angle + arc_angle, num_points)

    ref_x = center_x + arc_radius * np.cos(angles)
    ref_y = center_y + arc_radius * np.sin(angles)

    return ref_x, ref_y


def main():
    parser = argparse.ArgumentParser(description="高级多轨迹叠加可视化")
    parser.add_argument("--sac_traj", type=str, required=True, help="SAC 轨迹 JSON 文件")
    parser.add_argument("--output", type=str,
                       default="assets/results/demo_delivery/figures/advanced_trajectory_overlay",
                       help="输出路径（不含扩展名）")
    parser.add_argument("--show_time_gradient", action="store_true", default=True,
                       help="显示时间渐变色")

    args = parser.parse_args()

    # 应用科学出版物样式
    if HAS_SCI_STYLE:
        apply_publication_style("nature")
        set_color_palette("okabe_ito")

    # 加载数据
    sac_data = load_json(args.sac_traj)
    trajectories = sac_data['trajectories']

    print("=" * 60)
    print("高级多轨迹叠加可视化")
    print("=" * 60)
    print(f"SAC: {sac_data['success_count']}/{sac_data['episodes']} "
          f"({sac_data['success_rate']*100:.1f}%)")
    print("=" * 60)

    # 生成参考轨迹
    ref_x, ref_y = generate_arc_reference()

    # 泊位位置
    berth_x, berth_y = 69.77, 94.87

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 10))

    # 参考轨迹
    ax.plot(ref_x, ref_y, 'k--', linewidth=2.5, label='Reference Trajectory',
            alpha=0.7, zorder=1)

    # 泊位区域
    berth_radius = 2.0
    berth_circle = Circle(
        (berth_x, berth_y), berth_radius,
        facecolor='lightcoral', alpha=0.4, edgecolor='red', linewidth=2.5,
        label='Berth Area', zorder=2
    )
    ax.add_patch(berth_circle)

    # 泊位中心标记
    ax.plot(berth_x, berth_y, 'r*', markersize=20, label='Berth Center',
            zorder=10, markeredgecolor='darkred', markeredgewidth=1.5)

    # 统计成功/失败
    success_count = sum(1 for t in trajectories if t.get('success', False))
    failed_count = len(trajectories) - success_count

    # 绘制轨迹
    success_plotted = False
    failed_plotted = False

    for i, traj in enumerate(trajectories):
        x = np.array(traj['states']['x'])
        y = np.array(traj['states']['y'])
        success = traj.get('success', False)

        if success:
            color = '#2ca02c'  # 绿色
            alpha = 0.6
            linewidth = 2.5
            zorder = 5
            label = f'Successful ({success_count})' if not success_plotted else None
            success_plotted = True
        else:
            color = '#d62728'  # 红色
            alpha = 0.4
            linewidth = 2.0
            zorder = 4
            label = f'Failed ({failed_count})' if not failed_plotted else None
            failed_plotted = True

        if args.show_time_gradient and len(x) > 1:
            # 使用 LineCollection 创建渐变色轨迹
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # 颜色渐变（从浅到深）
            alphas = np.linspace(0.2, alpha, len(segments))
            colors = [(*plt.cm.colors.to_rgb(color), a) for a in alphas]

            lc = LineCollection(segments, colors=colors, linewidths=linewidth,
                              zorder=zorder, label=label)
            ax.add_collection(lc)
        else:
            ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth,
                   label=label, zorder=zorder)

        # 标注成功点（仅前5条）
        if success and i < 5:
            ax.plot(x[-1], y[-1], 'o', color=color, markersize=10,
                   markeredgecolor='white', markeredgewidth=1.5, zorder=6)

        # 标注失败点
        if not success and i < 10:
            ax.plot(x[-1], y[-1], 'X', color=color, markersize=12,
                   markeredgewidth=2.5, zorder=6)

    # 起点标记
    ax.plot(0, 0, 'bo', markersize=15, label='Start', zorder=10,
           markeredgecolor='darkblue', markeredgewidth=1.5)

    # 图例
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

    # 坐标轴设置
    ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')

    # 标题
    success_rate = success_count / len(trajectories) * 100
    title = (f'SAC Multi-Trajectory Overlay (N={len(trajectories)})\n'
             f'Success Rate: {success_rate:.1f}% ({success_count}/{len(trajectories)})')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    # 网格和纵横比
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=12)

    # 调整坐标轴范围
    all_x = list(ref_x) + [berth_x, 0]
    all_y = list(ref_y) + [berth_y, 0]
    for traj in trajectories:
        all_x.extend(traj['states']['x'])
        all_y.extend(traj['states']['y'])

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    output_base = Path(args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    if HAS_SCI_STYLE:
        save_publication_figure(fig, output_base, formats=["pdf", "png"], dpi=600)
        print(f"\n图表已保存（科学出版物格式）：")
        print(f"  - {output_base}.pdf (矢量图)")
        print(f"  - {output_base}.png (600 dpi)")
    else:
        fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches='tight')
        fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches='tight')
        print(f"\n图表已保存：{output_base}.png 和 {output_base}.pdf")

    plt.close(fig)

    # 打印统计摘要
    print("\n" + "=" * 60)
    print("轨迹统计摘要")
    print("=" * 60)
    print(f"总轨迹数: {len(trajectories)}")
    print(f"成功轨迹: {success_count} ({success_rate:.1f}%)")
    print(f"失败轨迹: {failed_count} ({100-success_rate:.1f}%)")
    print("=" * 60)


if __name__ == '__main__':
    main()
