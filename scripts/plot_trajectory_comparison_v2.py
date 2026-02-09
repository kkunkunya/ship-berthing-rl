"""
轨迹对比图 V2：SAC vs PPO vs TD3（改进版）

核心目标：直观展示 SAC 能成功到达终点，而 PPO/TD3 无法成功

改进点：
1. 图表更高，比例更合理
2. SAC 显示真实的多条成功轨迹（绿色）
3. PPO/TD3 显示多条失败轨迹（红色/橙色）
4. 清晰标注失败点和成功点
5. 使用正确的弧形参考轨迹和泊位位置

Input: SAC/PPO/TD3 轨迹 JSON 文件
Output: 高质量 PNG/PDF 图表
Pos: 轨迹对比可视化系统
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    """生成弧形参考轨迹（与环境一致）"""
    # 参数与 VectorizedOtterEnv 一致
    arc_radius = 100.0
    arc_length = 131.68
    arc_angle = arc_length / arc_radius  # 弧度

    # 圆心在 (0, arc_radius)
    center_x, center_y = 0.0, arc_radius

    # 从 -π/2 开始，沿逆时针方向
    start_angle = -np.pi / 2
    angles = np.linspace(start_angle, start_angle + arc_angle, num_points)

    ref_x = center_x + arc_radius * np.cos(angles)
    ref_y = center_y + arc_radius * np.sin(angles)

    return ref_x, ref_y


def plot_algorithm_panel(ax, trajectories, algorithm, success_rate, is_success,
                         max_trajectories=30, ref_x=None, ref_y=None,
                         berth_x=69.77, berth_y=94.87):
    """绘制单个算法的轨迹面板"""

    # 参考轨迹
    if ref_x is not None and ref_y is not None:
        ax.plot(ref_x, ref_y, 'k--', linewidth=2.5, label='Reference', alpha=0.7, zorder=1)

    # 泊位区域（圆形）
    berth_radius = 2.0
    berth_circle = Circle(
        (berth_x, berth_y), berth_radius,
        facecolor='lightblue', alpha=0.6, edgecolor='blue', linewidth=2,
        label='Berth', zorder=2
    )
    ax.add_patch(berth_circle)

    # 泊位中心标记
    ax.plot(berth_x, berth_y, 'b*', markersize=15, zorder=10)

    # 起点标记
    ax.plot(0, 0, 'ko', markersize=12, label='Start', zorder=10,
            markerfacecolor='white', markeredgewidth=2)

    # 颜色配置
    if is_success:
        main_color = '#2ca02c'  # 绿色
        bg_color = '#e6ffe6'
        title_color = 'darkgreen'
    else:
        main_color = '#d62728' if algorithm == 'PPO' else '#ff7f0e'  # 红色/橙色
        bg_color = '#ffe6e6' if algorithm == 'PPO' else '#fff3e6'
        title_color = 'darkred' if algorithm == 'PPO' else 'darkorange'

    # 绘制轨迹
    trajs = trajectories[:max_trajectories]
    for i, traj in enumerate(trajs):
        x = traj['states']['x']
        y = traj['states']['y']

        # 透明度根据轨迹编号递减
        alpha = max(0.3, 0.8 - i * 0.02)
        linewidth = 2.5 if i < 5 else 1.5

        ax.plot(x, y, color=main_color, linewidth=linewidth, alpha=alpha, zorder=3)

        # 标注终点
        if is_success:
            # 成功：绿色圆点
            if i < 5:  # 只标注前5条
                ax.plot(x[-1], y[-1], 'o', color=main_color, markersize=8,
                       markeredgecolor='white', markeredgewidth=1, zorder=5)
        else:
            # 失败：红色/橙色 X
            if i < 10:  # 标注前10条失败点
                ax.plot(x[-1], y[-1], 'X', color=main_color, markersize=10,
                       markeredgewidth=2, zorder=5)

    # 添加成功率标签
    success_count = sum(1 for t in trajectories if t.get('success', False))
    total_count = len(trajectories)

    label_text = f'{algorithm}: {success_count}/{total_count}\n({success_rate*100:.0f}% Success)'

    bbox_props = dict(
        boxstyle='round,pad=0.5',
        facecolor=bg_color,
        edgecolor=main_color,
        linewidth=2,
        alpha=0.9
    )

    ax.text(0.5, 0.95, label_text,
            transform=ax.transAxes,
            fontsize=14, fontweight='bold',
            color=title_color,
            ha='center', va='top',
            bbox=bbox_props,
            zorder=20)

    # 坐标轴设置
    ax.set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')

    # 子图标题
    status = '(Successful)' if is_success else '(Failed)'
    ax.set_title(f'{algorithm} {status}', fontsize=15, fontweight='bold',
                color=title_color, pad=10)

    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.tick_params(labelsize=11)
    ax.set_aspect('equal')


def main():
    parser = argparse.ArgumentParser(description="轨迹对比图 V2")
    parser.add_argument("--sac_traj", type=str, required=True, help="SAC 轨迹 JSON")
    parser.add_argument("--ppo_traj", type=str, required=True, help="PPO 轨迹 JSON")
    parser.add_argument("--td3_traj", type=str, required=True, help="TD3 轨迹 JSON")
    parser.add_argument("--output", type=str,
                       default="assets/results/demo_delivery/figures/trajectory_comparison",
                       help="输出路径（不含扩展名）")
    parser.add_argument("--max_trajectories", type=int, default=30, help="每个算法显示的最大轨迹数")

    args = parser.parse_args()

    # 应用科学出版物样式
    if HAS_SCI_STYLE:
        apply_publication_style("nature")
        set_color_palette("okabe_ito")

    # 加载数据
    sac_data = load_json(args.sac_traj)
    ppo_data = load_json(args.ppo_traj)
    td3_data = load_json(args.td3_traj)

    print("=" * 60)
    print("轨迹数据摘要")
    print("=" * 60)
    print(f"SAC: {sac_data['success_count']}/{sac_data['episodes']} "
          f"({sac_data['success_rate']*100:.1f}%)")
    print(f"PPO: {ppo_data['success_count']}/{ppo_data['episodes']} "
          f"({ppo_data['success_rate']*100:.1f}%)")
    print(f"TD3: {td3_data['success_count']}/{td3_data['episodes']} "
          f"({td3_data['success_rate']*100:.1f}%)")
    print("=" * 60)

    # 生成参考轨迹
    ref_x, ref_y = generate_arc_reference()

    # 泊位位置（与环境一致）
    berth_x, berth_y = 69.77, 94.87

    # 创建图表 - 更高的比例
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    # 计算坐标范围（基于参考轨迹和所有轨迹）
    all_x = list(ref_x) + [berth_x, 0]
    all_y = list(ref_y) + [berth_y, 0]
    for data in [sac_data, ppo_data, td3_data]:
        for traj in data['trajectories'][:args.max_trajectories]:
            all_x.extend(traj['states']['x'])
            all_y.extend(traj['states']['y'])

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # 添加边距
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    x_lim = (x_min - x_margin, x_max + x_margin)
    y_lim = (y_min - y_margin, y_max + y_margin)

    # 绘制三个面板
    # SAC（成功）
    plot_algorithm_panel(
        axes[0], sac_data['trajectories'], 'SAC',
        sac_data['success_rate'], is_success=True,
        max_trajectories=args.max_trajectories,
        ref_x=ref_x, ref_y=ref_y,
        berth_x=berth_x, berth_y=berth_y
    )
    axes[0].set_xlim(x_lim)
    axes[0].set_ylim(y_lim)
    axes[0].legend(loc='lower right', fontsize=10)

    # PPO（失败）
    plot_algorithm_panel(
        axes[1], ppo_data['trajectories'], 'PPO',
        ppo_data['success_rate'], is_success=False,
        max_trajectories=args.max_trajectories,
        ref_x=ref_x, ref_y=ref_y,
        berth_x=berth_x, berth_y=berth_y
    )
    axes[1].set_xlim(x_lim)
    axes[1].set_ylim(y_lim)
    axes[1].legend(loc='lower right', fontsize=10)

    # TD3（失败）
    plot_algorithm_panel(
        axes[2], td3_data['trajectories'], 'TD3',
        td3_data['success_rate'], is_success=False,
        max_trajectories=args.max_trajectories,
        ref_x=ref_x, ref_y=ref_y,
        berth_x=berth_x, berth_y=berth_y
    )
    axes[2].set_xlim(x_lim)
    axes[2].set_ylim(y_lim)
    axes[2].legend(loc='lower right', fontsize=10)

    # 添加总标题
    fig.suptitle('Trajectory Comparison: SAC vs PPO vs TD3\n'
                 '(30 Episodes, Fixed Disturbance: Vc=0.075 m/s, Vw=1.2 m/s)',
                 fontsize=16, fontweight='bold', y=0.98)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存图表
    output_base = Path(args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    if HAS_SCI_STYLE:
        save_publication_figure(fig, output_base, formats=["pdf", "png"], dpi=600)
        print(f"\n✓ 图表已保存（科学出版物格式）：")
        print(f"  - {output_base}.pdf")
        print(f"  - {output_base}.png")
    else:
        fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches='tight')
        fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches='tight')
        print(f"\n✓ 图表已保存：{output_base}.png 和 {output_base}.pdf")

    plt.close(fig)


if __name__ == "__main__":
    main()
