"""
性能雷达图：SAC vs PPO vs TD3

从 6 个维度展示算法性能对比：
1. 成功率（Success Rate）
2. 平均最小距离倒数（Distance Precision）
3. 路径完成度（Path Progress）
4. 步数效率（Step Efficiency）
5. 航向精度（Heading Precision）
6. 速度控制（Velocity Control）

Input: 3个评估JSON文件（sac_evaluation.json, ppo_evaluation.json, td3_evaluation.json）
Output: 高质量PNG图表（300 dpi）+ 数值表格
Pos: 基线对比可视化系统
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def _normalize_metric(value, min_val, max_val, invert=False):
    """
    归一化指标到 0-1 范围

    Args:
        value: 原始值
        min_val: 最小值
        max_val: 最大值
        invert: 是否反转（对于"越小越好"的指标）
    """
    if max_val == min_val:
        return 1.0 if not invert else 0.0

    normalized = (value - min_val) / (max_val - min_val)

    if invert:
        normalized = 1.0 - normalized

    return np.clip(normalized, 0.0, 1.0)


def compute_performance_metrics(data, name):
    """
    计算 6 维性能指标

    Returns:
        dict: 包含原始值和归一化值的字典
    """
    records = data["episode_records"]

    # 1. 成功率（0-100%）
    success_rate = data["success_rate"] * 100

    # 2. 平均最小距离倒数（归一化）
    min_distances = np.array([r["min_distance"] for r in records])
    avg_min_distance = np.mean(min_distances)
    # 使用倒数，距离越小越好
    distance_precision = 1.0 / (avg_min_distance + 0.1)  # 加 0.1 避免除零

    # 3. 路径完成度（0-100%）
    path_progress = np.mean([r["path_progress"] for r in records]) * 100

    # 4. 步数效率（归一化）
    steps = np.array([r["steps"] for r in records])
    avg_steps = np.mean(steps)
    # 使用倒数，步数越少越好（但要考虑成功的情况）
    step_efficiency = 1.0 / (avg_steps + 1.0)

    # 5. 航向精度（归一化）
    heading_errors = np.array([abs(r["final_heading_deg"]) for r in records])
    avg_heading_error = np.mean(heading_errors)
    # 使用倒数，误差越小越好
    heading_precision = 1.0 / (avg_heading_error + 1.0)

    # 6. 速度控制（归一化）
    velocities = np.array([r["final_velocity"] for r in records])
    avg_velocity = np.mean(velocities)
    # 使用倒数，速度越接近 0 越好
    velocity_control = 1.0 / (avg_velocity + 0.01)

    return {
        "raw": {
            "success_rate": success_rate,
            "avg_min_distance": avg_min_distance,
            "distance_precision": distance_precision,
            "path_progress": path_progress,
            "avg_steps": avg_steps,
            "step_efficiency": step_efficiency,
            "avg_heading_error": avg_heading_error,
            "heading_precision": heading_precision,
            "avg_velocity": avg_velocity,
            "velocity_control": velocity_control,
        },
        "normalized": None  # 将在全局归一化后填充
    }


def main():
    parser = argparse.ArgumentParser(description="性能雷达图")
    parser.add_argument("--sac", type=str, required=True, help="SAC评估JSON文件路径")
    parser.add_argument("--ppo", type=str, required=True, help="PPO评估JSON文件路径")
    parser.add_argument("--td3", type=str, required=True, help="TD3评估JSON文件路径")
    parser.add_argument(
        "--output",
        type=str,
        default="assets/results/demo_delivery/figures/performance_radar",
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
    alphas = [0.25, 0.20, 0.20]  # 透明度

    # 计算所有算法的性能指标
    metrics = {}
    for name in labels:
        metrics[name] = compute_performance_metrics(data[name], name)

    # 全局归一化（基于所有算法的最小值和最大值）
    metric_names = [
        "success_rate", "distance_precision", "path_progress",
        "step_efficiency", "heading_precision", "velocity_control"
    ]

    for metric_name in metric_names:
        values = [metrics[name]["raw"][metric_name] for name in labels]
        min_val = min(values)
        max_val = max(values)

        for name in labels:
            raw_value = metrics[name]["raw"][metric_name]
            normalized = _normalize_metric(raw_value, min_val, max_val, invert=False)

            if metrics[name]["normalized"] is None:
                metrics[name]["normalized"] = {}
            metrics[name]["normalized"][metric_name] = normalized

    # 创建图表（雷达图 + 数值表格）
    fig = plt.figure(figsize=(14, 7))

    # 左侧：雷达图
    ax_radar = plt.subplot(121, projection='polar')

    # 6 个维度的标签
    categories = [
        'Success Rate\n(0-100%)',
        'Distance\nPrecision',
        'Path Progress\n(0-100%)',
        'Step\nEfficiency',
        'Heading\nPrecision',
        'Velocity\nControl'
    ]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # 闭合雷达图
    angles += angles[:1]

    # 绘制每个算法的雷达图
    for name, color, alpha in zip(labels, colors, alphas):
        values = [metrics[name]["normalized"][m] for m in metric_names]
        values += values[:1]  # 闭合

        ax_radar.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax_radar.fill(angles, values, alpha=alpha, color=color)

    # 设置雷达图样式
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, fontsize=10)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax_radar.grid(True, linestyle='--', alpha=0.5)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax_radar.set_title('Performance Radar Chart\n(Normalized Metrics)',
                       fontsize=13, fontweight='bold', pad=20)

    # 右侧：数值表格
    ax_table = plt.subplot(122)
    ax_table.axis('off')

    # 准备表格数据
    table_data = []
    table_data.append(['Metric', 'SAC', 'PPO', 'TD3'])

    # 添加原始值
    table_data.append(['Success Rate (%)',
                      f"{metrics['SAC']['raw']['success_rate']:.1f}",
                      f"{metrics['PPO']['raw']['success_rate']:.1f}",
                      f"{metrics['TD3']['raw']['success_rate']:.1f}"])

    table_data.append(['Avg Min Distance (m)',
                      f"{metrics['SAC']['raw']['avg_min_distance']:.2f}",
                      f"{metrics['PPO']['raw']['avg_min_distance']:.2f}",
                      f"{metrics['TD3']['raw']['avg_min_distance']:.2f}"])

    table_data.append(['Path Progress (%)',
                      f"{metrics['SAC']['raw']['path_progress']:.1f}",
                      f"{metrics['PPO']['raw']['path_progress']:.1f}",
                      f"{metrics['TD3']['raw']['path_progress']:.1f}"])

    table_data.append(['Avg Steps',
                      f"{metrics['SAC']['raw']['avg_steps']:.0f}",
                      f"{metrics['PPO']['raw']['avg_steps']:.0f}",
                      f"{metrics['TD3']['raw']['avg_steps']:.0f}"])

    table_data.append(['Avg Heading Error (°)',
                      f"{metrics['SAC']['raw']['avg_heading_error']:.2f}",
                      f"{metrics['PPO']['raw']['avg_heading_error']:.2f}",
                      f"{metrics['TD3']['raw']['avg_heading_error']:.2f}"])

    table_data.append(['Avg Final Velocity (m/s)',
                      f"{metrics['SAC']['raw']['avg_velocity']:.3f}",
                      f"{metrics['PPO']['raw']['avg_velocity']:.3f}",
                      f"{metrics['TD3']['raw']['avg_velocity']:.3f}"])

    # 添加归一化值
    table_data.append(['', '', '', ''])  # 空行
    table_data.append(['Normalized Metrics', '', '', ''])

    for i, metric_name in enumerate(metric_names):
        display_name = categories[i].replace('\n', ' ')
        table_data.append([display_name,
                          f"{metrics['SAC']['normalized'][metric_name]:.3f}",
                          f"{metrics['PPO']['normalized'][metric_name]:.3f}",
                          f"{metrics['TD3']['normalized'][metric_name]:.3f}"])

    # 创建表格
    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                          colWidths=[0.35, 0.2, 0.2, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置归一化指标标题样式
    for i in range(4):
        table[(8, i)].set_facecolor('#d0d0d0')
        table[(8, i)].set_text_props(weight='bold')

    # 高亮最佳值（归一化指标）
    for row_idx in range(9, 15):
        values = [float(table_data[row_idx][j]) for j in range(1, 4)]
        best_idx = values.index(max(values)) + 1
        table[(row_idx, best_idx)].set_facecolor('#90EE90')  # 浅绿色
        table[(row_idx, best_idx)].set_text_props(weight='bold')

    ax_table.set_title('Performance Metrics Summary',
                      fontsize=13, fontweight='bold', pad=20)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    output_base = Path(args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    if HAS_SCI_STYLE:
        save_publication_figure(fig, output_base, formats=["pdf", "png"], dpi=600)
        print(f"✓ 图表已保存（科学出版物格式）：{output_base}.pdf 和 {output_base}.png")
    else:
        fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存：{output_base}.png")

    plt.close(fig)

    # 打印性能摘要
    print("\n" + "="*60)
    print("性能雷达图摘要")
    print("="*60)
    for name in labels:
        print(f"\n{name}:")
        print(f"  成功率: {metrics[name]['raw']['success_rate']:.1f}%")
        print(f"  平均最小距离: {metrics[name]['raw']['avg_min_distance']:.2f} m")
        print(f"  路径完成度: {metrics[name]['raw']['path_progress']:.1f}%")
        print(f"  平均步数: {metrics[name]['raw']['avg_steps']:.0f}")
        print(f"  平均航向误差: {metrics[name]['raw']['avg_heading_error']:.2f}°")
        print(f"  平均最终速度: {metrics[name]['raw']['avg_velocity']:.3f} m/s")
        print(f"\n  归一化综合得分: {np.mean([metrics[name]['normalized'][m] for m in metric_names]):.3f}")

    print("="*60)


if __name__ == "__main__":
    main()
