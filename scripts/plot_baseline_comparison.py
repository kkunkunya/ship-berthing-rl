"""
Baseline Comparison: SAC vs PPO vs TD3（优化版）

简洁的成功率/失败率对比柱状图

Input: 3个评估JSON文件
Output: PNG图表
Pos: 基线对比可视化系统

优化改进：
1. 图例放入图内，不超出边界
2. 添加数值标签
3. 改进布局和配色
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot SAC vs PPO/TD3 baseline comparison")
    parser.add_argument("--sac", type=str, default="assets/results/demo_delivery/final_evaluation.json")
    parser.add_argument("--ppo", type=str, default="assets/results/baselines/ppo_evaluation.json")
    parser.add_argument("--td3", type=str, default="assets/results/baselines/td3_evaluation.json")
    parser.add_argument("--output", type=str, default="assets/results/demo_delivery/figures/baseline_comparison.png")

    args = parser.parse_args()

    paths = {
        "SAC": Path(args.sac),
        "PPO": Path(args.ppo),
        "TD3": Path(args.td3),
    }

    results = {}
    for name, path in paths.items():
        if path.exists():
            results[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            results[name] = None

    labels = []
    success_rates = []
    fail_rates = []

    for name in ["SAC", "PPO", "TD3"]:
        data = results.get(name)
        if not data:
            continue
        labels.append(name)
        success_rates.append(data.get("success_rate", 0.0))
        fail_rates.append(data.get("fail_rate", 1.0 - data.get("success_rate", 0.0)))

    x = np.arange(len(labels))
    width = 0.35

    # 创建更大的图表
    fig, ax = plt.subplots(figsize=(10, 6))

    # 使用更好的配色
    colors_success = ["#2ca02c", "#1f77b4", "#1f77b4"]  # SAC绿色，其他蓝色
    colors_fail = ["#d62728", "#d62728", "#d62728"]  # 红色表示失败

    # 绘制柱状图
    bars_success = ax.bar(x - width / 2, success_rates, width, label="Success Rate",
                          color=colors_success, edgecolor='black', linewidth=1.5)
    bars_fail = ax.bar(x + width / 2, fail_rates, width, label="Fail Rate",
                       color=colors_fail, edgecolor='black', linewidth=1.5, alpha=0.7)

    # 在柱子上方添加数值标签
    for bar, rate in zip(bars_success, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate*100:.0f}%', ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='darkgreen')

    for bar, rate in zip(bars_fail, fail_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate*100:.0f}%', ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='darkred')

    # 坐标轴设置
    ax.set_ylabel("Rate", fontsize=14, fontweight='bold')
    ax.set_title("Baseline Comparison: SAC vs PPO vs TD3\n(Fixed Disturbance: Vc=0.075 m/s, Vw=1.2 m/s)",
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.15)  # 留出空间显示标签
    ax.tick_params(labelsize=12)

    # 图例放在图内右上角
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9,
              edgecolor='gray', fancybox=True)

    # 网格
    ax.grid(axis="y", alpha=0.3, linewidth=0.8)

    # 添加水平分隔线
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

    # 添加注释框说明结果
    summary_text = (f"SAC: {success_rates[0]*100:.0f}% Success\n"
                    f"PPO: {success_rates[1]*100:.0f}% Success\n"
                    f"TD3: {success_rates[2]*100:.0f}% Success")
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                      edgecolor='orange', alpha=0.9))

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"图表已保存: {output_path}")
    print(f"\n性能摘要:")
    for name, sr in zip(labels, success_rates):
        print(f"  {name}: {sr*100:.0f}% 成功率")


if __name__ == "__main__":
    main()
