# Input: checkpoint模型, 环境配置
# Output: 轨迹可视化图表（trajectory_comparison.png, trajectory_detailed.png）
# Pos: 轨迹可视化脚本 - 展示在环境干扰下到达目的地的过程

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle
from pathlib import Path
import torch
import json

# 导入环境和智能体
from src.envs.ship_berthing_env import OtterBerthingTrackingEnv
from src.train.train import EnhancedPerSacAgent

# 配置
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 路径配置
CHECKPOINT_PATH = Path(r"C:\Project\1800-船舶\SAC-otter10.5\assets\checkpoints\demo_delivery\per_sac_otter_checkpoint_400.pth")
OUTPUT_DIR = Path(r"C:\Project\1800-船舶\SAC-otter10.5\assets\results\demo_delivery\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 环境配置（与评估一致）
ENV_CONFIG = {
    'curriculum_level': 2,
    'enable_current': True,
    'V_c_max': 0.075,
    'current_mode': 'fixed',
    'enable_wind': True,
    'V_w_max': 1.2,
    'wind_mode': 'fixed',
    'max_episode_steps': 2000,
    'dt': 0.1
}

def collect_trajectory(env, agent, num_episodes=5):
    """收集轨迹数据"""
    trajectories = []

    for ep in range(num_episodes):
        obs = env.reset()
        trajectory = {
            'x': [], 'y': [], 'psi': [], 'u': [], 'vm': [], 'r': [],
            'x_ref': [], 'y_ref': [], 'psi_ref': [], 'u_ref': [],
            'actions': [], 'rewards': [], 'success': False
        }

        done = False
        step = 0
        while not done and step < env.max_episode_steps:
            # 智能体决策
            action = agent.select_action(obs, evaluate=True)

            # 记录状态
            trajectory['x'].append(env.ship.state['x'])
            trajectory['y'].append(env.ship.state['y'])
            trajectory['psi'].append(env.ship.state['psi'])
            trajectory['u'].append(env.ship.state['u'])
            trajectory['vm'].append(env.ship.state['vm'])
            trajectory['r'].append(env.ship.state['r'])

            # 记录参考轨迹
            if env.closest_path_idx < len(env.trajectory.s_arr):
                current_s = env.trajectory.s_arr[env.closest_path_idx]
                ref_x, ref_y, ref_psi, ref_velocity, ref_r = env._get_ref_full(current_s)
            else:
                ref_x, ref_y = env.trajectory.points[-1]
                ref_psi = env.trajectory.end_heading
                ref_velocity = 0.0
                ref_r = 0.0

            trajectory['x_ref'].append(ref_x)
            trajectory['y_ref'].append(ref_y)
            trajectory['psi_ref'].append(ref_psi)
            trajectory['u_ref'].append(ref_velocity)

            trajectory['actions'].append(action.copy())

            # 执行动作
            obs, reward, done, info = env.step(action)
            trajectory['rewards'].append(reward)
            step += 1

        # 记录最终状态
        trajectory['x'].append(env.ship.state['x'])
        trajectory['y'].append(env.ship.state['y'])
        trajectory['psi'].append(env.ship.state['psi'])
        trajectory['u'].append(env.ship.state['u'])
        trajectory['vm'].append(env.ship.state['vm'])
        trajectory['r'].append(env.ship.state['r'])

        # 记录最终参考轨迹
        if env.closest_path_idx < len(env.trajectory.s_arr):
            current_s = env.trajectory.s_arr[env.closest_path_idx]
            ref_x, ref_y, ref_psi, ref_velocity, ref_r = env._get_ref_full(current_s)
        else:
            ref_x, ref_y = env.trajectory.points[-1]
            ref_psi = env.trajectory.end_heading
            ref_velocity = 0.0
            ref_r = 0.0

        trajectory['x_ref'].append(ref_x)
        trajectory['y_ref'].append(ref_y)
        trajectory['psi_ref'].append(ref_psi)
        trajectory['u_ref'].append(ref_velocity)

        trajectory['success'] = info.get('success', False)
        trajectory['V_c'] = env.current_V_c
        trajectory['beta_c'] = env.current_beta_c
        trajectory['V_w'] = env.current_V_w
        trajectory['beta_w'] = env.current_beta_w

        # 转换为numpy数组
        for key in ['x', 'y', 'psi', 'u', 'vm', 'r', 'x_ref', 'y_ref', 'psi_ref', 'u_ref', 'rewards']:
            trajectory[key] = np.array(trajectory[key])
        trajectory['actions'] = np.array(trajectory['actions'])

        trajectories.append(trajectory)
        print(f"Episode {ep+1}/{num_episodes}: Success={trajectory['success']}, Steps={len(trajectory['x'])-1}")

    return trajectories

def plot_trajectory_comparison(trajectories, output_path):
    """绘制轨迹对比图（2x2网格）"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Berthing Trajectories Under Environmental Disturbances',
                 fontsize=16, fontweight='bold', y=0.995)

    # 选择4个代表性轨迹
    selected_indices = list(range(min(4, len(trajectories))))

    for idx, ax_idx in enumerate(selected_indices):
        ax = axes[idx // 2, idx % 2]
        traj = trajectories[ax_idx]

        # 参考轨迹
        ax.plot(traj['x_ref'], traj['y_ref'], '--', color='gray', linewidth=2,
                label='Reference Path', alpha=0.7)

        # 实际轨迹（颜色渐变）
        points = np.array([traj['x'], traj['y']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # 使用速度作为颜色映射
        colors = plt.cm.viridis(np.linspace(0, 1, len(traj['x'])-1))
        for i in range(len(segments)):
            ax.plot(segments[i, :, 0], segments[i, :, 1],
                   color=colors[i], linewidth=2.5, alpha=0.8)

        # 起点和终点
        ax.plot(traj['x'][0], traj['y'][0], 'go', markersize=12,
               label='Start', zorder=5)
        ax.plot(traj['x'][-1], traj['y'][-1], 'rs', markersize=12,
               label='End', zorder=5)

        # 泊位位置
        berth_x = 70.0
        berth_y = 95.0
        berth_rect = Rectangle((berth_x-2, berth_y-2), 4, 4,
                               linewidth=2, edgecolor='black',
                               facecolor='none', label='Berth')
        ax.add_patch(berth_rect)

        # 扰动向量（在起点附近显示）
        arrow_x = traj['x'][0] + 5
        arrow_y = traj['y'][0] + 5

        # 海流向量
        V_c = traj['V_c']
        beta_c = traj['beta_c']
        dx_c = V_c * np.cos(beta_c) * 10
        dy_c = V_c * np.sin(beta_c) * 10
        ax.arrow(arrow_x, arrow_y, dx_c, dy_c,
                head_width=1.5, head_length=1, fc='blue', ec='blue',
                linewidth=2, alpha=0.7)
        ax.text(arrow_x + dx_c/2, arrow_y + dy_c/2 - 2,
               f'Current\n{V_c:.3f} m/s', fontsize=8, color='blue',
               ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # 风力向量
        V_w = traj['V_w']
        beta_w = traj['beta_w']
        dx_w = V_w * np.cos(beta_w) * 3
        dy_w = V_w * np.sin(beta_w) * 3
        ax.arrow(arrow_x, arrow_y - 8, dx_w, dy_w,
                head_width=1.5, head_length=1, fc='red', ec='red',
                linewidth=2, alpha=0.7)
        ax.text(arrow_x + dx_w/2, arrow_y - 8 + dy_w/2 - 2,
               f'Wind\n{V_w:.1f} m/s', fontsize=8, color='red',
               ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # 设置坐标轴
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_title(f'({"abcd"[idx]}) Episode {ax_idx+1} - {"Success" if traj["success"] else "Failed"}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')

        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_path}")
    plt.close()

def plot_detailed_trajectory(trajectory, output_path, episode_idx=0):
    """绘制单个轨迹详细分析（4子图）"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    fig.suptitle(f'Detailed Trajectory Analysis - Episode {episode_idx+1}',
                 fontsize=16, fontweight='bold')

    # 子图1: 2D轨迹
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(trajectory['x_ref'], trajectory['y_ref'], '--', color='gray',
            linewidth=2, label='Reference', alpha=0.7)

    # 实际轨迹（颜色渐变）
    points = np.array([trajectory['x'], trajectory['y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory['x'])-1))
    for i in range(len(segments)):
        ax1.plot(segments[i, :, 0], segments[i, :, 1],
                color=colors[i], linewidth=2.5, alpha=0.8)

    ax1.plot(trajectory['x'][0], trajectory['y'][0], 'go', markersize=12, label='Start')
    ax1.plot(trajectory['x'][-1], trajectory['y'][-1], 'rs', markersize=12, label='End')

    # 泊位
    berth_rect = Rectangle((68, 93), 4, 4, linewidth=2, edgecolor='black',
                           facecolor='none', label='Berth')
    ax1.add_patch(berth_rect)

    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)
    ax1.set_title('(a) 2D Trajectory', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_aspect('equal', adjustable='box')

    # 子图2: 横向偏差
    ax2 = fig.add_subplot(gs[0, 1])
    cross_track_error = np.sqrt((trajectory['x'] - trajectory['x_ref'])**2 +
                                (trajectory['y'] - trajectory['y_ref'])**2)
    time = np.arange(len(cross_track_error)) * ENV_CONFIG['dt']
    ax2.plot(time, cross_track_error, color='#d62728', linewidth=2)
    ax2.axhline(0.8, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Tolerance (0.8m)')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Cross-track Error (m)', fontsize=11)
    ax2.set_title('(b) Position Error', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=9)

    # 子图3: 航向角
    ax3 = fig.add_subplot(gs[1, 0])
    time = np.arange(len(trajectory['psi'])) * ENV_CONFIG['dt']
    ax3.plot(time, np.rad2deg(trajectory['psi_ref']), '--', color='gray',
            linewidth=2, label='Reference', alpha=0.7)
    ax3.plot(time, np.rad2deg(trajectory['psi']), color='#1f77b4',
            linewidth=2, label='Actual')
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Heading (degrees)', fontsize=11)
    ax3.set_title('(c) Heading Angle', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='upper right', fontsize=9)

    # 子图4: 速度曲线
    ax4 = fig.add_subplot(gs[1, 1])
    time = np.arange(len(trajectory['u'])) * ENV_CONFIG['dt']
    ax4.plot(time, trajectory['u_ref'], '--', color='gray',
            linewidth=2, label='Reference', alpha=0.7)
    ax4.plot(time, trajectory['u'], color='#2ca02c',
            linewidth=2, label='Actual')
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Velocity (m/s)', fontsize=11)
    ax4.set_title('(d) Velocity Profile', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(loc='upper right', fontsize=9)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_path}")
    plt.close()

def main():
    print("="*60)
    print("轨迹可视化生成器 - Otter USV 靠泊系统")
    print("="*60)

    # 创建环境
    print("\n[1/4] 创建环境...")
    env = OtterBerthingTrackingEnv(**ENV_CONFIG)

    # 加载智能体
    print("[2/4] 加载智能体...")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = EnhancedPerSacAgent(state_dim, action_dim, fc1_dim=256, fc2_dim=256)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic1.load_state_dict(checkpoint['critic1_state_dict'])
    agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])
    print(f"✓ 已加载检查点: {CHECKPOINT_PATH}")

    # 收集轨迹
    print("\n[3/4] 收集轨迹数据...")
    trajectories = collect_trajectory(env, agent, num_episodes=5)

    # 生成可视化
    print("\n[4/4] 生成可视化图表...")

    # 图1: 轨迹对比
    plot_trajectory_comparison(trajectories, OUTPUT_DIR / "trajectory_comparison.png")

    # 图2: 详细分析（选择第一个成功的轨迹）
    success_idx = next((i for i, t in enumerate(trajectories) if t['success']), 0)
    plot_detailed_trajectory(trajectories[success_idx],
                            OUTPUT_DIR / "trajectory_detailed.png",
                            episode_idx=success_idx)

    # 保存轨迹数据
    trajectory_data = {
        'config': ENV_CONFIG,
        'num_episodes': len(trajectories),
        'success_count': sum(t['success'] for t in trajectories),
        'trajectories': []
    }

    for i, traj in enumerate(trajectories):
        trajectory_data['trajectories'].append({
            'episode': i,
            'success': bool(traj['success']),
            'steps': len(traj['x']) - 1,
            'final_position': [float(traj['x'][-1]), float(traj['y'][-1])],
            'final_heading': float(np.rad2deg(traj['psi'][-1])),
            'V_c': float(traj['V_c']),
            'beta_c': float(traj['beta_c']),
            'V_w': float(traj['V_w']),
            'beta_w': float(traj['beta_w'])
        })

    with open(OUTPUT_DIR / "trajectory_data.json", 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    print(f"✓ 已保存: {OUTPUT_DIR / 'trajectory_data.json'}")

    print("\n" + "="*60)
    print("所有轨迹图表生成完成！")
    print("="*60)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"  - trajectory_comparison.png (轨迹对比图)")
    print(f"  - trajectory_detailed.png (详细分析图)")
    print(f"  - trajectory_data.json (轨迹数据)")
    print("="*60)

if __name__ == "__main__":
    main()
