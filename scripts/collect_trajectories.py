"""Collect trajectory data and generate visualizations"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.ship_berthing_env import OtterBerthingTrackingEnv
from src.train.train_vectorized import VectorizedSacAgent, DEVICE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--curriculum_level", type=int, default=2)
    parser.add_argument("--enable_current", action="store_true")
    parser.add_argument("--V_c_max", type=float, default=0.075)
    parser.add_argument("--current_mode", type=str, default="fixed")
    parser.add_argument("--enable_wind", action="store_true")
    parser.add_argument("--V_w_max", type=float, default=1.2)
    parser.add_argument("--wind_mode", type=str, default="fixed")
    parser.add_argument("--output", type=str, default="assets/results/demo_delivery/trajectories.png")
    return parser.parse_args()


def main():
    args = parse_args()

    # 创建环境
    env = OtterBerthingTrackingEnv(
        curriculum_level=args.curriculum_level,
        enable_current=args.enable_current,
        V_c_max=args.V_c_max,
        current_mode=args.current_mode,
        enable_wind=args.enable_wind,
        V_w_max=args.V_w_max,
        wind_mode=args.wind_mode,
        verbose=False
    )

    # 加载模型
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = VectorizedSacAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_envs=1,
        checkpoint_dir=str(Path(args.checkpoint).parent)
    )
    agent.load_models(args.checkpoint)

    print(f"Observation space: {state_dim}D")
    print(f"Loading model: {args.checkpoint}")

    # 收集轨迹
    trajectories = []
    for ep in range(args.episodes):
        obs = env.reset()
        traj = {"x": [], "y": [], "psi": [], "u": [], "vm": [], "success": False}

        done = False
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action_output = agent.actor(obs_tensor)
                # Handle both tuple (mean, log_std) and tensor outputs
                if isinstance(action_output, tuple):
                    action = action_output[0].cpu().numpy()[0]
                else:
                    action = action_output.cpu().numpy()[0]

            traj["x"].append(env.ship.state["x"])
            traj["y"].append(env.ship.state["y"])
            traj["psi"].append(env.ship.state["psi"])
            traj["u"].append(env.ship.state["u"])
            traj["vm"].append(env.ship.state["vm"])

            obs, _, done, info = env.step(action)

        traj["success"] = info["success"]
        trajectories.append(traj)
        print(f"Episode {ep+1}: {'Success' if traj['success'] else 'Failed'}")

    # 可视化
    fig, ax = plt.subplots(figsize=(12, 10))

    # Reference trajectory
    ref_x = [p[0] for p in env.trajectory.points]
    ref_y = [p[1] for p in env.trajectory.points]
    ax.plot(ref_x, ref_y, 'k--', linewidth=2, label='Reference', alpha=0.7)

    # Actual trajectories
    for i, traj in enumerate(trajectories):
        color = 'green' if traj["success"] else 'red'
        alpha = 0.6 if traj["success"] else 0.4
        ax.plot(traj["x"], traj["y"], color=color, linewidth=1.5, alpha=alpha,
                label=f'Episode {i+1}' if i < 3 else None)

    # Berth
    berth_x, berth_y = env.DockingLine_x, env.DockingLine_y
    ax.add_patch(Rectangle((berth_x - 1, berth_y - 1), 2, 2,
                           facecolor='blue', alpha=0.3, label='Berth'))

    # Disturbance info
    if args.enable_current or args.enable_wind:
        info_text = []
        if args.enable_current:
            info_text.append(f"Current: {args.V_c_max} m/s ({args.current_mode})")
        if args.enable_wind:
            info_text.append(f"Wind: {args.V_w_max} m/s ({args.wind_mode})")
        ax.text(0.02, 0.98, '\n'.join(info_text), transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Trajectory Visualization (Level {args.curriculum_level})', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")

    # Statistics
    success_count = sum(1 for t in trajectories if t["success"])
    print(f"\nSuccess rate: {success_count}/{args.episodes} ({success_count/args.episodes*100:.1f}%)")


if __name__ == "__main__":
    main()
