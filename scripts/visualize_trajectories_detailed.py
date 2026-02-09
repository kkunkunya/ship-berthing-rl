"""Detailed trajectory visualization - includes velocity, heading, and cross-track error analysis"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.gridspec as gridspec

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.ship_berthing_env import OtterBerthingTrackingEnv
from src.train.train_vectorized import VectorizedSacAgent, DEVICE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--curriculum_level", type=int, default=2)
    parser.add_argument("--enable_current", action="store_true")
    parser.add_argument("--V_c_max", type=float, default=0.075)
    parser.add_argument("--current_mode", type=str, default="fixed")
    parser.add_argument("--enable_wind", action="store_true")
    parser.add_argument("--V_w_max", type=float, default=1.2)
    parser.add_argument("--wind_mode", type=str, default="fixed")
    parser.add_argument("--output", type=str, default="assets/results/demo_delivery/detailed_analysis.png")
    return parser.parse_args()


def main():
    args = parse_args()

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
    print(f"Level {args.curriculum_level} success criteria:")
    print(f"  - Position error < {env.position_tolerance}m")
    print(f"  - Heading error < {np.rad2deg(env.heading_tolerance):.1f}°")
    print(f"  - Velocity < {env.velocity_tolerance}m/s")
    print(f"  - Yaw rate < {env.yaw_rate_tolerance}rad/s")

    # 收集轨迹
    trajectories = []
    for ep in range(args.episodes):
        obs = env.reset()
        traj = {
            "x": [], "y": [], "psi": [], "u": [], "vm": [], "r": [],
            "cte": [], "dist": [], "success": False,
            "V_c": env.current_V_c, "beta_c": env.current_beta_c,
            "V_w": env.current_V_w, "beta_w": env.current_beta_w
        }

        done = False
        step = 0
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action_output = agent.actor(obs_tensor)
                if isinstance(action_output, tuple):
                    action = action_output[0].cpu().numpy()[0]
                else:
                    action = action_output.cpu().numpy()[0]

            traj["x"].append(env.ship.state["x"])
            traj["y"].append(env.ship.state["y"])
            traj["psi"].append(env.ship.state["psi"])
            traj["u"].append(env.ship.state["u"])
            traj["vm"].append(env.ship.state["vm"])
            traj["r"].append(env.ship.state["r"])

            berth_pos = env.trajectory.points[-1]
            dist = np.linalg.norm([env.ship.state["x"] - berth_pos[0],
                                  env.ship.state["y"] - berth_pos[1]])
            traj["dist"].append(dist)
            traj["cte"].append(env._calculate_cross_track_error(
                env.ship.state["x"], env.ship.state["y"]))

            obs, _, done, info = env.step(action)
            step += 1

        traj["success"] = info["success"]
        trajectories.append(traj)

        status = "✓ Success" if traj["success"] else "✗ Failed"
        final_dist = traj["dist"][-1]
        max_cte = max(traj["cte"])
        print(f"Episode {ep+1}: {status} | Final dist={final_dist:.3f}m | Max CTE={max_cte:.3f}m | Steps={step}")

    # 详细可视化
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. 轨迹图
    ax1 = fig.add_subplot(gs[0:2, 0])
    ref_x = [p[0] for p in env.trajectory.points]
    ref_y = [p[1] for p in env.trajectory.points]
    ax1.plot(ref_x, ref_y, 'k--', linewidth=2, label='Reference', alpha=0.7)

    for i, traj in enumerate(trajectories):
        color = 'green' if traj["success"] else 'red'
        alpha = 0.7 if traj["success"] else 0.5
        ax1.plot(traj["x"], traj["y"], color=color, linewidth=2, alpha=alpha,
                label=f'Ep{i+1}' if i < 3 else None)

        # Draw start and end points
        ax1.plot(traj["x"][0], traj["y"][0], 'go', markersize=8)
        ax1.plot(traj["x"][-1], traj["y"][-1], 'r^' if not traj["success"] else 'g^', markersize=10)

    # Berth
    berth_x, berth_y = env.DockingLine_x, env.DockingLine_y
    ax1.add_patch(Rectangle((berth_x - 1, berth_y - 1), 2, 2,
                           facecolor='blue', alpha=0.3, label='Berth'))

    # Disturbance arrows
    if args.enable_current and trajectories[0]["V_c"] > 0:
        V_c = trajectories[0]["V_c"]
        beta_c = trajectories[0]["beta_c"]
        arrow_scale = 20
        ax1.arrow(10, 10, V_c * np.cos(beta_c) * arrow_scale,
                 V_c * np.sin(beta_c) * arrow_scale,
                 head_width=2, head_length=3, fc='cyan', ec='cyan', linewidth=2,
                 label=f'Current {V_c:.3f}m/s')

    if args.enable_wind and trajectories[0]["V_w"] > 0:
        V_w = trajectories[0]["V_w"]
        beta_w = trajectories[0]["beta_w"]
        arrow_scale = 5
        ax1.arrow(10, 20, V_w * np.cos(beta_w) * arrow_scale,
                 V_w * np.sin(beta_w) * arrow_scale,
                 head_width=2, head_length=3, fc='orange', ec='orange', linewidth=2,
                 label=f'Wind {V_w:.3f}m/s')

    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title(f'Trajectory Visualization (Level {args.curriculum_level})', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 2. Cross-Track Error (CTE)
    ax2 = fig.add_subplot(gs[0, 1])
    for i, traj in enumerate(trajectories):
        color = 'green' if traj["success"] else 'red'
        ax2.plot(traj["cte"], color=color, alpha=0.6, linewidth=1.5)
    ax2.axhline(y=env.max_cross_track, color='r', linestyle='--', label=f'Max CTE={env.max_cross_track}m')
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Cross-Track Error (m)', fontsize=11)
    ax2.set_title('Cross-Track Error Over Time', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Velocity profile
    ax3 = fig.add_subplot(gs[1, 1])
    for i, traj in enumerate(trajectories):
        color = 'green' if traj["success"] else 'red'
        velocity = np.sqrt(np.array(traj["u"])**2 + np.array(traj["vm"])**2)
        ax3.plot(velocity, color=color, alpha=0.6, linewidth=1.5)
    ax3.axhline(y=env.velocity_tolerance, color='orange', linestyle='--',
               label=f'Tolerance={env.velocity_tolerance}m/s')
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Velocity (m/s)', fontsize=11)
    ax3.set_title('Velocity Profile', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Statistics
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    success_count = sum(1 for t in trajectories if t["success"])
    success_rate = success_count / len(trajectories) * 100

    final_dists = [t["dist"][-1] for t in trajectories]
    max_ctes = [max(t["cte"]) for t in trajectories]

    stats_text = f"""
    PERFORMANCE SUMMARY (Level {args.curriculum_level})
    ═══════════════════════════════════════════════════════════════
    Success Rate: {success_count}/{len(trajectories)} ({success_rate:.1f}%)

    Final Distance to Berth:
      - Mean: {np.mean(final_dists):.3f}m  |  Std: {np.std(final_dists):.3f}m
      - Min: {np.min(final_dists):.3f}m   |  Max: {np.max(final_dists):.3f}m
      - Tolerance: {env.position_tolerance}m

    Maximum Cross-Track Error:
      - Mean: {np.mean(max_ctes):.3f}m  |  Std: {np.std(max_ctes):.3f}m
      - Min: {np.min(max_ctes):.3f}m   |  Max: {np.max(max_ctes):.3f}m
      - Limit: {env.max_cross_track}m

    Disturbances:
      - Current: {"Enabled" if args.enable_current else "Disabled"} ({args.V_c_max} m/s, {args.current_mode})
      - Wind: {"Enabled" if args.enable_wind else "Disabled"} ({args.V_w_max} m/s, {args.wind_mode})

    Model: {Path(args.checkpoint).name}
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")
    print(f"\nOverall success rate: {success_count}/{len(trajectories)} ({success_rate:.1f}%)")


if __name__ == "__main__":
    main()
