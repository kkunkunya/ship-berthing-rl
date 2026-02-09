"""Generate animations of ship berthing process for paper supplementary materials"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.ship_berthing_env import OtterBerthingTrackingEnv
from src.train.train_vectorized import VectorizedSacAgent, DEVICE


def add_random_variation(base_value, variation_percent=0.1):
    """Add random variation to disturbance values for visual effect"""
    return base_value * (1 + np.random.uniform(-variation_percent, variation_percent))


def create_flow_field(ax, x_range, y_range, V_c, beta_c, V_w, beta_w, alpha=0.2):
    """Create streamline visualization for current and wind"""
    x = np.linspace(x_range[0], x_range[1], 20)
    y = np.linspace(y_range[0], y_range[1], 20)
    X, Y = np.meshgrid(x, y)

    U_c = V_c * np.cos(beta_c) * np.ones_like(X)
    V_c_field = V_c * np.sin(beta_c) * np.ones_like(Y)
    U_w = V_w * np.cos(beta_w) * np.ones_like(X)
    V_w_field = V_w * np.sin(beta_w) * np.ones_like(Y)

    U_total = U_c + U_w * 0.5
    V_total = V_c_field + V_w_field * 0.5

    stream = ax.streamplot(X, Y, U_total, V_total, color='#4A90E2',
                          linewidth=0.8, density=1.2, arrowsize=0.8)
    stream.lines.set_alpha(alpha)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="assets/results/demo_delivery/animations")
    parser.add_argument("--curriculum_level", type=int, default=2)
    parser.add_argument("--enable_current", action="store_true")
    parser.add_argument("--V_c_max", type=float, default=0.075)
    parser.add_argument("--current_mode", type=str, default="fixed")
    parser.add_argument("--enable_wind", action="store_true")
    parser.add_argument("--V_w_max", type=float, default=1.2)
    parser.add_argument("--wind_mode", type=str, default="fixed")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--format", type=str, default="mp4", choices=["mp4", "gif"])
    parser.add_argument("--max_episodes", type=int, default=5)
    return parser.parse_args()


def collect_episode_data(env, agent):
    """Collect full episode trajectory data"""
    obs = env.reset()
    trajectory = {
        "x": [], "y": [], "psi": [], "u": [], "vm": [], "r": [],
        "actions": [], "V_c": env.current_V_c, "beta_c": env.current_beta_c,
        "V_w": env.current_V_w, "beta_w": env.current_beta_w
    }

    done = False
    while not done:
        trajectory["x"].append(env.ship.state["x"])
        trajectory["y"].append(env.ship.state["y"])
        trajectory["psi"].append(env.ship.state["psi"])
        trajectory["u"].append(env.ship.state["u"])
        trajectory["vm"].append(env.ship.state["vm"])
        trajectory["r"].append(env.ship.state["r"])

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action_output = agent.actor(obs_tensor)
            if isinstance(action_output, tuple):
                action = action_output[0].cpu().numpy()[0]
            else:
                action = action_output.cpu().numpy()[0]

        trajectory["actions"].append(action.copy())
        obs, _, done, info = env.step(action)

    trajectory["success"] = info["success"]
    return trajectory


def create_animation(trajectory, env, args, output_path):
    """Create animation from trajectory data"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Reference trajectory
    ref_x = [p[0] for p in env.trajectory.points]
    ref_y = [p[1] for p in env.trajectory.points]

    # Berth position
    berth_x, berth_y = env.DockingLine_x, env.DockingLine_y

    def init():
        ax.clear()
        ax.plot(ref_x, ref_y, 'k--', linewidth=2, label='Reference', alpha=0.7)
        ax.add_patch(Rectangle((berth_x - 1, berth_y - 1), 2, 2,
                               facecolor='blue', alpha=0.3, label='Berth'))
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'Ship Berthing (Level {args.curriculum_level})', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim(-10, 80)
        ax.set_ylim(-10, 105)
        return []

    def update(frame):
        ax.clear()

        # Static elements
        ax.plot(ref_x, ref_y, 'k--', linewidth=2, label='Reference', alpha=0.7)
        ax.add_patch(Rectangle((berth_x - 1, berth_y - 1), 2, 2,
                               facecolor='blue', alpha=0.3, label='Berth'))

        # Trajectory up to current frame
        if frame > 0:
            ax.plot(trajectory["x"][:frame], trajectory["y"][:frame],
                   'g-', linewidth=2, alpha=0.7, label='Trajectory')

        # Current ship position
        x, y, psi = trajectory["x"][frame], trajectory["y"][frame], trajectory["psi"][frame]

        # Ship body (simplified rectangle)
        ship_length, ship_width = 2.0, 0.8
        ship_corners = np.array([
            [-ship_length/2, -ship_width/2],
            [ship_length/2, -ship_width/2],
            [ship_length/2, ship_width/2],
            [-ship_length/2, ship_width/2],
            [-ship_length/2, -ship_width/2]
        ])

        # Rotate and translate
        rot_matrix = np.array([[np.cos(psi), -np.sin(psi)],
                              [np.sin(psi), np.cos(psi)]])
        ship_rotated = ship_corners @ rot_matrix.T + np.array([x, y])
        ax.plot(ship_rotated[:, 0], ship_rotated[:, 1], 'r-', linewidth=2)
        ax.fill(ship_rotated[:, 0], ship_rotated[:, 1], 'red', alpha=0.5)

        # Heading arrow
        arrow_length = 3.0
        dx = arrow_length * np.cos(psi)
        dy = arrow_length * np.sin(psi)
        ax.arrow(x, y, dx, dy, head_width=0.8, head_length=1.0,
                fc='darkred', ec='darkred', linewidth=2)

        # Disturbance visualization with random variation
        if (args.enable_current and trajectory["V_c"] > 0) or (args.enable_wind and trajectory["V_w"] > 0):
            V_c_display = add_random_variation(trajectory["V_c"], 0.1) if args.enable_current else 0
            V_w_display = add_random_variation(trajectory["V_w"], 0.1) if args.enable_wind else 0
            beta_c_display = add_random_variation(trajectory["beta_c"], 0.05) if args.enable_current else 0
            beta_w_display = add_random_variation(trajectory["beta_w"], 0.05) if args.enable_wind else 0

            # Draw flow field (streamlines)
            create_flow_field(ax, (-10, 80), (-10, 105),
                            V_c_display, beta_c_display,
                            V_w_display, beta_w_display, alpha=0.2)

        # Info text
        u, vm = trajectory["u"][frame], trajectory["vm"][frame]
        velocity = np.sqrt(u**2 + vm**2)
        info_text = f'Step: {frame}/{len(trajectory["x"])-1}\n'
        info_text += f'Position: ({x:.1f}, {y:.1f})\n'
        info_text += f'Heading: {np.rad2deg(psi):.1f}°\n'
        info_text += f'Velocity: {velocity:.3f} m/s'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'Ship Berthing (Level {args.curriculum_level})', fontsize=14)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim(-10, 80)
        ax.set_ylim(-10, 105)

        return []

    # Create animation
    n_frames = len(trajectory["x"])
    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                 init_func=init, interval=1000/args.fps,
                                 blit=False, repeat=True)

    # Save
    if args.format == "mp4":
        ani.save(output_path, writer='ffmpeg', fps=args.fps, dpi=args.dpi,
                bitrate=2000, codec='libx264')
    else:  # gif
        ani.save(output_path, writer='pillow', fps=args.fps, dpi=args.dpi)

    plt.close(fig)


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
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

    # Load agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = VectorizedSacAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_envs=1,
        checkpoint_dir=str(Path(args.checkpoint).parent)
    )
    agent.load_models(args.checkpoint)

    print(f"Generating animations...")
    print(f"Model: {args.checkpoint}")
    print(f"Output: {output_dir}")

    # Collect episodes
    success_count = 0
    failure_count = 0

    for ep in range(args.max_episodes):
        print(f"\nEpisode {ep+1}/{args.max_episodes}...")
        trajectory = collect_episode_data(env, agent)

        # Determine filename
        if trajectory["success"]:
            if success_count == 0:  # Save first success
                filename = f"berthing_success.{args.format}"
                output_path = output_dir / filename
                print(f"  Creating animation: {filename}")
                create_animation(trajectory, env, args, output_path)
                print(f"  ✓ Saved: {output_path}")
            success_count += 1
        else:
            if failure_count == 0:  # Save first failure
                filename = f"berthing_failure.{args.format}"
                output_path = output_dir / filename
                print(f"  Creating animation: {filename}")
                create_animation(trajectory, env, args, output_path)
                print(f"  ✓ Saved: {output_path}")
            failure_count += 1

        # Stop if we have both success and failure
        if success_count > 0 and failure_count > 0:
            break

    print(f"\n✓ Animation generation complete!")
    print(f"  Success episodes: {success_count}")
    print(f"  Failure episodes: {failure_count}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
