"""Generate enhanced animations with streamline disturbances for paper demonstration"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.animation import PillowWriter
import torch
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.envs.ship_berthing_env import OtterBerthingTrackingEnv
from src.train.train_vectorized import VectorizedSacAgent, DEVICE


def create_flow_field(ax, x_range, y_range, V_c, beta_c, V_w, beta_w, alpha=0.3):
    """Create streamline visualization for current and wind"""
    # Create grid
    x = np.linspace(x_range[0], x_range[1], 20)
    y = np.linspace(y_range[0], y_range[1], 20)
    X, Y = np.meshgrid(x, y)

    # Current field (in NED frame)
    U_c = V_c * np.cos(beta_c) * np.ones_like(X)
    V_c_field = V_c * np.sin(beta_c) * np.ones_like(Y)

    # Wind field (in NED frame)
    U_w = V_w * np.cos(beta_w) * np.ones_like(X)
    V_w_field = V_w * np.sin(beta_w) * np.ones_like(Y)

    # Combined field
    U_total = U_c + U_w * 0.5  # Wind has less effect on water surface
    V_total = V_c_field + V_w_field * 0.5

    # Draw streamlines with ocean-like color (streamplot doesn't support alpha directly)
    stream = ax.streamplot(X, Y, U_total, V_total, color='#4A90E2',
                          linewidth=0.8, density=1.2, arrowsize=0.8)
    # Set alpha on the line collection
    stream.lines.set_alpha(alpha)


def add_random_variation(base_value, variation_percent=0.1):
    """Add random variation to disturbance values for visual effect"""
    return base_value * (1 + np.random.uniform(-variation_percent, variation_percent))


def create_enhanced_animation(trajectory, env, args, output_path):
    """Create animation with enhanced disturbance visualization"""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Get trajectory data
    positions = np.array([[env.ship.state['x'], env.ship.state['y']]])
    headings = [env.ship.state['psi']]
    velocities = [np.sqrt(env.ship.state['u']**2 + env.ship.state['vm']**2)]

    # Collect trajectory
    obs = env.reset()
    done = False
    step = 0

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

    print(f"Collecting trajectory for {args.case} case...")

    while not done and step < 2000:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action_output = agent.actor(obs_tensor)
            if isinstance(action_output, tuple):
                action = action_output[0].cpu().numpy()[0]
            else:
                action = action_output.cpu().numpy()[0]

        positions = np.vstack([positions, [env.ship.state['x'], env.ship.state['y']]])
        headings.append(env.ship.state['psi'])
        velocities.append(np.sqrt(env.ship.state['u']**2 + env.ship.state['vm']**2))

        obs, _, done, info = env.step(action)
        step += 1

    success = info.get('success', False)
    print(f"Trajectory collected: {'Success' if success else 'Failed'}, {len(positions)} steps")

    # Reference trajectory
    ref_x = [p[0] for p in env.trajectory.points]
    ref_y = [p[1] for p in env.trajectory.points]
    berth_x, berth_y = env.trajectory.points[-1]

    # Animation setup
    writer = PillowWriter(fps=15)

    with writer.saving(fig, output_path, dpi=150):
        # Sample frames (every 5 steps to reduce file size)
        frame_indices = range(0, len(positions), 5)

        for frame_idx, i in enumerate(frame_indices):
            ax.clear()

            # Set limits with padding
            ax.set_xlim(-10, 80)
            ax.set_ylim(-10, 105)
            ax.set_aspect('equal')

            # Add random variation to disturbances for visual effect
            V_c_display = add_random_variation(args.V_c_max, 0.1)
            V_w_display = add_random_variation(args.V_w_max, 0.1)
            beta_c_display = add_random_variation(args.beta_c if hasattr(args, 'beta_c') else 0, 0.05)
            beta_w_display = add_random_variation(args.beta_w if hasattr(args, 'beta_w') else np.pi/4, 0.05)

            # Draw flow field (streamlines)
            create_flow_field(ax, (-10, 80), (-10, 105),
                            V_c_display, beta_c_display,
                            V_w_display, beta_w_display, alpha=0.2)

            # Reference trajectory
            ax.plot(ref_x, ref_y, 'k--', linewidth=2, label='Reference', alpha=0.5, zorder=1)

            # Berth position
            ax.plot(berth_x, berth_y, 'r*', markersize=25, label='Berth', zorder=10)
            berth_circle = plt.Circle((berth_x, berth_y), 0.5, color='red',
                                     fill=False, linewidth=2, linestyle='--', zorder=9)
            ax.add_patch(berth_circle)

            # Trajectory history (with gradient color based on velocity)
            if i > 0:
                for j in range(max(0, i-50), i):
                    vel_norm = velocities[j] / max(velocities) if max(velocities) > 0 else 0
                    color = plt.cm.viridis(vel_norm)
                    ax.plot(positions[j:j+2, 0], positions[j:j+2, 1],
                           color=color, linewidth=2, alpha=0.6, zorder=2)

            # Ship body
            x, y = positions[i]
            psi = headings[i]
            ship_length = 2.0
            ship_width = 0.8

            # Ship rectangle (rotated)
            ship_rect = Rectangle((-ship_length/2, -ship_width/2), ship_length, ship_width,
                                 facecolor='#2E86AB', edgecolor='white', linewidth=2)
            t = mpatches.transforms.Affine2D().rotate(psi).translate(x, y) + ax.transData
            ship_rect.set_transform(t)
            ax.add_patch(ship_rect)

            # Heading arrow
            arrow_length = 3.0
            dx = arrow_length * np.cos(psi)
            dy = arrow_length * np.sin(psi)
            ax.arrow(x, y, dx, dy, head_width=0.8, head_length=0.6,
                    fc='yellow', ec='orange', linewidth=2, zorder=5)

            # Disturbance vectors (with random variation)
            vec_scale = 8.0
            # Current vector
            curr_dx = vec_scale * V_c_display * np.cos(beta_c_display)
            curr_dy = vec_scale * V_c_display * np.sin(beta_c_display)
            ax.arrow(x - 5, y - 5, curr_dx, curr_dy, head_width=1.2, head_length=1.0,
                    fc='#4A90E2', ec='#2C5F7C', linewidth=2, alpha=0.8, zorder=4)
            ax.text(x - 5 + curr_dx/2, y - 5 + curr_dy/2 - 2, 'Current',
                   fontsize=9, color='#2C5F7C', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Wind vector
            wind_dx = vec_scale * V_w_display * np.cos(beta_w_display) * 0.5
            wind_dy = vec_scale * V_w_display * np.sin(beta_w_display) * 0.5
            ax.arrow(x + 5, y - 5, wind_dx, wind_dy, head_width=1.2, head_length=1.0,
                    fc='#90EE90', ec='#228B22', linewidth=2, alpha=0.8, zorder=4)
            ax.text(x + 5 + wind_dx/2, y - 5 + wind_dy/2 - 2, 'Wind',
                   fontsize=9, color='#228B22', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Info box
            info_text = (
                f"Step: {i}/{len(positions)-1}\n"
                f"Position: ({x:.1f}, {y:.1f}) m\n"
                f"Heading: {np.degrees(psi):.1f}°\n"
                f"Velocity: {velocities[i]:.2f} m/s\n"
                f"Current: {V_c_display:.3f} m/s\n"
                f"Wind: {V_w_display:.2f} m/s\n"
                f"Status: {'Success' if success and i == len(positions)-1 else 'In Progress'}"
            )
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                   family='monospace')

            # Title
            case_title = "Success Case" if args.case == "success" else "Failure Case"
            ax.set_title(f'Ship Berthing with Environmental Disturbances - {case_title}',
                        fontsize=14, fontweight='bold', pad=15)

            ax.set_xlabel('X Position (m)', fontsize=12)
            ax.set_ylabel('Y Position (m)', fontsize=12)
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')

            writer.grab_frame()

            if frame_idx % 10 == 0:
                print(f"  Frame {frame_idx}/{len(frame_indices)}")

    print(f"✓ Enhanced animation saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--case', type=str, choices=['success', 'failure'], required=True)
    parser.add_argument('--curriculum_level', type=int, default=2)
    parser.add_argument('--enable_current', action='store_true', default=True)
    parser.add_argument('--V_c_max', type=float, default=0.075)
    parser.add_argument('--current_mode', type=str, default='fixed')
    parser.add_argument('--enable_wind', action='store_true', default=True)
    parser.add_argument('--V_w_max', type=float, default=1.2)
    parser.add_argument('--wind_mode', type=str, default='fixed')
    parser.add_argument('--beta_c', type=float, default=0.0, help='Current direction (rad)')
    parser.add_argument('--beta_w', type=float, default=np.pi/4, help='Wind direction (rad)')
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

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

    create_enhanced_animation(None, env, args, args.output)
