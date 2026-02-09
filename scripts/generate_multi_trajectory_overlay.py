"""Generate multi-trajectory overlay visualization"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.ship_berthing_env import OtterBerthingTrackingEnv
from src.train.train_vectorized import VectorizedSacAgent, DEVICE

def collect_trajectories(checkpoint_path, num_episodes=30, curriculum_level=2,
                         enable_current=True, V_c_max=0.075, current_mode='fixed',
                         enable_wind=True, V_w_max=1.2, wind_mode='fixed'):
    """Collect multiple trajectories"""
    env = OtterBerthingTrackingEnv(
        curriculum_level=curriculum_level,
        enable_current=enable_current,
        V_c_max=V_c_max,
        current_mode=current_mode,
        enable_wind=enable_wind,
        V_w_max=V_w_max,
        wind_mode=wind_mode,
        verbose=False
    )

    # Load agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = VectorizedSacAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_envs=1,
        checkpoint_dir=str(Path(checkpoint_path).parent)
    )
    agent.load_models(checkpoint_path)

    trajectories = []
    for ep in range(num_episodes):
        obs = env.reset()
        trajectory = {'positions': [], 'success': False}

        done = False
        step = 0
        while not done and step < 2000:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action_output = agent.actor(obs_tensor)
                if isinstance(action_output, tuple):
                    action = action_output[0].cpu().numpy()[0]
                else:
                    action = action_output.cpu().numpy()[0]

            trajectory['positions'].append([env.ship.state['x'], env.ship.state['y']])
            obs, _, done, info = env.step(action)
            step += 1

        trajectory['success'] = info.get('success', False)
        trajectories.append(trajectory)
        print(f"Episode {ep+1}/{num_episodes}: {'Success' if trajectory['success'] else 'Failed'}")

    return trajectories, env

def plot_overlay(trajectories, env, output_path):
    """Plot all trajectories on one figure"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Reference trajectory
    ref_x = [p[0] for p in env.trajectory.points]
    ref_y = [p[1] for p in env.trajectory.points]
    ax.plot(ref_x, ref_y, 'k--', linewidth=2, label='Reference', alpha=0.7)

    # Berth position
    berth_x, berth_y = env.trajectory.points[-1]
    ax.plot(berth_x, berth_y, 'r*', markersize=20, label='Berth', zorder=10)
    ax.add_patch(plt.Circle((berth_x, berth_y), 0.5, color='red', fill=False, linewidth=2, linestyle='--'))

    # Count successes
    success_count = sum(1 for t in trajectories if t['success'])

    # Plot trajectories
    for i, traj in enumerate(trajectories):
        positions = np.array(traj['positions'])
        color = 'green' if traj['success'] else 'red'
        alpha = 0.4 if traj['success'] else 0.3
        linewidth = 1.5 if traj['success'] else 1.0

        # Only add label for first of each type
        if i == 0:
            label = f'Success ({success_count})' if traj['success'] else f'Failed ({len(trajectories)-success_count})'
        elif i == next((j for j, t in enumerate(trajectories) if not t['success']), None):
            label = f'Failed ({len(trajectories)-success_count})'
        else:
            label = None

        ax.plot(positions[:, 0], positions[:, 1], color=color, alpha=alpha,
                linewidth=linewidth, label=label)

    # Start position
    ax.plot(0, 0, 'bo', markersize=12, label='Start', zorder=10)

    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Multi-Trajectory Overlay (N={len(trajectories)}, Success Rate={success_count/len(trajectories)*100:.1f}%)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Multi-trajectory overlay saved to {output_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--curriculum_level', type=int, default=2)
    parser.add_argument('--enable_current', action='store_true', default=True)
    parser.add_argument('--V_c_max', type=float, default=0.075)
    parser.add_argument('--current_mode', type=str, default='fixed')
    parser.add_argument('--enable_wind', action='store_true', default=True)
    parser.add_argument('--V_w_max', type=float, default=1.2)
    parser.add_argument('--wind_mode', type=str, default='fixed')
    parser.add_argument('--output', type=str, default='assets/results/demo_delivery/multi_trajectory_overlay.png')

    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    trajectories, env = collect_trajectories(
        args.checkpoint, args.episodes, args.curriculum_level,
        args.enable_current, args.V_c_max, args.current_mode,
        args.enable_wind, args.V_w_max, args.wind_mode
    )

    plot_overlay(trajectories, env, args.output)
