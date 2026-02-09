"""Generate velocity and heading change plot for scientific publication"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.ship_berthing_env import OtterBerthingTrackingEnv
from src.train.train_vectorized import VectorizedSacAgent, DEVICE


def collect_trajectory(env, agent):
    """Collect trajectory data from one episode"""
    obs = env.reset()
    data = {"time": [], "velocity": [], "heading_deg": [], "u": [], "vm": [], "r": []}

    t = 0
    dt = env.dt
    done = False

    while not done:
        u = env.ship.state["u"]
        vm = env.ship.state["vm"]
        r = env.ship.state["r"]
        psi = env.ship.state["psi"]
        velocity = np.sqrt(u**2 + vm**2)

        data["time"].append(t)
        data["velocity"].append(velocity)
        data["heading_deg"].append(np.rad2deg(psi))
        data["u"].append(u)
        data["vm"].append(vm)
        data["r"].append(r)

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action_output = agent.actor(obs_tensor)
            if isinstance(action_output, tuple):
                action = action_output[0].cpu().numpy()[0]
            else:
                action = action_output.cpu().numpy()[0]

        obs, _, done, info = env.step(action)
        t += dt

    return data, info["success"]


def main():
    checkpoint_path = "assets/checkpoints/demo_delivery/per_sac_otter_checkpoint_400.pth"
    output_path = "assets/results/demo_delivery/velocity_components_analysis.png"

    # Create environment
    env = OtterBerthingTrackingEnv(
        curriculum_level=2,
        enable_current=True,
        V_c_max=0.075,
        current_mode="fixed",
        enable_wind=True,
        V_w_max=1.2,
        wind_mode="fixed",
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

    # Collect successful trajectory
    print("Collecting trajectory...")
    max_attempts = 100
    for attempt in range(max_attempts):
        data, success = collect_trajectory(env, agent)
        if success:
            print(f"✓ Success on attempt {attempt+1}")
            break
    else:
        raise RuntimeError(f"Failed to find successful trajectory after {max_attempts} attempts")

    # Create plot with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    # u (surge velocity) plot
    ax1.plot(data["time"], data["u"], 'b-', linewidth=2, label='u (Surge)')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('u (m/s)', fontsize=12)
    ax1.set_title('Surge Velocity (u) During Berthing', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)

    # v (sway velocity) plot
    ax2.plot(data["time"], data["vm"], 'g-', linewidth=2, label='v (Sway)')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('v (m/s)', fontsize=12)
    ax2.set_title('Sway Velocity (v) During Berthing', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)

    # r (yaw rate) plot
    ax3.plot(data["time"], data["r"], 'r-', linewidth=2, label='r (Yaw Rate)')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('r (rad/s)', fontsize=12)
    ax3.set_title('Yaw Rate (r) During Berthing', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
