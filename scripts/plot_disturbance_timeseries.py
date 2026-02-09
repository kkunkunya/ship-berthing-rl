import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.ship_berthing_env import OtterBerthingTrackingEnv
from src.train.train_vectorized import VectorizedSacAgent, DEVICE
from src.physics.otter import OtterUSVCore
from src.utils.disturbance_metrics import (
    compute_disturbance_series,
    override_curriculum_disturbance_config,
)
from src.utils.disturbance_forces import compute_combined_disturbance_tau


parser = argparse.ArgumentParser(description="Plot disturbance time series for a single trajectory")
parser.add_argument("--checkpoint", type=str, required=True, help="SAC checkpoint path")
parser.add_argument("--output_dir", type=str, default="assets/results/demo_delivery/figures")
parser.add_argument("--curriculum_level", type=int, default=2)
parser.add_argument("--enable_current", action="store_true", default=True)
parser.add_argument("--V_c_max", type=float, default=0.075)
parser.add_argument("--current_mode", type=str, default="fixed")
parser.add_argument("--enable_wind", action="store_true", default=True)
parser.add_argument("--V_w_max", type=float, default=1.2)
parser.add_argument("--wind_mode", type=str, default="fixed")
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

env = OtterBerthingTrackingEnv(
    curriculum_level=args.curriculum_level,
    enable_current=args.enable_current,
    V_c_max=args.V_c_max,
    current_mode=args.current_mode,
    enable_wind=args.enable_wind,
    V_w_max=args.V_w_max,
    wind_mode=args.wind_mode,
    verbose=False,
)
override_curriculum_disturbance_config(
    env,
    level=args.curriculum_level,
    V_c_max=args.V_c_max,
    V_w_max=args.V_w_max,
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = VectorizedSacAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    num_envs=1,
    checkpoint_dir=str(Path(args.checkpoint).parent),
)
agent.load_models(args.checkpoint)

obs = env.reset()
psi_series = []
tau_current_series = []
tau_wind_series = []
tau_combined_series = []
core = OtterUSVCore()
step = 0
done = False

while not done and step < args.max_steps:
    psi_series.append(env.ship.state["psi"])
    state_before = env.ship.full_state.copy()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
    action_tensor = agent.select_action_batch(obs_tensor, evaluate=True)
    action = action_tensor.cpu().numpy()[0]

    obs, _, done, _ = env.step(action)
    n_port = env.ship.last_n_port
    n_starboard = env.ship.last_n_starboard
    tau = compute_combined_disturbance_tau(
        core=core,
        x=state_before,
        n=np.array([n_port, n_starboard]),
        V_c=env.current_V_c,
        beta_c=env.current_beta_c,
        V_w=env.current_V_w,
        beta_w=env.current_beta_w,
    )
    tau_current_series.append(tau["tau_current"])
    tau_wind_series.append(tau["tau_wind"])
    tau_combined_series.append(tau["tau_combined"])
    step += 1

psi_series = np.asarray(psi_series, dtype=float)
time_series = np.arange(len(psi_series)) * env.dt
tau_current_series = np.asarray(tau_current_series, dtype=float)
tau_wind_series = np.asarray(tau_wind_series, dtype=float)
tau_combined_series = np.asarray(tau_combined_series, dtype=float)

disturbance_series = compute_disturbance_series(
    psi_series=psi_series,
    V_c=env.current_V_c,
    beta_c=env.current_beta_c,
    V_w=env.current_V_w,
    beta_w=env.current_beta_w,
)

data = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dt": env.dt,
    "steps": int(len(psi_series)),
    "V_c": env.current_V_c,
    "beta_c": env.current_beta_c,
    "V_w": env.current_V_w,
    "beta_w": env.current_beta_w,
    "u_c": disturbance_series["u_c"].tolist(),
    "v_c": disturbance_series["v_c"].tolist(),
    "u_w": disturbance_series["u_w"].tolist(),
    "v_w": disturbance_series["v_w"].tolist(),
    "tau_current": tau_current_series.tolist(),
    "tau_wind": tau_wind_series.tolist(),
    "tau_combined": tau_combined_series.tolist(),
}

json_path = output_dir / "disturbance_timeseries.json"
json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(time_series, tau_combined_series[:, 0], label=r"$\tau_u$ (Surge)", color="#1f77b4")
axes[0].set_ylabel("Force (N)")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="upper right")

axes[1].plot(time_series, tau_combined_series[:, 1], label=r"$\tau_v$ (Sway)", color="#ff7f0e")
axes[1].set_ylabel("Force (N)")
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc="upper right")

axes[2].plot(time_series, tau_combined_series[:, 5], label=r"$\tau_r$ (Yaw)", color="#2ca02c")
axes[2].set_ylabel("Moment (N*m)")
axes[2].set_xlabel("Time (s)")
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc="upper right")

title = "Combined Disturbance Forces (Wind + Current Equivalent)"
fig.suptitle(title, fontsize=14)

note = f"Vc={env.current_V_c:.3f} m/s, Vw={env.current_V_w:.3f} m/s"
fig.text(0.5, 0.92, note, ha="center", fontsize=10, color="gray")

output_path = output_dir / "disturbance_timeseries.png"
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(output_path, dpi=300)
plt.close(fig)

print(f"Saved: {output_path}")
print(f"Saved: {json_path}")
