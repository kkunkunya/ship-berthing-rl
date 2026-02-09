import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.noise import NormalActionNoise

from src.baselines.config import get_baseline_config
from src.baselines.wrappers import prepare_baseline_env
from src.envs.ship_berthing_env import OtterBerthingTrackingEnv


parser = argparse.ArgumentParser(description="Train PPO/TD3 baselines with stable-baselines3")
parser.add_argument("--algorithm", type=str, required=True, choices=["ppo", "td3"])
parser.add_argument("--output_dir", type=str, default="assets/checkpoints/baselines")
parser.add_argument("--total_timesteps", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--curriculum_level", type=int, default=2)
parser.add_argument("--enable_current", action="store_true", default=True)
parser.add_argument("--V_c_max", type=float, default=0.075)
parser.add_argument("--current_mode", type=str, default="fixed")
parser.add_argument("--enable_wind", action="store_true", default=True)
parser.add_argument("--V_w_max", type=float, default=1.2)
parser.add_argument("--wind_mode", type=str, default="fixed")
parser.add_argument("--max_episode_steps", type=int, default=2000)

args = parser.parse_args()

cfg = get_baseline_config(args.algorithm)
seed = args.seed if args.seed is not None else cfg.seed
total_timesteps = args.total_timesteps if args.total_timesteps is not None else cfg.train_timesteps

np.random.seed(seed)
torch.manual_seed(seed)

vec_env = VecMonitor(
    DummyVecEnv(
        [
            lambda: prepare_baseline_env(
                OtterBerthingTrackingEnv(
                    curriculum_level=args.curriculum_level,
                    enable_current=args.enable_current,
                    V_c_max=args.V_c_max,
                    current_mode=args.current_mode,
                    enable_wind=args.enable_wind,
                    V_w_max=args.V_w_max,
                    wind_mode=args.wind_mode,
                    max_episode_steps=args.max_episode_steps,
                    verbose=False,
                ),
                level=args.curriculum_level,
                V_c_max=args.V_c_max,
                V_w_max=args.V_w_max,
            )
        ]
    )
)

if args.algorithm == "ppo":
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        clip_range=cfg.clip_range,
        seed=seed,
        verbose=1,
    )
else:
    n_actions = vec_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3(
        "MlpPolicy",
        vec_env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        tau=cfg.tau,
        gamma=cfg.gamma,
        policy_delay=cfg.policy_delay,
        target_policy_noise=cfg.target_policy_noise,
        target_noise_clip=cfg.target_noise_clip,
        action_noise=action_noise,
        seed=seed,
        verbose=1,
    )

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(args.output_dir) / f"{args.algorithm}_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

model.learn(total_timesteps=total_timesteps)
model_path = output_dir / "model.zip"
model.save(model_path)

config_path = output_dir / "train_config.json"
config_path.write_text(
    json.dumps(
        {
            "algorithm": args.algorithm,
            "seed": seed,
            "total_timesteps": total_timesteps,
            "env": {
                "curriculum_level": args.curriculum_level,
                "enable_current": args.enable_current,
                "V_c_max": args.V_c_max,
                "current_mode": args.current_mode,
                "enable_wind": args.enable_wind,
                "V_w_max": args.V_w_max,
                "wind_mode": args.wind_mode,
                "max_episode_steps": args.max_episode_steps,
            },
            "hyperparams": cfg.__dict__,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(f"Saved model: {model_path}")
print(f"Saved config: {config_path}")
