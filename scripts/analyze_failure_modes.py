import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.vectorized_env import VectorizedOtterEnv
from src.train.train_vectorized import VectorizedSacAgent, DEVICE


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze failure modes for berthing success conditions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes to evaluate")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel envs")
    parser.add_argument("--curriculum_level", type=int, default=2, help="Curriculum level")

    parser.add_argument("--enable_current", action="store_true", default=True)
    parser.add_argument("--no_current", action="store_false", dest="enable_current")
    parser.add_argument("--V_c_max", type=float, default=0.1)
    parser.add_argument("--current_mode", type=str, default="curriculum")

    parser.add_argument("--enable_wind", action="store_true", default=True)
    parser.add_argument("--no_wind", action="store_false", dest="enable_wind")
    parser.add_argument("--V_w_max", type=float, default=1.5)
    parser.add_argument("--wind_mode", type=str, default="curriculum")

    parser.add_argument(
        "--output",
        type=str,
        default="assets/results/results_failure_analysis/analysis_level2.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def mean(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main():
    args = parse_args()

    env = VectorizedOtterEnv(
        num_envs=args.num_envs,
        device=DEVICE,
        curriculum_level=args.curriculum_level,
        enable_current=args.enable_current,
        V_c_max=args.V_c_max,
        current_mode=args.current_mode,
        enable_wind=args.enable_wind,
        V_w_max=args.V_w_max,
        wind_mode=args.wind_mode,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = VectorizedSacAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_envs=args.num_envs,
        checkpoint_dir=str(Path(args.checkpoint).parent),
    )
    agent.load_models(args.checkpoint)

    obs = env.reset()

    total_episodes = 0
    success_count = 0
    fail_count = 0

    fail_counts = {"position": 0, "velocity": 0, "heading": 0, "yaw_rate": 0}
    fail_values = {"position": [], "velocity": [], "heading_deg": [], "yaw_rate": []}

    timeout_count = 0
    out_of_bounds_count = 0

    while total_episodes < args.episodes:
        actions = agent.select_action_batch(obs, evaluate=True)
        obs, _, dones, _ = env.step(actions)

        if dones.any():
            done_indices = torch.where(dones)[0]

            state = env.usv.get_state()
            u_all = state[:, 0]
            vm_all = state[:, 1]
            r_all = state[:, 2]
            x_all = state[:, 3]
            y_all = state[:, 4]
            psi_all = state[:, 5]

            u = u_all[done_indices]
            vm = vm_all[done_indices]
            r = r_all[done_indices]
            x = x_all[done_indices]
            y = y_all[done_indices]
            psi = psi_all[done_indices]

            dist = torch.sqrt((x - env.end_pos[0]) ** 2 + (y - env.end_pos[1]) ** 2)
            velocity = torch.sqrt(u ** 2 + vm ** 2)
            heading_error = torch.abs(env._normalize_angle(psi - env.end_heading))
            yaw_rate = torch.abs(r)

            success = env.episode_success[done_indices]
            failed = ~success

            pos_fail = dist > env.position_tolerance
            vel_fail = velocity > env.velocity_tolerance
            heading_fail = heading_error > env.heading_tolerance
            yaw_fail = yaw_rate > env.yaw_rate_tolerance

            failed_pos = pos_fail & failed
            failed_vel = vel_fail & failed
            failed_heading = heading_fail & failed
            failed_yaw = yaw_fail & failed

            fail_counts["position"] += int(failed_pos.sum().item())
            fail_counts["velocity"] += int(failed_vel.sum().item())
            fail_counts["heading"] += int(failed_heading.sum().item())
            fail_counts["yaw_rate"] += int(failed_yaw.sum().item())

            fail_values["position"].extend(dist[failed_pos].detach().cpu().tolist())
            fail_values["velocity"].extend(velocity[failed_vel].detach().cpu().tolist())
            fail_values["heading_deg"].extend(
                (heading_error[failed_heading].detach().cpu() * 180.0 / np.pi).tolist()
            )
            fail_values["yaw_rate"].extend(yaw_rate[failed_yaw].detach().cpu().tolist())

            cte_full = env._calculate_cross_track_error(x_all, y_all)
            cte = cte_full[done_indices]
            timeout = env.current_step[done_indices] >= env.max_episode_steps
            out_of_bounds = cte > env.max_cross_track * 2

            timeout_count += int((timeout & failed).sum().item())
            out_of_bounds_count += int((out_of_bounds & failed).sum().item())

            total_episodes += int(done_indices.numel())
            success_count += int(success.sum().item())
            fail_count += int(failed.sum().item())

            reset_obs = env.reset(env_indices=done_indices)
            obs[done_indices] = reset_obs[done_indices]

    success_rate = success_count / total_episodes if total_episodes else 0.0
    fail_rate = fail_count / total_episodes if total_episodes else 0.0

    failure_ratios = {
        k: (fail_counts[k] / fail_count if fail_count else 0.0) for k in fail_counts
    }

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": total_episodes,
        "success_count": success_count,
        "success_rate": success_rate,
        "fail_count": fail_count,
        "fail_rate": fail_rate,
        "failure_ratios_among_failed": failure_ratios,
        "failure_counts": fail_counts,
        "failure_value_means": {
            "position": mean(fail_values["position"]),
            "velocity": mean(fail_values["velocity"]),
            "heading_deg": mean(fail_values["heading_deg"]),
            "yaw_rate": mean(fail_values["yaw_rate"]),
        },
        "termination_counts": {
            "timeout": timeout_count,
            "out_of_bounds": out_of_bounds_count,
        },
        "tolerances": {
            "position": env.position_tolerance,
            "velocity": env.velocity_tolerance,
            "heading_rad": env.heading_tolerance,
            "yaw_rate": env.yaw_rate_tolerance,
        },
        "config": {
            "curriculum_level": args.curriculum_level,
            "enable_current": args.enable_current,
            "V_c_max": env.V_c_max,
            "current_mode": args.current_mode,
            "enable_wind": args.enable_wind,
            "V_w_max": env.V_w_max,
            "wind_mode": args.wind_mode,
        },
        "checkpoint": args.checkpoint,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
