import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.baselines.metrics import compute_failure_flags, compute_success
from src.envs.ship_berthing_env import OtterBerthingTrackingEnv
from src.envs.vectorized_env import VectorizedOtterEnv
from src.train.train_vectorized import VectorizedSacAgent, DEVICE
from src.utils.disturbance_metrics import override_curriculum_disturbance_config


parser = argparse.ArgumentParser(description="Evaluate SAC policy")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--output", type=str, default="assets/results/baselines/sac_evaluation.json")
parser.add_argument("--curriculum_level", type=int, default=2)
parser.add_argument("--enable_current", action="store_true", default=True)
parser.add_argument("--V_c_max", type=float, default=0.075)
parser.add_argument("--current_mode", type=str, default="fixed")
parser.add_argument("--enable_wind", action="store_true", default=True)
parser.add_argument("--V_w_max", type=float, default=1.2)
parser.add_argument("--wind_mode", type=str, default="fixed")
parser.add_argument("--max_episode_steps", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--position_tolerance", type=float, default=None)
parser.add_argument("--heading_tolerance", type=float, default=None)
parser.add_argument("--velocity_tolerance", type=float, default=None)
parser.add_argument("--yaw_rate_tolerance", type=float, default=None)
parser.add_argument("--vectorized", action="store_true", default=True, help="Use VectorizedOtterEnv")
parser.add_argument("--gym", action="store_true", default=False, help="Use OtterBerthingTrackingEnv")
parser.add_argument("--num_envs", type=int, default=1, help="Vectorized env count (1 only)")

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

use_vectorized = args.vectorized and not args.gym

if use_vectorized:
    if args.num_envs != 1:
        raise ValueError("vectorized evaluation currently supports num_envs=1 only")

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
        max_episode_steps=args.max_episode_steps,
    )
    override_curriculum_disturbance_config(env, args.curriculum_level, args.V_c_max, args.V_w_max)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = VectorizedSacAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_envs=args.num_envs,
        checkpoint_dir=str(Path(args.checkpoint).parent),
    )
    agent.load_models(args.checkpoint)

    position_tolerance = (
        args.position_tolerance if args.position_tolerance is not None else env.position_tolerance
    )
    heading_tolerance = (
        args.heading_tolerance if args.heading_tolerance is not None else env.heading_tolerance
    )
    velocity_tolerance = (
        args.velocity_tolerance if args.velocity_tolerance is not None else env.velocity_tolerance
    )
    yaw_rate_tolerance = (
        args.yaw_rate_tolerance if args.yaw_rate_tolerance is not None else env.yaw_rate_tolerance
    )

    success_count = 0
    fail_count = 0
    fail_counts = {"position": 0, "velocity": 0, "heading": 0, "yaw_rate": 0}
    fail_values = {"position": [], "velocity": [], "heading_deg": [], "yaw_rate": []}
    timeout_count = 0
    out_of_bounds_count = 0
    episode_records = []
    best_attempt = None
    best_attempt_min_dist = float("inf")

    total_episodes = 0
    obs = env.reset()
    distance_trace = []
    progress_trace = []
    step_count = 0

    while total_episodes < args.episodes:
        action = agent.select_action_batch(obs, evaluate=True)
        obs, _, dones, infos = env.step(action)

        distance_trace.append(float(infos["distance_to_goal"][0]))
        progress_trace.append(float(infos["path_progress"][0]))
        step_count += 1

        if bool(dones[0].item()):
            state = env.usv.get_state()
            u = state[0, 0]
            vm = state[0, 1]
            r = state[0, 2]
            x = state[0, 3]
            y = state[0, 4]
            psi = state[0, 5]

            position_error = float(torch.sqrt((x - env.end_pos[0]) ** 2 + (y - env.end_pos[1]) ** 2).item())
            resultant_velocity = float(torch.sqrt(u ** 2 + vm ** 2).item())
            yaw_rate = abs(float(r.item()))
            heading_error = float(torch.abs(env._normalize_angle(psi - env.end_heading)).item())

            success = compute_success(
                position_error=position_error,
                heading_error=heading_error,
                velocity=resultant_velocity,
                yaw_rate=yaw_rate,
                position_tolerance=position_tolerance,
                heading_tolerance=heading_tolerance,
                velocity_tolerance=velocity_tolerance,
                yaw_rate_tolerance=yaw_rate_tolerance,
            )

            if success:
                success_count += 1
            else:
                fail_count += 1

            flags = compute_failure_flags(
                position_error=position_error,
                heading_error=heading_error,
                velocity=resultant_velocity,
                yaw_rate=yaw_rate,
                position_tolerance=position_tolerance,
                heading_tolerance=heading_tolerance,
                velocity_tolerance=velocity_tolerance,
                yaw_rate_tolerance=yaw_rate_tolerance,
            )

            if not success:
                for key, flag in flags.items():
                    if flag:
                        fail_counts[key] += 1
                if flags["position"]:
                    fail_values["position"].append(position_error)
                if flags["velocity"]:
                    fail_values["velocity"].append(resultant_velocity)
                if flags["heading"]:
                    fail_values["heading_deg"].append(np.rad2deg(heading_error))
                if flags["yaw_rate"]:
                    fail_values["yaw_rate"].append(yaw_rate)

            cross_track_error = float(infos["cross_track_error"][0])
            timeout = int(env.current_step[0].item()) >= env.max_episode_steps
            out_of_bounds = cross_track_error > env.max_cross_track * 2
            if timeout:
                timeout_count += 1
            if out_of_bounds:
                out_of_bounds_count += 1

            min_distance = min(distance_trace) if distance_trace else position_error
            episode_records.append(
                {
                    "success": success,
                    "steps": step_count,
                    "final_distance": position_error,
                    "min_distance": min_distance,
                    "final_heading_deg": float(np.rad2deg(heading_error)),
                    "final_velocity": resultant_velocity,
                    "final_yaw_rate": yaw_rate,
                    "final_cross_track": cross_track_error,
                    "path_progress": float(env.path_progress[0].item()),
                    "timeout": timeout,
                    "out_of_bounds": out_of_bounds,
                }
            )

            if min_distance < best_attempt_min_dist:
                best_attempt_min_dist = min_distance
                best_attempt = {
                    "success": success,
                    "min_distance": min_distance,
                    "final_distance": position_error,
                    "distance_trace": distance_trace,
                    "progress_trace": progress_trace,
                    "dt": env.dt,
                }

            total_episodes += 1
            obs = env.reset()
            distance_trace = []
            progress_trace = []
            step_count = 0

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
            "position": float(np.mean(fail_values["position"])) if fail_values["position"] else 0.0,
            "velocity": float(np.mean(fail_values["velocity"])) if fail_values["velocity"] else 0.0,
            "heading_deg": float(np.mean(fail_values["heading_deg"])) if fail_values["heading_deg"] else 0.0,
            "yaw_rate": float(np.mean(fail_values["yaw_rate"])) if fail_values["yaw_rate"] else 0.0,
        },
        "termination_counts": {
            "timeout": timeout_count,
            "out_of_bounds": out_of_bounds_count,
        },
        "tolerances": {
            "position": position_tolerance,
            "velocity": velocity_tolerance,
            "heading_rad": heading_tolerance,
            "yaw_rate": yaw_rate_tolerance,
        },
        "config": {
            "curriculum_level": args.curriculum_level,
            "enable_current": args.enable_current,
            "V_c_max": args.V_c_max,
            "current_mode": args.current_mode,
            "enable_wind": args.enable_wind,
            "V_w_max": args.V_w_max,
            "wind_mode": args.wind_mode,
        },
        "checkpoint": args.checkpoint,
        "episode_records": episode_records,
        "best_attempt": best_attempt,
    }
else:
    env = OtterBerthingTrackingEnv(
        curriculum_level=args.curriculum_level,
        enable_current=args.enable_current,
        V_c_max=args.V_c_max,
        current_mode=args.current_mode,
        enable_wind=args.enable_wind,
        V_w_max=args.V_w_max,
        wind_mode=args.wind_mode,
        max_episode_steps=args.max_episode_steps,
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

    position_tolerance = (
        args.position_tolerance if args.position_tolerance is not None else env.position_tolerance
    )
    heading_tolerance = (
        args.heading_tolerance if args.heading_tolerance is not None else env.heading_tolerance
    )
    velocity_tolerance = (
        args.velocity_tolerance if args.velocity_tolerance is not None else env.velocity_tolerance
    )
    yaw_rate_tolerance = (
        args.yaw_rate_tolerance if args.yaw_rate_tolerance is not None else env.yaw_rate_tolerance
    )

    success_count = 0
    fail_count = 0
    fail_counts = {"position": 0, "velocity": 0, "heading": 0, "yaw_rate": 0}
    fail_values = {"position": [], "velocity": [], "heading_deg": [], "yaw_rate": []}
    timeout_count = 0
    out_of_bounds_count = 0
    episode_records = []
    best_attempt = None
    best_attempt_min_dist = float("inf")

    for _ in range(args.episodes):
        obs = env.reset()
        done = False
        last_info = {}
        distance_trace = []
        progress_trace = []
        step_count = 0

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            action_tensor = agent.select_action_batch(obs_tensor, evaluate=True)
            action = action_tensor.cpu().numpy()[0]
            obs, _, done, info = env.step(action)
            last_info = info
            distance_trace.append(float(info.get("distance_to_goal", 0.0)))
            progress_trace.append(float(getattr(env, "path_progress", 0.0)))
            step_count += 1

        position_error = float(last_info.get("distance_to_goal", 0.0))
        resultant_velocity = float(last_info.get("resultant_velocity", 0.0))
        yaw_rate = abs(float(last_info.get("ship_yaw_rate", 0.0)))
        berth_heading = env.trajectory.end_heading
        heading_error = abs(env._normalize_angle(np.deg2rad(last_info.get("ship_heading", 0.0)) - berth_heading))

        success = compute_success(
            position_error=position_error,
            heading_error=heading_error,
            velocity=resultant_velocity,
            yaw_rate=yaw_rate,
            position_tolerance=position_tolerance,
            heading_tolerance=heading_tolerance,
            velocity_tolerance=velocity_tolerance,
            yaw_rate_tolerance=yaw_rate_tolerance,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

        flags = compute_failure_flags(
            position_error=position_error,
            heading_error=heading_error,
            velocity=resultant_velocity,
            yaw_rate=yaw_rate,
            position_tolerance=position_tolerance,
            heading_tolerance=heading_tolerance,
            velocity_tolerance=velocity_tolerance,
            yaw_rate_tolerance=yaw_rate_tolerance,
        )

        if not success:
            for key, flag in flags.items():
                if flag:
                    fail_counts[key] += 1
            if flags["position"]:
                fail_values["position"].append(position_error)
            if flags["velocity"]:
                fail_values["velocity"].append(resultant_velocity)
            if flags["heading"]:
                fail_values["heading_deg"].append(np.rad2deg(heading_error))
            if flags["yaw_rate"]:
                fail_values["yaw_rate"].append(yaw_rate)

        cross_track_error = float(last_info.get("cross_track_error", 0.0))
        timeout = env.current_step >= env.max_episode_steps
        out_of_bounds = cross_track_error > env.max_cross_track * 2
        if timeout:
            timeout_count += 1
        if out_of_bounds:
            out_of_bounds_count += 1

        min_distance = min(distance_trace) if distance_trace else position_error
        episode_records.append(
            {
                "success": success,
                "steps": step_count,
                "final_distance": position_error,
                "min_distance": min_distance,
                "final_heading_deg": float(np.rad2deg(heading_error)),
                "final_velocity": resultant_velocity,
                "final_yaw_rate": yaw_rate,
                "final_cross_track": cross_track_error,
                "path_progress": float(getattr(env, "path_progress", 0.0)),
                "timeout": timeout,
                "out_of_bounds": out_of_bounds,
            }
        )

        if min_distance < best_attempt_min_dist:
            best_attempt_min_dist = min_distance
            best_attempt = {
                "success": success,
                "min_distance": min_distance,
                "final_distance": position_error,
                "distance_trace": distance_trace,
                "progress_trace": progress_trace,
                "dt": env.dt,
            }

    total_episodes = success_count + fail_count
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
            "position": float(np.mean(fail_values["position"])) if fail_values["position"] else 0.0,
            "velocity": float(np.mean(fail_values["velocity"])) if fail_values["velocity"] else 0.0,
            "heading_deg": float(np.mean(fail_values["heading_deg"])) if fail_values["heading_deg"] else 0.0,
            "yaw_rate": float(np.mean(fail_values["yaw_rate"])) if fail_values["yaw_rate"] else 0.0,
        },
        "termination_counts": {
            "timeout": timeout_count,
            "out_of_bounds": out_of_bounds_count,
        },
        "tolerances": {
            "position": position_tolerance,
            "velocity": velocity_tolerance,
            "heading_rad": heading_tolerance,
            "yaw_rate": yaw_rate_tolerance,
        },
        "config": {
            "curriculum_level": args.curriculum_level,
            "enable_current": args.enable_current,
            "V_c_max": args.V_c_max,
            "current_mode": args.current_mode,
            "enable_wind": args.enable_wind,
            "V_w_max": args.V_w_max,
            "wind_mode": args.wind_mode,
        },
        "checkpoint": args.checkpoint,
        "episode_records": episode_records,
        "best_attempt": best_attempt,
    }

output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
