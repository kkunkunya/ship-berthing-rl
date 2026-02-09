"""
采集 SAC 轨迹数据（使用向量化环境）

关键：SAC 模型使用向量化环境（30维观测）训练，
必须使用 VectorizedOtterEnv 来采集轨迹，否则维度不匹配会导致错误。

Input: SAC checkpoint 文件
Output: JSON 文件（包含完整轨迹）
Pos: 轨迹数据采集系统（向量化版本）
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.vectorized_env import VectorizedOtterEnv
from src.train.train_vectorized import VectorizedSacAgent, DEVICE
from src.utils.disturbance_metrics import override_curriculum_disturbance_config


def parse_args():
    parser = argparse.ArgumentParser(description="采集 SAC 轨迹数据（向量化环境）")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--curriculum_level", type=int, default=2)
    parser.add_argument("--enable_current", action="store_true", default=True)
    parser.add_argument("--V_c_max", type=float, default=0.075)
    parser.add_argument("--current_mode", type=str, default="fixed")
    parser.add_argument("--enable_wind", action="store_true", default=True)
    parser.add_argument("--V_w_max", type=float, default=1.2)
    parser.add_argument("--wind_mode", type=str, default="fixed")
    parser.add_argument("--max_episode_steps", type=int, default=2000)
    parser.add_argument("--output", type=str, default="assets/results/baselines/sac_trajectories.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("SAC 轨迹数据采集（向量化环境）")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.episodes}")
    print(f"Current: V_c={args.V_c_max}, mode={args.current_mode}")
    print(f"Wind: V_w={args.V_w_max}, mode={args.wind_mode}")
    print("=" * 60)

    # 创建向量化环境
    env = VectorizedOtterEnv(
        num_envs=1,
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

    # 加载模型
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"观测空间: {state_dim}D, 动作空间: {action_dim}D")

    agent = VectorizedSacAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_envs=1,
        checkpoint_dir=str(Path(args.checkpoint).parent),
    )
    agent.load_models(args.checkpoint)
    print("✓ 模型加载成功\n")

    # 采集轨迹
    trajectories = []
    success_count = 0

    for ep in range(args.episodes):
        obs = env.reset()

        # 初始化轨迹记录
        trajectory = {
            "episode": ep,
            "success": False,
            "steps": 0,
            "states": {"x": [], "y": [], "psi": [], "u": [], "vm": [], "r": []},
            "actions": [],
            "disturbances": {
                "V_c": args.V_c_max if args.enable_current else 0.0,
                "beta_c": 0.0,  # fixed mode = 0
                "V_w": args.V_w_max if args.enable_wind else 0.0,
                "beta_w": 0.0,  # fixed mode = 0
            },
        }

        done = False
        step_count = 0

        while not done:
            # 选择动作
            action = agent.select_action_batch(obs, evaluate=True)

            # 获取当前状态
            state = env.usv.get_state()
            u = float(state[0, 0].item())
            vm = float(state[0, 1].item())
            r = float(state[0, 2].item())
            x = float(state[0, 3].item())
            y = float(state[0, 4].item())
            psi = float(state[0, 5].item())

            # 记录状态
            trajectory["states"]["x"].append(x)
            trajectory["states"]["y"].append(y)
            trajectory["states"]["psi"].append(psi)
            trajectory["states"]["u"].append(u)
            trajectory["states"]["vm"].append(vm)
            trajectory["states"]["r"].append(r)

            # 记录动作
            act = action.cpu().numpy()[0]
            trajectory["actions"].append([float(act[0]), float(act[1])])

            # 执行动作
            obs, _, dones, infos = env.step(action)
            step_count += 1

            if bool(dones[0].item()):
                done = True

        # 获取最终状态判断成功
        state = env.usv.get_state()
        x = float(state[0, 3].item())
        y = float(state[0, 4].item())
        psi = float(state[0, 5].item())
        u = float(state[0, 0].item())
        vm = float(state[0, 1].item())
        r = float(state[0, 2].item())

        # 记录最后一个状态点
        trajectory["states"]["x"].append(x)
        trajectory["states"]["y"].append(y)
        trajectory["states"]["psi"].append(psi)
        trajectory["states"]["u"].append(u)
        trajectory["states"]["vm"].append(vm)
        trajectory["states"]["r"].append(r)

        # 计算成功条件
        position_error = float(torch.sqrt((state[0, 3] - env.end_pos[0]) ** 2 + (state[0, 4] - env.end_pos[1]) ** 2).item())
        resultant_velocity = float(torch.sqrt(state[0, 0] ** 2 + state[0, 1] ** 2).item())
        heading_error = float(torch.abs(env._normalize_angle(state[0, 5] - env.end_heading)).item())
        yaw_rate = abs(float(state[0, 2].item()))

        success = (
            position_error < env.position_tolerance
            and resultant_velocity < env.velocity_tolerance
            and heading_error < env.heading_tolerance
            and yaw_rate < env.yaw_rate_tolerance
        )

        trajectory["success"] = success
        trajectory["steps"] = step_count
        trajectory["final_position_error"] = position_error
        trajectory["final_velocity"] = resultant_velocity

        if success:
            success_count += 1

        trajectories.append(trajectory)

        status = "✓ Success" if success else "✗ Failed"
        print(f"Episode {ep + 1:2d}/{args.episodes}: {status} (steps={step_count}, pos_err={position_error:.2f}m)")

    # 保存为 JSON
    output_data = {
        "algorithm": "SAC",
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "success_count": success_count,
        "success_rate": success_count / args.episodes,
        "config": {
            "curriculum_level": args.curriculum_level,
            "enable_current": args.enable_current,
            "V_c_max": args.V_c_max,
            "current_mode": args.current_mode,
            "enable_wind": args.enable_wind,
            "V_w_max": args.V_w_max,
            "wind_mode": args.wind_mode,
        },
        "trajectories": trajectories,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("采集完成")
    print("=" * 60)
    print(f"成功率: {success_count}/{args.episodes} ({success_count / args.episodes * 100:.1f}%)")
    print(f"输出文件: {output_path}")
    print(f"文件大小: {output_path.stat().st_size / 1024:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
