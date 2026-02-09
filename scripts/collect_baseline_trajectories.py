"""
采集 PPO/TD3 轨迹数据并保存为 JSON

使用 Stable-Baselines3 加载 PPO/TD3 模型，采集完整的轨迹时间序列数据。

Input: PPO/TD3 checkpoint 文件（.zip）
Output: JSON 文件（包含 30 episodes 的完整轨迹）
Pos: 轨迹数据采集系统
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.envs.ship_berthing_env import OtterBerthingTrackingEnv
from stable_baselines3 import PPO, TD3


def parse_args():
    parser = argparse.ArgumentParser(description="采集 PPO/TD3 轨迹数据")
    parser.add_argument("--algorithm", type=str, required=True, choices=["ppo", "td3"],
                       help="算法类型")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径（.zip）")
    parser.add_argument("--episodes", type=int, default=30, help="采集的 episode 数量")
    parser.add_argument("--curriculum_level", type=int, default=2, help="课程等级")
    parser.add_argument("--enable_current", action="store_true", help="启用海流")
    parser.add_argument("--V_c_max", type=float, default=0.075, help="最大海流速度")
    parser.add_argument("--current_mode", type=str, default="fixed", help="海流模式")
    parser.add_argument("--enable_wind", action="store_true", help="启用风")
    parser.add_argument("--V_w_max", type=float, default=1.2, help="最大风速")
    parser.add_argument("--wind_mode", type=str, default="fixed", help="风模式")
    parser.add_argument("--output", type=str, required=True, help="输出 JSON 文件路径")
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print(f"{args.algorithm.upper()} 轨迹数据采集")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Curriculum Level: {args.curriculum_level}")
    print(f"Current: {args.enable_current} (V_c={args.V_c_max}, mode={args.current_mode})")
    print(f"Wind: {args.enable_wind} (V_w={args.V_w_max}, mode={args.wind_mode})")
    print(f"Output: {args.output}")
    print("="*60)

    # 创建环境
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

    # 加载模型
    if args.algorithm == "ppo":
        model = PPO.load(args.model)
    else:  # td3
        model = TD3.load(args.model)

    print(f"\n✓ 模型加载成功")
    print(f"  观测空间: {env.observation_space.shape[0]}D")
    print(f"  动作空间: {env.action_space.shape[0]}D")

    # 收集轨迹
    trajectories = []
    success_count = 0

    for ep in range(args.episodes):
        obs = env.reset()

        # 初始化轨迹记录
        trajectory = {
            "episode": ep,
            "success": False,
            "steps": 0,
            "states": {
                "x": [],
                "y": [],
                "psi": [],
                "u": [],
                "vm": [],
                "r": []
            },
            "actions": [],
            "disturbances": {
                "V_c": env.current_V_c if args.enable_current else 0.0,
                "beta_c": env.current_beta_c if args.enable_current else 0.0,
                "V_w": env.current_V_w if args.enable_wind else 0.0,
                "beta_w": env.current_beta_w if args.enable_wind else 0.0
            }
        }

        done = False
        step_count = 0

        while not done:
            # 选择动作（评估模式，deterministic=True）
            action, _ = model.predict(obs, deterministic=True)

            # 记录当前状态
            trajectory["states"]["x"].append(float(env.ship.state["x"]))
            trajectory["states"]["y"].append(float(env.ship.state["y"]))
            trajectory["states"]["psi"].append(float(env.ship.state["psi"]))
            trajectory["states"]["u"].append(float(env.ship.state["u"]))
            trajectory["states"]["vm"].append(float(env.ship.state["vm"]))
            trajectory["states"]["r"].append(float(env.ship.state["r"]))

            # 记录动作
            trajectory["actions"].append([float(action[0]), float(action[1])])

            # 执行动作
            obs, _, done, info = env.step(action)
            step_count += 1

        # 记录最终信息
        trajectory["success"] = info["success"]
        trajectory["steps"] = step_count

        if trajectory["success"]:
            success_count += 1

        trajectories.append(trajectory)

        # 打印进度
        status = "✓ Success" if trajectory["success"] else "✗ Failed"
        print(f"Episode {ep+1:2d}/{args.episodes}: {status} ({step_count:4d} steps)")

    # 保存为 JSON
    output_data = {
        "algorithm": args.algorithm.upper(),
        "model_path": args.model,
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
            "wind_mode": args.wind_mode
        },
        "trajectories": trajectories
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("采集完成")
    print("="*60)
    print(f"成功率: {success_count}/{args.episodes} ({success_count/args.episodes*100:.1f}%)")
    print(f"输出文件: {output_path}")
    print(f"文件大小: {output_path.stat().st_size / 1024:.1f} KB")
    print("="*60)


if __name__ == "__main__":
    main()
