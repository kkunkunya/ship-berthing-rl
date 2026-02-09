<!-- Logo 区域 - 替换为你的项目 Logo -->
<div align="center">

# Ship Berthing RL

**基于 PER-SAC 强化学习算法的 Otter USV 无人水面艇自动靠泊控制系统，支持环境干扰建模与多基线算法对比**

<!-- 徽章区域 -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/kkunkunya/ship-berthing-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/kkunkunya/ship-berthing-rl/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)

</div>

<!-- 截图/GIF 展示区域 -->
<!-- ![Demo](assets/demo.gif) -->

## 📋 目录

- [功能特性](#-功能特性)
- [快速开始](#-快速开始)
- [安装](#-安装)
- [使用方法](#-使用方法)
- [项目结构](#-项目结构)
- [贡献](#-贡献)
- [许可证](#-许可证)

## ✨ 功能特性

- PER-SAC 自研算法（优先经验回放 + SAC）
- Otter USV 双螺旋桨物理模型
- 环境干扰建模（风/浪/流）
- Gymnasium 标准环境接口
- 向量化并行训练
- 多基线算法对比（PPO/TD3/SAC）
- 丰富的可视化脚本（轨迹图/雷达图/收敛曲线）

## 🚀 快速开始

```bash
# 克隆并安装
git clone https://github.com/kkunkunya/ship-berthing-rl.git
cd ship-berthing-rl && uv sync

# 训练 PER-SAC 算法
uv run python -m src.train.train

# 评估训练结果
uv run python scripts/eval_sac_gym.py
```

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/kkunkunya/ship-berthing-rl.git
cd ship-berthing-rl

# 使用 uv 安装依赖（推荐）
uv sync

# 或使用 pip
pip install -e .
```

## 📖 使用方法

```bash
# 训练 PER-SAC 算法
uv run python -m src.train.train

# 向量化并行训练（更快）
uv run python -m src.train.train_vectorized

# 训练基线算法（PPO/TD3）
uv run python scripts/train_baselines_sb3.py

# 评估并生成轨迹
uv run python scripts/collect_sac_trajectories.py

# 生成可视化图表
uv run python scripts/plot_trajectory_comparison.py
```

## 📁 项目结构

```
ship-berthing-rl/
├── src/                          # 核心源代码
│   ├── envs/                     # RL 环境（Gymnasium 接口）
│   │   ├── ship_berthing_env.py  # Otter USV 靠泊环境
│   │   └── vectorized_env.py     # 向量化并行环境
│   ├── physics/                  # 物理模型
│   │   ├── otter.py              # Otter USV 动力学
│   │   └── otter_torch.py        # PyTorch 版本
│   ├── train/                    # 训练模块
│   ├── baselines/                # 基线算法（PPO/TD3）
│   └── utils/                    # 工具（PER Buffer 等）
├── scripts/                      # 训练/评估/可视化脚本
├── docs/                         # 项目文档
├── assets/                       # 资源文件
│   ├── checkpoints/              # 模型检查点
│   └── results/                  # 实验结果
└── pyproject.toml                # 项目配置（uv）
```

## 🤝 贡献

欢迎贡献！请查看 [贡献指南](CONTRIBUTING.md) 了解详情。

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。
