# 项目结构与模块索引

## 1. 总览

本项目以 `SAC-otter10.5` 为核心工程目录，训练主流程分为两条：

- **原版环境训练**：`src/train/train.py` → `src/envs/ship_berthing_env.py` → `src/physics/otter.py`
- **向量化/GPU训练**：`src/train/train_vectorized.py` → `src/envs/vectorized_env.py` → `src/physics/otter_torch.py`

辅助脚本位于 `scripts/`，训练产物位于 `assets/`，运行日志位于 `log/`。

## 2. 目录结构（按职责）

```
SAC-otter10.5/
  src/
    envs/          # 环境封装
    physics/       # 船舶动力学
    train/         # 训练入口
    utils/         # 公共工具
  scripts/         # 评估脚本
  assets/
    results/       # 训练输出
    checkpoints/   # 模型权重
  docs/             # 项目规格/日志/索引
  log/              # 运行日志
```

## 3. 模块说明（文件级）

### 3.1 物理模型（physics）

- `src/physics/otter.py`
  - 原版 Otter USV 6DOF 动力学（含海流/风干扰）
- `src/physics/otter_torch.py`
  - GPU 向量化动力学实现（PyTorch）
- `src/physics/otter.m`
  - MATLAB 参考模型

### 3.2 环境封装（envs）

- `src/envs/ship_berthing_env.py`
  - 单环境训练封装（原版）
- `src/envs/vectorized_env.py`
  - 多环境向量化封装（GPU 训练主用）

### 3.3 训练入口（train）

- `src/train/train.py`
  - 原版训练入口（非向量化）
- `src/train/train_vectorized.py`
  - 向量化训练入口（当前主训练）

### 3.4 工具（utils）

- `src/utils/per_buffer.py`
  - PER 经验回放（含 GPU replay）
- `src/utils/generate_path.py`
  - 轨迹/路径生成

### 3.5 脚本（scripts）

- `scripts/analyze_failure_modes.py`
  - 失败原因统计（位置/速度/航向/偏航角速度）

## 4. 关键调用关系（简图）

```
train_vectorized.py
  -> vectorized_env.py
     -> otter_torch.py
  -> per_buffer.py

train.py
  -> ship_berthing_env.py
     -> otter.py

analyze_failure_modes.py
  -> vectorized_env.py
  -> train_vectorized.py (VectorizedSacAgent)
```

## 5. 当前关键产物

- 训练产物已清空（按重新启动方案A要求）
- 新产物将生成于：
  - `assets/results/`
  - `assets/checkpoints/`
  - `log/`（命令输出 .txt）
