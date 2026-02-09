# Otter USV 靠泊强化学习系统 - 技术指导文档

## 1. 核心模块架构

### 1.1 系统整体架构

本系统运用模块化的设计思路，各个模块之间借助标准接口来开展交互工作。从整体架构来看，可以划分为以下四个核心层次：

```
┌─────────────────────────────────────────────────────────────┐
│                      训练模块 (train/)                       │
│   SAC算法 │ 课程学习管理器 │ 熵自适应 │ 学习率调度           │
└─────────────────────────────────────────────────────────────┘
                              ↓ 调用
┌─────────────────────────────────────────────────────────────┐
│                      环境模块 (envs/)                        │
│   OtterBerthingTrackingEnv │ 观测空间 │ 动作空间 │ 奖励函数  │
└─────────────────────────────────────────────────────────────┘
                              ↓ 调用
┌─────────────────────────────────────────────────────────────┐
│                      物理模块 (physics/)                     │
│   OtterUSV_DualPropeller │ 6DOF动力学 │ 干扰力计算           │
└─────────────────────────────────────────────────────────────┘
                              ↓ 调用
┌─────────────────────────────────────────────────────────────┐
│                      工具模块 (utils/)                       │
│   ExponentialTrajectoryGenerator │ 路径规划 │ 速度规划       │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 环境模块 (src/envs/)

环境模块的核心类为 `OtterBerthingTrackingEnv`，该类继承自 OpenAI Gym 的标准接口 `gym.Env`。

**文件位置**: `src/envs/ship_berthing_env.py`

**观测空间设计** (30维连续状态):

| 维度 | 内容 | 说明 |
|------|------|------|
| 0-2 | 位置误差 | x误差、y误差、横向跟踪误差 |
| 3-5 | 航向信息 | sin(ψ)、cos(ψ)、航向误差 |
| 6-10 | 速度状态 | u、vm、r、速度误差、参考速度 |
| 11-13 | 参考航向 | sin(ψ_ref)、cos(ψ_ref)、前瞻航向变化 |
| 14-17 | 目标信息 | 距离、相对角度sin/cos、路径进度 |
| 18-19 | 上一动作 | 左舷/右舷螺旋桨归一化转速 |
| 20-23 | 阶段标志 | 滑行区、靠泊区、靠泊阶段、参考偏航角速度 |
| 24-29 | 干扰信息 | 海流速度/方向、风速/方向 |

**动作空间设计** (2维连续动作):
- 左舷螺旋桨转速: [-1, 1] → 映射到 [-101.8, 104.0] rad/s
- 右舷螺旋桨转速: [-1, 1] → 映射到 [-101.8, 104.0] rad/s

**课程学习机制**:

环境内置4级课程学习，通过逐步收紧容差来提升任务难度：

| 等级 | 位置容差 | 航向容差 | 速度容差 | 偏航角速度容差 |
|------|---------|---------|---------|--------------|
| 1 | 0.5 m | 10° | 0.08 m/s | 0.05 rad/s |
| 2 | 0.3 m | 7° | 0.05 m/s | 0.03 rad/s |
| 3 | 0.2 m | 5° | 0.03 m/s | 0.02 rad/s |
| 4 | 0.1 m | 2° | 0.015 m/s | 0.015 rad/s |

### 1.3 物理模块 (src/physics/)

物理模块负责船舶动力学仿真，核心类为 `OtterUSV_DualPropeller`。

**文件位置**: `src/physics/otter.py`

**模块结构**:
```
OtterUSVCore          # 核心动力学计算
    ├── dynamics()    # 状态导数计算
    └── 辅助函数
        ├── Smtrx()       # 反对称矩阵
        ├── Hmtrx()       # 6x6变换矩阵
        ├── Rzyx()        # ZYX欧拉角旋转矩阵
        ├── m2c()         # 附加质量转科氏矩阵
        ├── crossFlowDrag()  # 横流阻力
        └── windForce()   # 风力计算

OtterUSV_DualPropeller  # 包装类
    ├── reset()       # 状态重置
    ├── step()        # 单步仿真 (RK4积分)
    └── get_state_vector()  # 获取状态
```

### 1.4 轨迹生成模块 (src/utils/)

轨迹生成模块负责规划参考路径和速度曲线。

**文件位置**: `src/utils/generate_path.py`

**核心类**: `ExponentialTrajectoryGenerator`

**路径设计**:
- 直线段: 从起点到转向点，斜率为 kc
- 指数衰减段: 从转向点平滑过渡到泊位

**ODE求解流程**:
1. 定义耦合微分方程组
2. 使用 `scipy.integrate.solve_ivp` 求解
3. 计算派生变量（位置、航向、速度、偏航角速度）
4. 生成 `PathPoint` 列表供环境查询

### 1.5 训练模块 (src/train/)

训练模块实现了增强版 PER-SAC 算法。

**文件位置**: `src/train/train.py`

**网络架构**:
- Actor网络: 3层全连接 (256-256-256) + LayerNorm
- Critic网络: 双Q网络，3层全连接 (256-256-256) + LayerNorm

**增强特性**:
1. `CurriculumManager`: 课程学习管理器
2. `AdaptiveEntropyManager`: 自适应熵调整
3. `WarmupCosineScheduler`: 学习率调度
4. `SuccessTrajectoryBuffer`: 成功轨迹回放
5. `PrioritizedReplayBuffer`: 优先级经验回放

---

## 2. 公式说明

### 2.1 船舶动力学模型

#### 2.1.1 六自由度运动方程

船舶的运动可以借助刚体动力学方程来进行描述：

$$
M \dot{\nu} + C(\nu)\nu + D(\nu)\nu + g(\eta) = \tau + \tau_{env}
$$

其中：
- $\nu = [u, v, w, p, q, r]^T$ 表示船体坐标系下的速度向量
- $\eta = [x, y, z, \phi, \theta, \psi]^T$ 表示地球坐标系下的位置与姿态
- $M = M_{RB} + M_A$ 表示惯性矩阵（刚体质量 + 附加质量）
- $C(\nu)$ 表示科氏-向心力矩阵
- $D(\nu)$ 表示阻尼矩阵
- $g(\eta)$ 表示静水力恢复力
- $\tau$ 表示推进器产生的控制力
- $\tau_{env}$ 表示环境干扰力（海流 + 风）

#### 2.1.2 附加质量矩阵

附加质量矩阵 $M_A$ 为对角矩阵：

$$
M_A = -\text{diag}(X_{\dot{u}}, Y_{\dot{v}}, Z_{\dot{w}}, K_{\dot{p}}, M_{\dot{q}}, N_{\dot{r}})
$$

各分量的计算方式如下：
- $X_{\dot{u}} = -0.1 \cdot m$ （纵荡附加质量）
- $Y_{\dot{v}} = -1.5 \cdot m$ （横荡附加质量）
- $Z_{\dot{w}} = -1.0 \cdot m$ （垂荡附加质量）
- $K_{\dot{p}} = -0.2 \cdot I_{xx}$ （横摇附加惯性矩）
- $M_{\dot{q}} = -0.8 \cdot I_{yy}$ （纵摇附加惯性矩）
- $N_{\dot{r}} = -1.7 \cdot I_{zz}$ （艏摇附加惯性矩）

#### 2.1.3 横流阻力

横流阻力的计算公式为：

$$
\tau_{cf,v} = -\frac{1}{2} \rho C_d B T L |v_r| v_r
$$

$$
\tau_{cf,r} = -\frac{1}{2} \rho C_d B T \frac{L^2}{12} |r| r
$$

其中 $C_d = 1.0$ 为阻力系数，$\rho = 1025$ kg/m³ 为海水密度。

#### 2.1.4 螺旋桨推力

双螺旋桨推力模型：

$$
T_i = \begin{cases}
k_{pos} \cdot n_i |n_i| & \text{if } n_i > 0 \\
k_{neg} \cdot n_i |n_i| & \text{if } n_i \leq 0
\end{cases}
$$

其中：
- $k_{pos} = 0.02216 / 2$ （正转推力系数）
- $k_{neg} = 0.01289 / 2$ （反转推力系数）
- $n_i$ 为螺旋桨转速 (rad/s)

总推力与力矩：

$$
\tau_u = T_1 + T_2
$$

$$
\tau_r = -l_1 T_1 - l_2 T_2
$$

其中 $l_1 = -0.395$ m，$l_2 = 0.395$ m 为螺旋桨横向位置。

### 2.2 环境干扰模型

#### 2.2.1 海流力

海流对船舶的作用借助相对速度来进行计算：

$$
u_c = V_c \cos(\beta_c - \psi)
$$

$$
v_c = V_c \sin(\beta_c - \psi)
$$

$$
\nu_r = \nu - \nu_c
$$

其中：
- $V_c$ 为海流速度 (m/s)
- $\beta_c$ 为海流方向 (rad)
- $\psi$ 为船舶航向 (rad)

#### 2.2.2 风力模型

风力的计算公式为：

$$
\tau_{wind} = \frac{1}{2} \rho_{air} V_{rw}^2 \cdot C \cdot A
$$

具体分量：

$$
X_{wind} = q \cdot C_X \cdot A_{Fw}
$$

$$
Y_{wind} = q \cdot C_Y \cdot A_{Lw}
$$

$$
N_{wind} = q \cdot C_N \cdot A_{Lw} \cdot L
$$

其中：
- $q = \frac{1}{2} \rho_{air} V_{rw}^2$ 为动压
- $\rho_{air} = 1.225$ kg/m³ 为空气密度
- $A_{Fw} = B \cdot H$ 为正面投影面积
- $A_{Lw} = L \cdot H$ 为侧面投影面积
- $C_X = -0.9 \cos(\gamma_{rw})$ 为纵向力系数
- $C_Y = 0.95 \sin(\gamma_{rw})$ 为横向力系数
- $C_N = 0.20 \sin(\gamma_{rw}) \cos(\gamma_{rw})$ 为艏摇力矩系数

### 2.3 轨迹生成公式

#### 2.3.1 路径定义

路径由两段组成：

**直线段** ($\theta \leq \theta_{turn}$)：

$$
x = \theta, \quad y = k_c \cdot \theta
$$

**指数衰减段** ($\theta > \theta_{turn}$)：

$$
x = x_{dock} - k_d \cdot e^{-(\theta - \theta_{turn})/T_d}
$$

$$
y = \theta - \theta_{turn} + k_c \cdot \theta_{turn}
$$

其中：
- $k_c = 0.45$ 为直线段斜率
- $\theta_{turn} = 35$ 为转向点参数
- $k_d = x_{dock} - \theta_{turn}$
- $T_d = k_d \cdot k_c$

#### 2.3.2 速度规划 (Tanh连续函数)

期望速度运用 Tanh 函数来进行规划：

$$
u_d = u_{max} \cdot \tanh(k \cdot |y - y_{dock}|)
$$

其中：

$$
k = \frac{\text{arctanh}(\alpha)}{d_{safe}}
$$

- $u_{max} = 1.5$ m/s 为最大速度
- $\alpha = 0.95$ 为目标比例
- $d_{safe} = 60$ m 为安全距离

#### 2.3.3 航向计算

期望航向由路径切线方向来确定：

$$
\psi_d = \frac{\pi}{2} - \arctan2\left(\frac{dx}{d\theta}, \frac{dy}{d\theta}\right)
$$

#### 2.3.4 ODE耦合方程

轨迹生成借助求解耦合微分方程组来得以实现：

$$
\frac{d\theta}{dt} = \frac{u_d}{\sqrt{(dx/d\theta)^2 + (dy/d\theta)^2}} - w_s
$$

$$
\frac{dw_s}{dt} = -\zeta \cdot w_s
$$

其中 $\zeta = 0.2$ 为阻尼系数，$w_s$ 为速度设定误差函数。

### 2.4 奖励函数

奖励函数由多个分量加权组合而成：

$$
r_{total} = w_1 r_{track} + w_2 r_{speed} + w_3 r_{smooth} + w_4 r_{safe} + w_5 r_{progress} + w_6 r_{yaw} + r_{sparse}
$$

#### 2.4.1 轨迹跟踪奖励

$$
r_{track} = -2.0 \cdot e_{pos} - 3.0 \cdot e_{heading}
$$

其中：
- $e_{pos} = \sqrt{(x - x_{ref})^2 + (y - y_{ref})^2}$ 为位置误差
- $e_{heading} = |\psi - \psi_{ref}|$ 为航向误差

#### 2.4.2 速度跟踪奖励

$$
r_{speed} = \begin{cases}
-1.0 \cdot e_{speed} + 3.0 & \text{if 靠泊阶段且 } e_{speed} < 0.03 \\
-5.0 \cdot e_{speed} + 2.0 & \text{if } e_{speed} < 0.05 \\
-5.0 \cdot e_{speed} & \text{otherwise}
\end{cases}
$$

#### 2.4.3 动作平滑奖励

$$
r_{smooth} = \begin{cases}
2.0 & \text{if 靠泊阶段且 } \|\Delta a\| < 0.02 \\
-3.0 \cdot \|\Delta a\| & \text{if 靠泊阶段} \\
-0.5 \cdot \|\Delta a\| & \text{otherwise}
\end{cases}
$$

#### 2.4.4 稀疏奖励

$$
r_{sparse} = \begin{cases}
+100 & \text{if 成功靠泊} \\
-100 & \text{if 超出边界} \\
0 & \text{otherwise}
\end{cases}
$$

### 2.5 SAC算法核心公式

#### 2.5.1 策略损失

$$
J_\pi = \mathbb{E}_{s \sim D} \left[ \alpha \log \pi(a|s) - Q(s, a) \right]
$$

#### 2.5.2 Q值损失

$$
J_Q = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( Q(s,a) - (r + \gamma (Q_{target}(s', a') - \alpha \log \pi(a'|s'))) \right)^2 \right]
$$

#### 2.5.3 熵系数自动调整

$$
J_\alpha = \mathbb{E}_{a \sim \pi} \left[ -\alpha (\log \pi(a|s) + \bar{H}) \right]
$$

其中 $\bar{H}$ 为目标熵，通常设置为 $-\dim(A)$。

---

## 3. 参数说明

### 3.1 船舶物理参数

以下参数来源于 Otter USV 官方数据：

| 参数 | 符号 | 数值 | 单位 | 说明 |
|------|------|------|------|------|
| 船长 | $L$ | 2.0 | m | 船体总长度 |
| 船宽 | $B$ | 1.08 | m | 船体最大宽度 |
| 质量 | $m$ | 55.0 | kg | 无负载质量 |
| 吃水 | $T$ | 0.134 | m | 无负载吃水深度 |
| 最大速度 | $U_{max}$ | 3.0 | m/s | 设计最大航速 |
| 干舷高度 | $H$ | 0.3 | m | 水面以上高度 |
| 重心纵向位置 | $x_G$ | 0.2 | m | 相对于船中 |

### 3.2 螺旋桨参数

| 参数 | 数值 | 单位 | 说明 |
|------|------|------|------|
| 正转最大转速 | 993 (104.0) | RPM (rad/s) | 前进推力 |
| 反转最大转速 | 972 (101.8) | RPM (rad/s) | 后退推力 |
| 正转推力系数 | 0.01108 | - | $k_{pos} = 0.02216/2$ |
| 反转推力系数 | 0.00645 | - | $k_{neg} = 0.01289/2$ |
| 左舷横向位置 | -0.395 | m | 相对于中心线 |
| 右舷横向位置 | +0.395 | m | 相对于中心线 |

### 3.3 环境干扰参数

#### 3.3.1 海流配置

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `enable_current` | False | True/False | 是否启用海流 |
| `V_c_max` | 0.3 | 0~0.5 m/s | 海流速度上限 |
| `current_mode` | 'random' | random/fixed/curriculum | 海流模式 |

**模式说明**：
- `random`: 每个 episode 随机采样海流速度与方向
- `fixed`: 固定最大海流，方向为 0°（北向）
- `curriculum`: 根据课程等级逐步提升海流强度

#### 3.3.2 风力配置

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `enable_wind` | False | True/False | 是否启用风力 |
| `V_w_max` | 5.0 | 0~10 m/s | 风速上限 |
| `wind_mode` | 'random' | random/fixed/curriculum | 风力模式 |

#### 3.3.3 课程学习干扰配置

| 等级 | 海流速度 | 风速 |
|------|---------|------|
| 1 | 0.0 m/s | 0.0 m/s |
| 2 | 0.05 m/s | 0.5 m/s |
| 3 | 0.1 m/s | 1.5 m/s |
| 4 | 0.2 m/s | 3.0 m/s |

### 3.4 轨迹生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `kc` | 0.45 | 直线段斜率 |
| `DockingLine_x` | 70.0 m | 泊位 X 坐标 |
| `DockingLine_y` | 95.0 m | 泊位 Y 坐标 |
| `turnpoint` | 35.0 | 转向点参数 θ |
| `u_max` | 1.5 m/s | 最大期望速度 |
| `safe_distance` | 60.0 m | 安全距离（速度规划用） |
| `target_ratio` | 0.95 | Tanh 目标比例 |
| `zeta` | 0.2 | ODE 阻尼系数 |
| `ws_0` | 1.5 | 初始速度误差函数值 |
| `t_final` | 200.0 s | ODE 求解终止时间 |
| `dt` | 0.1 s | 时间步长 |

### 3.5 训练超参数

#### 3.5.1 SAC 算法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `actor_lr` | 1e-5 | Actor 网络学习率 |
| `critic_lr` | 1e-4 | Critic 网络学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `tau` | 0.005 | 目标网络软更新系数 |
| `batch_size` | 256 | 批量大小 |
| `buffer_size` | 1,000,000 | 经验回放缓冲区容量 |
| `target_entropy` | -2 | 目标熵（动作维度的负值） |

#### 3.5.2 课程学习参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `level_thresholds` | [0.5, 0.6, 0.7, 0.8] | 各等级升级所需成功率 |
| `window_size` | 100 | 成功率计算窗口 |
| `patience` | 200 | 升级前最小稳定 episode 数 |

#### 3.5.3 网络架构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fc1_dim` | 256 | 第一隐藏层维度 |
| `fc2_dim` | 256 | 第二隐藏层维度 |
| `fc3_dim` | 256 | 第三隐藏层维度 |
| 归一化 | LayerNorm | 各层归一化方式 |

### 3.6 环境参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_episode_steps` | 2000 | 单 episode 最大步数 |
| `dt` | 0.1 s | 仿真时间步长 |
| `initial_velocity` | 1.5 m/s | 初始纵向速度 |
| `curriculum_level` | 1 | 初始课程等级 |
| `max_n_rate` | 20.0 rad/s² | 螺旋桨转速变化率限制 |

---

## 4. 使用说明

### 4.1 环境搭建

#### 4.1.1 依赖安装

本项目运用 `uv` 来开展 Python 依赖管理工作：

```bash
# 安装 uv（如尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步项目依赖
cd /Users/kunkun/Project/2000-船强化学习
uv sync
```

#### 4.1.2 主要依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| torch | ≥2.0.0 | 深度学习框架 |
| gymnasium | ≥0.29.0 | 强化学习环境接口 |
| stable-baselines3 | ≥2.2.1 | PPO/TD3 基线算法 |
| scipy | ≥1.11.0 | ODE 求解器 |
| matplotlib | ≥3.7.0 | 可视化绑图 |
| numpy | ≥1.24.0 | 数值计算 |

### 4.2 训练流程

#### 4.2.1 SAC 训练

**基础训练命令**：

```bash
uv run python -m src.train.train \
  --episodes 2000 \
  --curriculum_level 1
```

**带干扰训练命令**：

```bash
uv run python -m src.train.train \
  --episodes 2000 \
  --enable_current \
  --V_c_max 0.075 \
  --current_mode fixed \
  --enable_wind \
  --V_w_max 1.2 \
  --wind_mode fixed
```

**向量化训练（GPU 加速）**：

```bash
uv run python -m src.train.train_vectorized \
  --num_envs 16 \
  --episodes 2000 \
  --enable_current \
  --V_c_max 0.075
```

#### 4.2.2 基线算法训练

**PPO 训练**：

```bash
uv run python scripts/train_baselines_sb3.py --algorithm ppo
```

**TD3 训练**：

```bash
uv run python scripts/train_baselines_sb3.py --algorithm td3
```

#### 4.2.3 训练输出

训练过程会生成以下文件：

| 路径 | 内容 |
|------|------|
| `assets/checkpoints/` | 模型检查点 (.pth) |
| `assets/results/` | 训练曲线数据 (.json) |
| `log/` | 训练日志 (.txt) |

### 4.3 评估流程

#### 4.3.1 SAC 模型评估

```bash
uv run python scripts/eval_sac_gym.py \
  --checkpoint assets/checkpoints/best_model.pth \
  --num_episodes 100 \
  --enable_current \
  --V_c_max 0.075 \
  --current_mode fixed
```

#### 4.3.2 基线模型评估

```bash
uv run python scripts/eval_baselines_sb3.py \
  --algorithm ppo \
  --checkpoint assets/checkpoints/baselines/ppo_model \
  --num_episodes 100
```

#### 4.3.3 失败模式分析

```bash
uv run python scripts/analyze_failure_modes.py \
  --checkpoint assets/checkpoints/best_model.pth
```

### 4.4 可视化流程

#### 4.4.1 轨迹采集

```bash
# 采集 SAC 轨迹
uv run python scripts/collect_sac_trajectories.py

# 采集基线轨迹
uv run python scripts/collect_baseline_trajectories.py
```

#### 4.4.2 生成对比图

```bash
# 轨迹对比图
uv run python scripts/plot_trajectory_comparison_v2.py

# 基线对比图（论文级）
uv run python scripts/plot_baseline_comparison_paper.py

# 干扰时序图
uv run python scripts/plot_disturbance_timeseries.py

# 性能雷达图
uv run python scripts/plot_performance_radar.py
```

#### 4.4.3 生成动画

```bash
uv run python scripts/generate_enhanced_animations.py
```

### 4.5 完整工作流示例

以下为从训练到评估的完整流程：

```bash
# 1. 环境搭建
cd /Users/kunkun/Project/2000-船强化学习
uv sync

# 2. 无干扰基线训练
uv run python -m src.train.train \
  --episodes 1000 \
  --curriculum_level 1

# 3. 带干扰训练
uv run python -m src.train.train \
  --episodes 2000 \
  --enable_current --V_c_max 0.075 --current_mode fixed \
  --enable_wind --V_w_max 1.2 --wind_mode fixed

# 4. 评估
uv run python scripts/eval_sac_gym.py \
  --checkpoint assets/checkpoints/best_model.pth \
  --num_episodes 100

# 5. 可视化
uv run python scripts/plot_trajectory_comparison_v2.py
```

### 4.6 关键文件索引

| 文件路径 | 功能说明 |
|---------|---------|
| `src/envs/ship_berthing_env.py` | Gym 环境定义 |
| `src/physics/otter.py` | 船舶动力学模型 |
| `src/utils/generate_path.py` | 轨迹生成器 |
| `src/train/train.py` | SAC 训练主程序 |
| `src/train/train_vectorized.py` | 向量化训练 |
| `src/baselines/config.py` | 基线算法配置 |
| `scripts/eval_sac_gym.py` | SAC 评估脚本 |
| `scripts/eval_baselines_sb3.py` | 基线评估脚本 |
| `scripts/plot_*.py` | 各类可视化脚本 |

---

## 附录：符号表

| 符号 | 含义 | 单位 |
|------|------|------|
| $u$ | 纵向速度 (surge) | m/s |
| $v$ | 横向速度 (sway) | m/s |
| $r$ | 偏航角速度 (yaw rate) | rad/s |
| $\psi$ | 航向角 (heading) | rad |
| $x, y$ | 位置坐标 | m |
| $n$ | 螺旋桨转速 | rad/s |
| $V_c$ | 海流速度 | m/s |
| $\beta_c$ | 海流方向 | rad |
| $V_w$ | 风速 | m/s |
| $\beta_w$ | 风向 | rad |
| $\tau$ | 力/力矩 | N / N·m |

