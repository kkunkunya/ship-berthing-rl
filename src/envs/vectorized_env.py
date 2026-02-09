"""
Otter USV 向量化环境

将 ship_berthing_env.py 转换为 PyTorch 批量版本，
支持同时运行 N 个环境实例。

核心改动：
1. 使用 otter_torch.py 的向量化物理仿真
2. 批量计算观测、奖励、done
3. 保持与原环境兼容的接口
"""

import torch
import numpy as np
import math
from typing import Tuple, Optional, Dict, List

from ..physics.otter_torch import OtterTorchUSV, DEVICE, DTYPE
from ..utils.generate_path import ExponentialTrajectoryGenerator


class VectorizedOtterEnv:
    """
    向量化 Otter USV 靠泊环境

    同时管理 N 个环境实例，所有计算在 GPU 上并行执行
    """

    def __init__(
        self,
        num_envs: int = 64,
        device: torch.device = DEVICE,
        max_episode_steps: int = 2000,
        dt: float = 0.1,
        initial_velocity: float = 1.5,
        # 路径参数
        kc: float = 0.45,
        DockingLine_x: float = 70.0,
        DockingLine_y: float = 95.0,
        turnpoint: float = 35.0,
        safe_distance: float = 60.0,
        target_ratio: float = 0.95,
        # 课程学习
        curriculum_level: int = 1,
        # 干扰参数
        enable_current: bool = False,
        V_c_max: float = 0.3,
        current_mode: str = 'random',
        enable_wind: bool = False,
        V_w_max: float = 5.0,
        wind_mode: str = 'random',
        pos_scale: float = 1.0,
    ):
        self.num_envs = num_envs
        self.device = device
        self.max_episode_steps = max_episode_steps
        self.dt = dt
        self.initial_velocity = initial_velocity
        self.pos_scale = pos_scale

        # 路径参数
        self.kc = kc
        self.DockingLine_x = DockingLine_x
        self.DockingLine_y = DockingLine_y
        self.turnpoint = turnpoint
        self.safe_distance = safe_distance
        self.target_ratio = target_ratio

        # 干扰配置
        self.enable_current = enable_current
        self.V_c_max = V_c_max
        self.current_mode = current_mode
        self.enable_wind = enable_wind
        self.V_w_max = V_w_max
        self.wind_mode = wind_mode

        # 课程学习配置
        self.curriculum_level = curriculum_level
        self.curriculum_current_config = {
            1: {'V_c_max': 0.0},
            2: {'V_c_max': 0.1},
            3: {'V_c_max': 0.2},
            4: {'V_c_max': 0.3},
        }
        self.curriculum_wind_config = {
            1: {'V_w_max': 0.0},
            2: {'V_w_max': 1.5},
            3: {'V_w_max': 3.0},
            4: {'V_w_max': 5.0},
        }
        self._setup_curriculum(curriculum_level)

        # 初始化轨迹生成器（共享，只需一份）
        self.trajectory = ExponentialTrajectoryGenerator(
            kc=kc,
            DockingLine_x=DockingLine_x,
            DockingLine_y=DockingLine_y,
            turnpoint=turnpoint,
            u_max=initial_velocity,
            safe_distance=safe_distance,
            target_ratio=target_ratio,
        )

        # 预计算轨迹点和参考值（转为 tensor）
        self._precompute_trajectory()

        # 初始化向量化 USV
        self.usv = OtterTorchUSV(batch_size=num_envs, device=device)

        # 动作和观测空间（gym 兼容）
        self.action_dim = 2
        self.obs_dim = 30  # 24 + 6 (干扰观测: V_c, sin/cos(beta_c), V_w, sin/cos(beta_w))
        self.max_n = self.usv.n_max
        self.min_n = self.usv.n_min

        # 动作平滑参数（与原版环境同步）
        self.max_n_rate = 20.0  # 最大转速变化率
        self.final_phase_smooth_alpha = 0.05  # 最终阶段平滑系数

        # 计算初始转速（与原版环境同步）
        # 基于物理公式: n² = |Xu| * u / (2 * k_pos)
        g = 9.81
        Umax = 3.0
        k_pos = 0.02216 / 2
        Xu = -24.4 * g / Umax
        n_squared = abs(Xu) * self.initial_velocity / (2 * k_pos)
        self.initial_n = math.sqrt(max(0, n_squared))
        self.initial_n_normalized = min(self.initial_n / self.max_n, 1.0)

        # Gym-style spaces (用于兼容训练脚本)
        class SimpleSpace:
            """简单的空间定义，用于兼容 gym 接口"""
            def __init__(self, shape):
                self.shape = shape

        self.observation_space = SimpleSpace((self.obs_dim,))
        self.action_space = SimpleSpace((self.action_dim,))

        # 区域阈值
        self.coasting_distance = self.trajectory.arc_length * 0.15
        self.berthing_zone = self.trajectory.arc_length * 0.05
        self.final_approach_zone = self.trajectory.arc_length * 0.02
        self.max_cross_track = 3.0
        self.reward_clip_min = -100.0
        self.reward_clip_max = 100.0

        # 状态变量（批量）
        self._init_batch_state()

        print(f"[VectorizedOtterEnv] 初始化完成")
        print(f"  环境数量: {num_envs}")
        print(f"  设备: {device}")
        print(f"  轨迹弧长: {self.trajectory.arc_length:.2f}m")

    def _setup_curriculum(self, level: int):
        """设置课程学习级别 (与 ship_berthing_env.py 同步)"""
        self.curriculum_level = level
        config = {
            1: {'position_tolerance': 1.2, 'velocity_tolerance': 0.15, 'heading_tolerance': 0.25,
                'yaw_rate_tolerance': 0.1, 'max_cross_track': 5.0},  # Relaxed for initial training
            2: {'position_tolerance': 0.8, 'velocity_tolerance': 0.10, 'heading_tolerance': 0.175,
                'yaw_rate_tolerance': 0.05, 'max_cross_track': 4.0}, # Demo version: 0.8m/10° for realistic targets
            3: {'position_tolerance': 0.2, 'velocity_tolerance': 0.05, 'heading_tolerance': 0.08,
                'yaw_rate_tolerance': 0.03, 'max_cross_track': 2.0}, # Adjusted: 0.4 -> 0.2
            4: {'position_tolerance': 0.1, 'velocity_tolerance': 0.015, 'heading_tolerance': 0.035,
                'yaw_rate_tolerance': 0.015, 'max_cross_track': 1.0}
        }.get(level, {})

        for k, v in config.items():
            setattr(self, k, v)

        # 更新干扰上限
        if self.current_mode == 'curriculum':
            self.V_c_max = self.curriculum_current_config.get(level, {}).get('V_c_max', 0.0)
        if self.wind_mode == 'curriculum':
            self.V_w_max = self.curriculum_wind_config.get(level, {}).get('V_w_max', 0.0)

    def set_curriculum_level(self, level: int):
        """设置课程学习级别"""
        self._setup_curriculum(level)

    def set_pos_scale(self, scale: float):
        """设置位置误差缩放系数"""
        self.pos_scale = scale

    def _precompute_trajectory(self):
        """预计算轨迹参考值"""
        # 将轨迹点转为 tensor
        self.traj_points = torch.tensor(
            self.trajectory.points, device=self.device, dtype=DTYPE
        )  # (N_points, 2)

        self.traj_s_arr = torch.tensor(
            self.trajectory.s_arr, device=self.device, dtype=DTYPE
        )  # (N_points,)

        # 预计算参考值
        n_points = len(self.trajectory.s_arr)
        ref_x = np.zeros(n_points)
        ref_y = np.zeros(n_points)
        ref_psi = np.zeros(n_points)
        ref_velocity = np.zeros(n_points)
        ref_r = np.zeros(n_points)

        for i, s in enumerate(self.trajectory.s_arr):
            # 使用 get_ref_with_velocity 获取预计算的参考速度（与原版环境同步）
            x, y, psi, u_d = self.trajectory.get_ref_with_velocity(s)
            ref_x[i] = x
            ref_y[i] = y
            ref_psi[i] = psi
            ref_velocity[i] = u_d
            ref_r[i] = self.trajectory.get_target_yaw_rate(s)

        self.traj_ref_x = torch.tensor(ref_x, device=self.device, dtype=DTYPE)
        self.traj_ref_y = torch.tensor(ref_y, device=self.device, dtype=DTYPE)
        self.traj_ref_psi = torch.tensor(ref_psi, device=self.device, dtype=DTYPE)
        self.traj_ref_velocity = torch.tensor(ref_velocity, device=self.device, dtype=DTYPE)
        self.traj_ref_r = torch.tensor(ref_r, device=self.device, dtype=DTYPE)

        # 起点和终点
        self.start_pos = torch.tensor(self.trajectory.start_pos, device=self.device, dtype=DTYPE)
        self.start_heading = torch.tensor(self.trajectory.start_heading, device=self.device, dtype=DTYPE)
        self.end_pos = torch.tensor(self.trajectory.points[-1], device=self.device, dtype=DTYPE)
        self.end_heading = torch.tensor(self.trajectory.end_heading, device=self.device, dtype=DTYPE)

    def _init_batch_state(self):
        """初始化批量状态变量"""
        B = self.num_envs

        # 步数计数
        self.current_step = torch.zeros(B, device=self.device, dtype=torch.long)

        # 路径跟踪
        self.closest_path_idx = torch.zeros(B, device=self.device, dtype=torch.long)
        self.path_progress = torch.zeros(B, device=self.device, dtype=DTYPE)
        self.last_progress = torch.zeros(B, device=self.device, dtype=DTYPE)
        self.last_dist_to_goal = torch.zeros(B, device=self.device, dtype=DTYPE)

        # 上一步动作
        self.last_action = torch.zeros(B, 2, device=self.device, dtype=DTYPE)

        # 干扰参数（每个环境可能不同）
        self.V_c = torch.zeros(B, device=self.device, dtype=DTYPE)
        self.beta_c = torch.zeros(B, device=self.device, dtype=DTYPE)
        self.V_w = torch.zeros(B, device=self.device, dtype=DTYPE)
        self.beta_w = torch.zeros(B, device=self.device, dtype=DTYPE)

        # 靠泊阶段标志
        self.berthing_phase = torch.zeros(B, device=self.device, dtype=torch.bool)

        # 成功标志
        self.episode_success = torch.zeros(B, device=self.device, dtype=torch.bool)

    def _sample_disturbances(self, env_mask: Optional[torch.Tensor] = None):
        """采样干扰参数"""
        if env_mask is None:
            env_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        n_reset = env_mask.sum().item()

        if self.enable_current and self.V_c_max > 0:
            if self.current_mode == 'random' or self.current_mode == 'curriculum':
                self.V_c[env_mask] = torch.rand(n_reset, device=self.device) * self.V_c_max
                self.beta_c[env_mask] = torch.rand(n_reset, device=self.device) * 2 * math.pi
            elif self.current_mode == 'fixed':
                self.V_c[env_mask] = self.V_c_max
                self.beta_c[env_mask] = 0.0
        else:
            self.V_c[env_mask] = 0.0
            self.beta_c[env_mask] = 0.0

        if self.enable_wind and self.V_w_max > 0:
            if self.wind_mode == 'random' or self.wind_mode == 'curriculum':
                self.V_w[env_mask] = torch.rand(n_reset, device=self.device) * self.V_w_max
                self.beta_w[env_mask] = torch.rand(n_reset, device=self.device) * 2 * math.pi
            elif self.wind_mode == 'fixed':
                self.V_w[env_mask] = self.V_w_max
                self.beta_w[env_mask] = 0.0
        else:
            self.V_w[env_mask] = 0.0
            self.beta_w[env_mask] = 0.0

    def reset(self, env_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        重置环境

        Args:
            env_indices: 要重置的环境索引，None 表示全部重置

        Returns:
            (B, obs_dim) 观测
        """
        if env_indices is None:
            # 重置所有环境
            env_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

            # 重置 USV 状态
            initial_states = torch.zeros(self.num_envs, 6, device=self.device, dtype=DTYPE)
            initial_states[:, 0] = self.initial_velocity  # u
            initial_states[:, 1] = 0.0  # v
            initial_states[:, 2] = 0.0  # r
            initial_states[:, 3] = self.start_pos[0]  # x
            initial_states[:, 4] = self.start_pos[1]  # y
            initial_states[:, 5] = self.start_heading  # psi

            self.usv.reset(initial_states)

            # 重置状态变量
            self._init_batch_state()

            # 设置初始转速（与原版环境同步）
            self.last_action[:, :] = self.initial_n_normalized

            # 采样干扰
            self._sample_disturbances()

            # 初始化到终点距离
            start_dist = torch.norm(self.start_pos - self.end_pos)
            self.last_dist_to_goal = start_dist.expand(self.num_envs).clone()

        else:
            # 部分重置
            env_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            env_mask[env_indices] = True
            n_reset = env_mask.sum().item()

            # 重置选定环境的 USV 状态
            self.usv.full_state[env_mask, :] = 0.0
            self.usv.full_state[env_mask, 0] = self.initial_velocity
            self.usv.full_state[env_mask, 6] = self.start_pos[0]
            self.usv.full_state[env_mask, 7] = self.start_pos[1]
            self.usv.full_state[env_mask, 11] = self.start_heading

            # 重置状态变量
            self.current_step[env_mask] = 0
            self.closest_path_idx[env_mask] = 0
            self.path_progress[env_mask] = 0.0
            self.last_progress[env_mask] = 0.0
            self.last_action[env_mask] = self.initial_n_normalized  # 设置初始转速
            self.berthing_phase[env_mask] = False
            self.episode_success[env_mask] = False
            start_dist = torch.norm(self.start_pos - self.end_pos)
            self.last_dist_to_goal[env_mask] = start_dist

            # 采样干扰
            self._sample_disturbances(env_mask)

        return self._get_obs()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        执行一步

        Args:
            actions: (B, 2) 动作 [-1, 1]

        Returns:
            obs: (B, obs_dim) 观测
            rewards: (B,) 奖励
            dones: (B,) 是否结束
            infos: 字典
        """
        # =====================================================
        # 动作平滑处理（与原版 ship_berthing_env.py 同步）
        # =====================================================
        B = self.num_envs

        # 获取当前速度 u
        state = self.usv.get_state()  # (B, 6)
        u = state[:, 0]  # 纵向速度

        # 确定 alpha 系数（靠泊阶段使用更小的 alpha 使响应更平滑）
        alpha = torch.full((B,), 0.15, device=self.device, dtype=DTYPE)
        berthing_mask = self.berthing_phase
        alpha[berthing_mask] = 0.08
        low_speed_mask = berthing_mask & (u.abs() < 0.1)
        alpha[low_speed_mask] = self.final_phase_smooth_alpha

        # 非对称动作缩放：正动作用 max_n，负动作用 |min_n|
        target_n = torch.where(
            actions >= 0,
            actions * self.max_n,
            actions * abs(self.min_n)
        )  # (B, 2)

        # 当前转速（从归一化的 last_action 恢复）
        current_n = torch.where(
            self.last_action >= 0,
            self.last_action * self.max_n,
            self.last_action * abs(self.min_n)
        )  # (B, 2)

        # Alpha 滤波平滑
        alpha_2d = alpha.unsqueeze(1)  # (B, 1)
        n_smoothed = current_n + alpha_2d * (target_n - current_n)  # (B, 2)

        # 变化率限制 (解除靠泊阶段的限制，允许快速响应干扰)
        rate_factor = torch.where(
            berthing_mask & (u.abs() < 0.1),
            torch.tensor(0.6, device=self.device),  # 原为 0.3，现改为 0.6 以平衡抗扰能力与稳定性
            torch.tensor(1.0, device=self.device)
        )  # (B,)
        rate_limit = self.max_n_rate * self.dt * rate_factor.unsqueeze(1)  # (B, 1)

        delta_n = n_smoothed - current_n
        delta_n_clipped = torch.clamp(delta_n, -rate_limit, rate_limit)
        n = current_n + delta_n_clipped  # (B, 2)

        # 最终裁剪到有效范围
        n = torch.clamp(n, self.min_n, self.max_n)

        # 更新 last_action（归一化存储）
        self.last_action = torch.where(
            n >= 0,
            n / self.max_n,
            n / abs(self.min_n)
        )

        # 物理仿真
        self.usv.step(
            n, dt=self.dt,
            V_c=self.V_c, beta_c=self.beta_c,
            V_w=self.V_w, beta_w=self.beta_w
        )

        # 更新路径跟踪
        self._update_path_progress()

        # 计算观测
        obs = self._get_obs()

        # 计算奖励
        rewards = self._compute_reward(actions)

        # 检查终止条件
        dones = self._check_done()

        # 更新状态
        self.current_step += 1

        # 构建 info
        infos = self._get_info()

        return obs, rewards, dones, infos

    def _get_obs(self) -> torch.Tensor:
        """
        获取批量观测

        Returns:
            (B, 30) 观测向量
            - 0-23: 原始状态观测
            - 24-26: 海流观测 (V_c_norm, sin(beta_c), cos(beta_c))
            - 27-29: 风观测 (V_w_norm, sin(beta_w), cos(beta_w))
        """
        B = self.num_envs
        state = self.usv.get_state()  # (B, 6) [u, v, r, x, y, psi]

        u = state[:, 0]
        vm = state[:, 1]
        r = state[:, 2]
        x = state[:, 3]
        y = state[:, 4]
        psi = state[:, 5]

        # 获取参考值
        idx = self.closest_path_idx.clamp(0, len(self.traj_ref_x) - 1)
        ref_x = self.traj_ref_x[idx]
        ref_y = self.traj_ref_y[idx]
        ref_psi = self.traj_ref_psi[idx]
        ref_velocity = self.traj_ref_velocity[idx]
        ref_r = self.traj_ref_r[idx]

        # 误差计算
        pos_error_x = x - ref_x
        pos_error_y = y - ref_y
        heading_error = self._normalize_angle(psi - ref_psi)
        cross_track_error = self._calculate_cross_track_error(x, y)

        # 到终点的距离和角度
        dist_to_goal = torch.sqrt((x - self.end_pos[0])**2 + (y - self.end_pos[1])**2)
        angle_to_goal = torch.atan2(self.end_pos[1] - y, self.end_pos[0] - x)
        relative_angle_to_goal = self._normalize_angle(angle_to_goal - psi)

        # 前瞻航向变化
        lookahead_idx = (idx + 10).clamp(0, len(self.traj_ref_psi) - 1)
        future_psi = self.traj_ref_psi[lookahead_idx]
        lookahead_heading_change = self._normalize_angle(future_psi - ref_psi)

        velocity_error = u - ref_velocity

        # 干扰观测（归一化到合理范围）
        # V_c 归一化：最大海流速度约 0.3 m/s
        V_c_norm = self.V_c / 0.3
        # V_w 归一化：最大风速约 5.0 m/s
        V_w_norm = self.V_w / 5.0

        # 组装观测 (30维)
        # 归一化因子
        # pos_scale = 1.0   # 1m -> 1.0 (Increased sensitivity from 0.2)
        dist_scale = 0.01 # 100m -> 1.0

        obs = torch.stack([
            pos_error_x * self.pos_scale,              # 0
            pos_error_y * self.pos_scale,              # 1
            cross_track_error,                    # 2
            torch.sin(psi),                       # 3
            torch.cos(psi),                       # 4
            heading_error,                        # 5
            u,                                    # 6
            vm,                                   # 7
            r,                                    # 8
            velocity_error,                       # 9
            ref_velocity,                         # 10
            torch.sin(ref_psi),                   # 11
            torch.cos(ref_psi),                   # 12
            lookahead_heading_change,             # 13
            dist_to_goal * dist_scale,            # 14
            torch.sin(relative_angle_to_goal),    # 15
            torch.cos(relative_angle_to_goal),    # 16
            self.path_progress,                   # 17
            self.last_action[:, 0],               # 18
            self.last_action[:, 1],               # 19
            (dist_to_goal < self.coasting_distance).float(),  # 20
            (dist_to_goal < self.berthing_zone).float(),      # 21
            self.berthing_phase.float(),          # 22
            ref_r,                                # 23
            # 新增干扰观测 (6维)
            V_c_norm,                             # 24: 归一化海流速度
            torch.sin(self.beta_c),               # 25: 海流方向正弦
            torch.cos(self.beta_c),               # 26: 海流方向余弦
            V_w_norm,                             # 27: 归一化风速
            torch.sin(self.beta_w),               # 28: 风向正弦
            torch.cos(self.beta_w),               # 29: 风向余弦
        ], dim=1)

        return obs

    def _get_reward_weights(self, dist_to_goal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """根据距终点距离计算奖励权重"""
        in_coasting = dist_to_goal <= self.coasting_distance
        in_berthing = dist_to_goal <= self.berthing_zone
        in_final = dist_to_goal <= self.final_approach_zone
        far_zone = ~in_coasting

        w_track = torch.ones_like(dist_to_goal)
        w_speed = torch.ones_like(dist_to_goal) * 1.5
        w_smooth = torch.ones_like(dist_to_goal) * 0.3
        w_safe = torch.ones_like(dist_to_goal)
        w_progress = torch.ones_like(dist_to_goal) * 2.0
        w_yaw = torch.ones_like(dist_to_goal)
        w_goal_progress = torch.ones_like(dist_to_goal)
        w_goal_heading = torch.ones_like(dist_to_goal)

        w_track = torch.where(far_zone, torch.full_like(w_track, 0.6), w_track)
        w_yaw = torch.where(far_zone, torch.full_like(w_yaw, 0.6), w_yaw)
        w_goal_progress = torch.where(far_zone, torch.full_like(w_goal_progress, 1.8), w_goal_progress)
        w_goal_heading = torch.where(far_zone, torch.full_like(w_goal_heading, 2.0), w_goal_heading)

        w_track = torch.where(in_coasting & ~in_berthing, torch.full_like(w_track, 1.5), w_track)
        w_smooth = torch.where(in_coasting & ~in_berthing, torch.full_like(w_smooth, 0.5), w_smooth)
        w_safe = torch.where(in_coasting & ~in_berthing, torch.full_like(w_safe, 1.5), w_safe)
        w_progress = torch.where(in_coasting & ~in_berthing, torch.full_like(w_progress, 1.0), w_progress)
        w_yaw = torch.where(in_coasting & ~in_berthing, torch.full_like(w_yaw, 1.5), w_yaw)
        w_goal_progress = torch.where(in_coasting & ~in_berthing, torch.full_like(w_goal_progress, 1.2), w_goal_progress)
        w_goal_heading = torch.where(in_coasting & ~in_berthing, torch.full_like(w_goal_heading, 1.2), w_goal_heading)

        w_track = torch.where(in_berthing, torch.full_like(w_track, 2.0), w_track)
        w_speed = torch.where(in_berthing, torch.full_like(w_speed, 1.0), w_speed)
        w_smooth = torch.where(in_berthing, torch.full_like(w_smooth, 2.0), w_smooth)  # 原为 2.5，恢复到 2.0 以抑制震荡
        w_safe = torch.where(in_berthing, torch.full_like(w_safe, 2.0), w_safe)
        w_progress = torch.where(in_berthing, torch.full_like(w_progress, 0.3), w_progress)
        w_goal_progress = torch.where(in_berthing, torch.full_like(w_goal_progress, 0.8), w_goal_progress)
        w_goal_heading = torch.where(in_berthing, torch.full_like(w_goal_heading, 0.6), w_goal_heading)

        w_track = torch.where(in_final, torch.full_like(w_track, 3.0), w_track)
        w_yaw = torch.where(in_final, torch.full_like(w_yaw, 2.0), w_yaw)
        w_goal_progress = torch.where(in_final, torch.full_like(w_goal_progress, 0.3), w_goal_progress)
        w_goal_heading = torch.where(in_final, torch.full_like(w_goal_heading, 0.3), w_goal_heading)

        return {
            "w_track": w_track,
            "w_speed": w_speed,
            "w_smooth": w_smooth,
            "w_safe": w_safe,
            "w_progress": w_progress,
            "w_yaw": w_yaw,
            "w_goal_progress": w_goal_progress,
            "w_goal_heading": w_goal_heading,
        }

    def _compute_goal_progress_reward(self, dist_to_goal: torch.Tensor) -> torch.Tensor:
        """按靠泊距离变化给予奖励"""
        delta = self.last_dist_to_goal - dist_to_goal
        reward = torch.where(
            delta > 0,
            15.0 * delta,
            -7.5 * torch.abs(delta)
        )
        self.last_dist_to_goal = dist_to_goal.clone()
        return reward

    def _compute_convergence_penalty(self, dist_to_goal: torch.Tensor) -> torch.Tensor:
        """靠泊收敛惩罚：距离越远惩罚越大，超步时追加惩罚"""
        dist_norm = dist_to_goal / (self.trajectory.arc_length + 1e-8)
        base_penalty = -3.0 * dist_norm

        timeout_mask = self.current_step >= (self.max_episode_steps - 1)
        timeout_penalty = torch.where(
            timeout_mask,
            -10.0 * dist_norm,
            torch.zeros_like(dist_norm)
        )

        return base_penalty + timeout_penalty

    def _compute_reward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        批量计算奖励

        Args:
            actions: (B, 2) 动作

        Returns:
            (B,) 奖励
        """
        state = self.usv.get_state()
        u = state[:, 0]
        vm = state[:, 1]
        r = state[:, 2]
        x = state[:, 3]
        y = state[:, 4]
        psi = state[:, 5]

        # 获取参考值
        idx = self.closest_path_idx.clamp(0, len(self.traj_ref_x) - 1)
        ref_x = self.traj_ref_x[idx]
        ref_y = self.traj_ref_y[idx]
        ref_psi = self.traj_ref_psi[idx]
        ref_velocity = self.traj_ref_velocity[idx]
        ref_r = self.traj_ref_r[idx]

        # 计算误差
        position_error = torch.sqrt((x - ref_x)**2 + (y - ref_y)**2)
        heading_error = torch.abs(self._normalize_angle(psi - ref_psi))
        cross_track_error = self._calculate_cross_track_error(x, y)
        dist_to_goal = torch.sqrt((x - self.end_pos[0])**2 + (y - self.end_pos[1])**2)
        angle_to_goal = torch.atan2(self.end_pos[1] - y, self.end_pos[0] - x)
        relative_angle_to_goal = self._normalize_angle(angle_to_goal - psi)

        # 轨迹跟踪奖励
        r_track = -2.0 * position_error - 3.0 * heading_error

        # 速度跟踪奖励
        speed_error = torch.abs(u - ref_velocity)
        r_speed = -5.0 * speed_error + 2.0 * (speed_error < 0.05).float()

        # 平滑奖励
        action_change = torch.norm(actions - self.last_action, dim=1)
        r_smooth = -0.5 * action_change

        # 安全奖励
        r_safe = torch.where(
            cross_track_error > 0.8,
            -15.0 * (cross_track_error - 0.8),
            torch.zeros_like(cross_track_error)
        )

        # 进度奖励
        progress_delta = self.path_progress - self.last_progress
        r_progress = torch.where(
            progress_delta > 0,
            50.0 * progress_delta,
            -20.0 * torch.abs(progress_delta)
        )
        self.last_progress = self.path_progress.clone()

        # 偏航角速度奖励
        r_yaw = -2.0 * torch.abs(r - ref_r)

        # 稀疏奖励
        resultant_velocity = torch.sqrt(u**2 + vm**2)
        success = self._check_berthing_success(x, y, psi, resultant_velocity, r)
        out_of_bounds = cross_track_error > self.max_cross_track

        r_sparse = torch.zeros_like(u)
        r_sparse = torch.where(success, torch.full_like(r_sparse, 100.0), r_sparse)
        r_sparse = torch.where(out_of_bounds, torch.full_like(r_sparse, -100.0), r_sparse)

        weights = self._get_reward_weights(dist_to_goal)
        r_goal = self._compute_goal_progress_reward(dist_to_goal)
        r_converge = self._compute_convergence_penalty(dist_to_goal)
        r_goal_heading = -torch.abs(relative_angle_to_goal)

        # 总奖励
        total_reward = (
            weights["w_track"] * r_track +
            weights["w_speed"] * r_speed +
            weights["w_smooth"] * r_smooth +
            weights["w_safe"] * r_safe +
            weights["w_progress"] * r_progress +
            weights["w_yaw"] * r_yaw +
            r_converge +
            weights["w_goal_progress"] * r_goal +
            weights["w_goal_heading"] * r_goal_heading +
            r_sparse
        )

        return torch.clamp(total_reward, self.reward_clip_min, self.reward_clip_max)

    def _update_path_progress(self):
        """更新路径进度 - 与原始环境完全一致的实现

        计算方式：
        1. 增量式搜索最近轨迹点
        2. 计算船舶在当前线段上的投影位置
        3. 使用弧长数组插值计算精确的 current_s
        4. path_progress = current_s / arc_length
        """
        state = self.usv.get_state()
        x = state[:, 3]  # (B,)
        y = state[:, 4]  # (B,)
        B = self.num_envs
        n_points = len(self.traj_ref_x)
        search_range = 50

        # 向量化搜索：为每个环境构建搜索索引矩阵
        offsets = torch.arange(search_range, device=self.device, dtype=torch.long)
        search_indices = self.closest_path_idx.unsqueeze(1) + offsets.unsqueeze(0)
        search_indices = search_indices.clamp(0, n_points - 1)

        # 获取搜索范围内的轨迹点坐标
        traj_x_search = self.traj_ref_x[search_indices]
        traj_y_search = self.traj_ref_y[search_indices]

        # 计算到各点的距离
        dist_sq = (x.unsqueeze(1) - traj_x_search)**2 + (y.unsqueeze(1) - traj_y_search)**2
        min_offset = dist_sq.argmin(dim=1)

        # 更新最近点索引
        self.closest_path_idx = (self.closest_path_idx + min_offset).clamp(0, n_points - 1)

        # === 精确弧长插值计算（与原始环境一致）===
        idx = self.closest_path_idx
        at_end = idx >= n_points - 1

        # 对于未到达终点的情况，计算线段内的精确位置
        idx_clamped = idx.clamp(0, n_points - 2)
        idx_next = (idx_clamped + 1).clamp(0, n_points - 1)

        # 获取线段端点
        p1_x = self.traj_ref_x[idx_clamped]
        p1_y = self.traj_ref_y[idx_clamped]
        p2_x = self.traj_ref_x[idx_next]
        p2_y = self.traj_ref_y[idx_next]

        # 线段向量
        seg_x = p2_x - p1_x
        seg_y = p2_y - p1_y
        seg_len_sq = seg_x**2 + seg_y**2
        seg_len = torch.sqrt(seg_len_sq + 1e-8)

        # 船舶到 p1 的向量
        ship_x = x - p1_x
        ship_y = y - p1_y

        # 在线段上的投影长度（限制在 [0, seg_len]）
        projection = (ship_x * seg_x + ship_y * seg_y) / (seg_len + 1e-8)
        projection = torch.clamp(projection, min=0.0)  # 下限为0
        projection = torch.min(projection, seg_len)    # 上限为 seg_len

        # 获取弧长数组
        s1 = self.traj_s_arr[idx_clamped]
        s2 = self.traj_s_arr[idx_next]

        # 插值计算当前弧长
        t = projection / (seg_len + 1e-8)
        current_s = s1 + (s2 - s1) * t

        # 计算进度
        arc_length = self.trajectory.arc_length
        progress_normal = current_s / arc_length

        # 到达终点的情况，进度设为 1.0
        self.path_progress = torch.where(at_end, torch.ones_like(progress_normal), progress_normal)

        # 更新靠泊阶段标志
        dist_to_goal = torch.sqrt((x - self.end_pos[0])**2 + (y - self.end_pos[1])**2)
        self.berthing_phase = dist_to_goal < self.berthing_zone

    def _check_done(self) -> torch.Tensor:
        """检查终止条件"""
        state = self.usv.get_state()
        x = state[:, 3]
        y = state[:, 4]

        cross_track_error = self._calculate_cross_track_error(x, y)

        # 终止条件 (边界 = max_cross_track * 2，与原版一致)
        timeout = self.current_step >= self.max_episode_steps
        out_of_bounds = cross_track_error > self.max_cross_track * 2
        success = self.episode_success

        return timeout | out_of_bounds | success

    def _check_berthing_success(
        self, x: torch.Tensor, y: torch.Tensor, psi: torch.Tensor,
        velocity: torch.Tensor, r: torch.Tensor
    ) -> torch.Tensor:
        """检查靠泊是否成功"""
        dist = torch.sqrt((x - self.end_pos[0])**2 + (y - self.end_pos[1])**2)
        heading_error = torch.abs(self._normalize_angle(psi - self.end_heading))

        success = (
            (dist < self.position_tolerance) &
            (velocity < self.velocity_tolerance) &
            (heading_error < self.heading_tolerance) &
            (torch.abs(r) < self.yaw_rate_tolerance)  # 偏航角速度容差
        )

        # 更新成功标志
        self.episode_success = self.episode_success | success

        return success

    def _calculate_cross_track_error(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """批量计算横向跟踪误差（与原始环境完全一致的实现）

        计算方式：
        1. 如果到达轨迹末端：计算垂直于终点航向的横向误差
        2. 否则：计算到当前轨迹线段的垂直距离
        """
        B = x.shape[0]
        n_points = len(self.traj_ref_x)

        idx = self.closest_path_idx.clamp(0, n_points - 1)

        # 判断是否到达轨迹末端
        at_end = idx >= n_points - 1

        # === 情况1: 到达轨迹末端 ===
        # 计算垂直于终点航向的横向误差
        dx_end = x - self.end_pos[0]
        dy_end = y - self.end_pos[1]
        # 横向误差 = 垂直于航向方向的分量
        # 航向方向 = (cos(psi), sin(psi))
        # 横向方向 = (-sin(psi), cos(psi))
        lateral_error_end = torch.abs(-torch.sin(self.end_heading) * dx_end +
                                       torch.cos(self.end_heading) * dy_end)

        # === 情况2: 正常轨迹跟踪 ===
        # 计算到当前线段的垂直距离
        idx_clamped = idx.clamp(0, n_points - 2)  # 确保有下一个点
        idx_next = (idx_clamped + 1).clamp(0, n_points - 1)

        # 线段起点 p1 和终点 p2
        p1_x = self.traj_ref_x[idx_clamped]
        p1_y = self.traj_ref_y[idx_clamped]
        p2_x = self.traj_ref_x[idx_next]
        p2_y = self.traj_ref_y[idx_next]

        # 线段向量
        seg_x = p2_x - p1_x
        seg_y = p2_y - p1_y
        seg_len_sq = seg_x**2 + seg_y**2

        # 船舶到 p1 的向量
        to_ship_x = x - p1_x
        to_ship_y = y - p1_y

        # 投影参数 t = (vec_to_ship · vec_segment) / |vec_segment|²
        # 限制在 [0, 1] 范围内
        t = (to_ship_x * seg_x + to_ship_y * seg_y) / (seg_len_sq + 1e-8)
        t = t.clamp(0, 1)

        # 最近点
        closest_x = p1_x + t * seg_x
        closest_y = p1_y + t * seg_y

        # 到最近点的距离
        cte_segment = torch.sqrt((x - closest_x)**2 + (y - closest_y)**2)

        # 根据是否到达末端选择计算方式
        cte = torch.where(at_end, lateral_error_end, cte_segment)

        return cte

    def _normalize_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """归一化角度到 [-pi, pi]"""
        return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi

    def _get_info(self) -> Dict:
        """获取信息字典"""
        state = self.usv.get_state()
        x = state[:, 3]
        y = state[:, 4]
        psi = state[:, 5]
        u = state[:, 0]
        vm = state[:, 1]
        r = state[:, 2]

        dist_to_goal = torch.sqrt((x - self.end_pos[0])**2 + (y - self.end_pos[1])**2)
        resultant_velocity = torch.sqrt(u**2 + vm**2)

        return {
            'position': (x.cpu().numpy(), y.cpu().numpy()),
            'heading': psi.cpu().numpy(),
            'velocity': u.cpu().numpy(),
            'distance_to_goal': dist_to_goal.cpu().numpy(),
            'path_progress': self.path_progress.cpu().numpy(),
            'cross_track_error': self._calculate_cross_track_error(x, y).cpu().numpy(),
            'success': self.episode_success.cpu().numpy(),
            'V_c': self.V_c.cpu().numpy(),
            'V_w': self.V_w.cpu().numpy(),
        }


# ==================== 测试代码 ====================

def test_vectorized_env():
    """测试向量化环境"""
    import time

    print("=" * 60)
    print("测试 VectorizedOtterEnv")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 64

    env = VectorizedOtterEnv(
        num_envs=num_envs,
        device=device,
        enable_current=True,
        enable_wind=True,
        current_mode='curriculum',
        wind_mode='curriculum',
        curriculum_level=2,
    )

    # 重置
    obs = env.reset()
    print(f"\n初始观测形状: {obs.shape}")
    print(f"观测范围: [{obs.min():.3f}, {obs.max():.3f}]")

    # 性能测试
    n_steps = 1000
    total_env_steps = num_envs * n_steps

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(n_steps):
        # 随机动作
        actions = torch.rand(num_envs, 2, device=device) * 2 - 1
        obs, rewards, dones, infos = env.step(actions)

        # 重置完成的环境
        if dones.any():
            done_indices = torch.where(dones)[0]
            env.reset(done_indices)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    steps_per_sec = total_env_steps / elapsed

    print(f"\n性能测试结果:")
    print(f"  总步数: {total_env_steps:,}")
    print(f"  耗时: {elapsed:.3f} 秒")
    print(f"  吞吐量: {steps_per_sec:,.0f} steps/sec")

    # 估算加速比
    original_estimate = 500  # 原版单环境估计
    speedup = steps_per_sec / original_estimate
    print(f"  估计加速比: {speedup:.1f}x")

    print(f"\n最终统计:")
    print(f"  成功: {infos['success'].sum()}")
    print(f"  平均进度: {infos['path_progress'].mean():.2%}")


if __name__ == "__main__":
    test_vectorized_env()
