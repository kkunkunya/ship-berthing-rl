"""
Otter USV 靠泊轨迹跟踪环境 - 基于berthing_sim_smoothest.py

使用路径参数 (与berthing_sim_smoothest.py完全一致):
- kc = 0.45 (直线段斜率)
- DockingLine_x = 70 (泊位x坐标)
- DockingLine_y = 95 (泊位y坐标)
- turnpoint = 35 (转向点)
- safe_distance = 60 (安全距离)
- target_ratio = 0.95 (Tanh比例)
- zeta = 0.2 (ODE阻尼)
- ws_0 = 1.5 (初始速度误差)

速度规划: u_d = u_max * tanh(k * |y - DockingLine_y|)
航向: ψ_d = π/2 - arctan2(dx/dθ, dy/dθ)
偏航角速度: r = dψ/dt
"""

import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict, Any
from collections import deque

# 导入Otter USV双螺旋桨模型
from ..physics.otter import OtterUSV_DualPropeller
from ..utils.generate_path import ExponentialTrajectoryGenerator


class OtterBerthingTrackingEnv(gym.Env):
    """Otter USV靠泊轨迹跟踪环境 - berthing_sim_smoothest.py版本"""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 max_episode_steps=2000,
                 dt=0.1,
                 render_mode=None,
                 log_interval=100,
                 verbose=True,
                 training_phase=1,
                 # 速度规划
                 initial_velocity=1.5,
                 use_paper_velocity=True,
                 # 路径参数 (berthing_sim_smoothest.py)
                 kc=0.45,
                 DockingLine_x=70.0,
                 DockingLine_y=95.0,
                 turnpoint=35.0,
                 safe_distance=60.0,
                 target_ratio=0.95,
                 zeta=0.2,
                 ws_0=1.5,
                 t_final=200.0,
                 # 课程学习
                 curriculum_level=1,
                 # 控制稳定性
                 final_phase_smooth_alpha=0.05,
                 velocity_deadzone=0.02,
                 rpm_deadzone=0.03,
                 # ===== 海流干扰参数 =====
                 enable_current=False,
                 V_c_max=0.3,
                 current_mode='random',
                 # ===== 风干扰参数 =====
                 enable_wind=False,
                 V_w_max=5.0,
                 wind_mode='random'):
        """
        Args:
            ...（原有参数）...
            enable_current: 是否启用海流干扰
            V_c_max: 海流速度上限 (m/s)
            current_mode: 海流模式 - 'random'(随机), 'fixed'(固定最大), 'curriculum'(课程学习)
            enable_wind: 是否启用风干扰
            V_w_max: 风速上限 (m/s)
            wind_mode: 风模式 - 'random'(随机), 'fixed'(固定最大), 'curriculum'(课程学习)
        """
        super(OtterBerthingTrackingEnv, self).__init__()

        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.verbose = verbose
        self.training_phase = training_phase

        self.initial_velocity = initial_velocity
        self.use_paper_velocity = use_paper_velocity
        
        # 路径参数
        self.kc = kc
        self.DockingLine_x = DockingLine_x
        self.DockingLine_y = DockingLine_y
        self.turnpoint = turnpoint
        self.safe_distance = safe_distance
        self.target_ratio = target_ratio

        # 控制参数
        self.final_phase_smooth_alpha = final_phase_smooth_alpha
        self.velocity_deadzone = velocity_deadzone
        self.rpm_deadzone = rpm_deadzone
        self.stable_rpm = None

        # ===== 海流干扰配置 =====
        self.enable_current = enable_current
        self.V_c_max = V_c_max
        self.current_mode = current_mode
        # 当前episode的海流参数（在reset中设置）
        self.current_V_c = 0.0
        self.current_beta_c = 0.0
        # 课程学习海流配置 (渐进过渡版本)
        self.curriculum_current_config = {
            1: {'V_c_max': 0.0},      # Level 1: 无海流
            2: {'V_c_max': 0.05},     # Level 2: 超弱海流 (渐进过渡)
            3: {'V_c_max': 0.1},      # Level 3: 弱海流
            4: {'V_c_max': 0.2},      # Level 4: 中等海流
        }

        # ===== 风干扰配置 =====
        self.enable_wind = enable_wind
        self.V_w_max = V_w_max
        self.wind_mode = wind_mode
        # 当前episode的风参数（在reset中设置）
        self.current_V_w = 0.0
        self.current_beta_w = 0.0
        # 课程学习风配置 (渐进过渡版本)
        self.curriculum_wind_config = {
            1: {'V_w_max': 0.0},      # Level 1: 无风
            2: {'V_w_max': 0.5},      # Level 2: 超微风 (渐进过渡)
            3: {'V_w_max': 1.5},      # Level 3: 微风 (1级风)
            4: {'V_w_max': 3.0},      # Level 4: 轻风 (2级风)
        }

        # 初始化船舶模型
        self.ship = OtterUSV_DualPropeller(verbose=verbose)
        self.ship_length = self.ship.L
        self.ship_beam = self.ship.B

        # 初始化轨迹生成器 - 使用berthing_sim_smoothest.py参数
        self.trajectory = ExponentialTrajectoryGenerator(
            kc=kc,
            DockingLine_x=DockingLine_x,
            DockingLine_y=DockingLine_y,
            turnpoint=turnpoint,
            u_max=initial_velocity,
            safe_distance=safe_distance,
            target_ratio=target_ratio,
            zeta=zeta,
            ws_0=ws_0,
            t_final=t_final,
            dt=dt,
            start_pos=(0, 0),
            object_length=self.ship_length,
            object_width=self.ship_beam
        )

        self._setup_curriculum(curriculum_level)

        # 动作空间
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32
        )

        # 观测空间 (30维) - 包含扰动信息用于评估旧模型
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(30,), dtype=np.float32
        )

        self.max_n = self.ship.core.n_max
        self.min_n = self.ship.core.n_min
        self.max_n_rate = 20.0

        self.coasting_distance = self.trajectory.arc_length * 0.15
        self.berthing_zone = self.trajectory.arc_length * 0.05
        self.final_approach_zone = self.trajectory.arc_length * 0.02

        self._init_state_variables()

        if self.verbose:
            print(f"\n[环境初始化] berthing_sim_smoothest.py路径:")
            print(f"  kc={kc}, DockingLine=({DockingLine_x}, {DockingLine_y})")
            print(f"  turnpoint={turnpoint}, safe_distance={safe_distance}")
            print(f"  弧长={self.trajectory.arc_length:.2f}m")

    def _setup_curriculum(self, level):
        self.curriculum_level = level
        # Level 2 使用更宽松的 CTE 边界 (4.0*2=8m)，给予更多学习时间
        config = {
            1: {'position_tolerance': 0.5, 'heading_tolerance': np.deg2rad(10),
                'velocity_tolerance': 0.08, 'yaw_rate_tolerance': 0.05, 'max_cross_track': 2.0},
            2: {'position_tolerance': 0.3, 'heading_tolerance': np.deg2rad(7),
                'velocity_tolerance': 0.05, 'yaw_rate_tolerance': 0.03, 'max_cross_track': 4.0},  # 增加容差
            3: {'position_tolerance': 0.2, 'heading_tolerance': np.deg2rad(5),
                'velocity_tolerance': 0.03, 'yaw_rate_tolerance': 0.02, 'max_cross_track': 2.0},  # 调整
            4: {'position_tolerance': 0.1, 'heading_tolerance': np.deg2rad(2),
                'velocity_tolerance': 0.015, 'yaw_rate_tolerance': 0.015, 'max_cross_track': 1.0}  # 最终精度
        }.get(level, {})
        for k, v in config.items():
            setattr(self, k, v)

    def set_curriculum_level(self, level):
        self._setup_curriculum(level)

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        np.random.seed(seed)
        return [seed]

    def _init_state_variables(self):
        self.current_step = 0
        self.closest_path_idx = 0
        self.path_progress = 0.0
        self.last_action = np.array([0.0, 0.0])
        self.last_dist_to_goal = None
        self.near_berth_steps = 0
        self.stable_position_steps = 0
        self.last_progress = 0.0
        self.high_speed_near_berth_steps = 0
        self.last_positions = deque(maxlen=20)
        self.accumulated_cross_track = 0.0
        self.berthing_phase = False
        self.final_approach_active = False
        self.stable_rpm = None
        self.in_deadzone = False
        self.ship_trajectory = []
        self.control_history = []
        self.error_history = []
        self.reward_history = []
        self.episode_success = False

    def reset(self):
        initial_state = {
            'x': self.trajectory.start_pos[0],
            'y': self.trajectory.start_pos[1],
            'psi': self.trajectory.start_heading,
            'u': self.initial_velocity,
            'vm': 0.0,
            'r': 0.0,
        }
        self.ship.reset(initial_state)
        self._init_state_variables()

        # ===== 海流域随机化 =====
        if self.enable_current:
            # 根据课程学习等级获取海流上限
            level_config = self.curriculum_current_config.get(
                self.curriculum_level, {'V_c_max': self.V_c_max}
            )
            V_c_max_level = level_config['V_c_max']

            if self.current_mode == 'random':
                # 随机采样海流速度和方向
                self.current_V_c = np.random.uniform(0, V_c_max_level)
                self.current_beta_c = np.random.uniform(0, 2 * np.pi)
            elif self.current_mode == 'fixed':
                # 固定最大海流，固定方向
                self.current_V_c = V_c_max_level
                self.current_beta_c = 0.0
            elif self.current_mode == 'curriculum':
                # 课程学习模式：在当前等级上限内随机采样
                self.current_V_c = np.random.uniform(0, V_c_max_level)
                self.current_beta_c = np.random.uniform(0, 2 * np.pi)
            else:
                self.current_V_c = 0.0
                self.current_beta_c = 0.0

            if self.verbose:
                print(f"[Current] V_c={self.current_V_c:.3f} m/s, "
                      f"beta_c={np.degrees(self.current_beta_c):.1f}°")
        else:
            self.current_V_c = 0.0
            self.current_beta_c = 0.0

        # ===== 风域随机化 =====
        if self.enable_wind:
            # 根据课程学习等级获取风速上限
            wind_level_config = self.curriculum_wind_config.get(
                self.curriculum_level, {'V_w_max': self.V_w_max}
            )
            V_w_max_level = wind_level_config['V_w_max']

            if self.wind_mode == 'random':
                # 随机采样风速和风向
                self.current_V_w = np.random.uniform(0, V_w_max_level)
                self.current_beta_w = np.random.uniform(0, 2 * np.pi)
            elif self.wind_mode == 'fixed':
                # 固定最大风速，固定方向
                self.current_V_w = V_w_max_level
                self.current_beta_w = 0.0
            elif self.wind_mode == 'curriculum':
                # 课程学习模式：在当前等级上限内随机采样
                self.current_V_w = np.random.uniform(0, V_w_max_level)
                self.current_beta_w = np.random.uniform(0, 2 * np.pi)
            else:
                self.current_V_w = 0.0
                self.current_beta_w = 0.0

            if self.verbose:
                print(f"[Wind] V_w={self.current_V_w:.3f} m/s, "
                      f"beta_w={np.degrees(self.current_beta_w):.1f}°")
        else:
            self.current_V_w = 0.0
            self.current_beta_w = 0.0

        g = 9.81
        Umax = 3.0
        k_pos = 0.02216 / 2
        Xu = -24.4 * g / Umax
        n_squared = abs(Xu) * self.initial_velocity / (2 * k_pos)
        initial_n = np.sqrt(max(0, n_squared))
        initial_n_normalized = np.clip(initial_n / self.max_n, -1.0, 1.0)

        self.last_action = np.array([initial_n_normalized, initial_n_normalized])
        self.stable_rpm = initial_n_normalized

        self._update_closest_path_point_improved()
        self.last_progress = self.path_progress

        if self.verbose:
            print(f"\n[Reset] 位置: ({initial_state['x']:.2f}, {initial_state['y']:.2f}), "
                  f"速度: {initial_state['u']:.2f}m/s, 目标: ({self.DockingLine_x}, {self.DockingLine_y})")

        return self._get_observation()

    def _get_target_velocity(self, distance_to_goal):
        if self.use_paper_velocity:
            if self.closest_path_idx < len(self.trajectory.s_arr):
                current_s = self.trajectory.s_arr[self.closest_path_idx]
                _, _, _, ref_velocity = self.trajectory.get_ref_with_velocity(current_s)
                return ref_velocity
            return 0.0
        return self.trajectory.get_target_velocity(distance_to_goal)

    def _get_ref_full(self, s):
        return self.trajectory.get_ref_full(s)

    def step(self, action):
        self.ship_trajectory.append(self.ship.get_state_vector())
        self.last_positions.append([self.ship.state['x'], self.ship.state['y']])

        u = self.ship.state['u']
        berth_pos = self.trajectory.points[-1]
        dist_to_goal = np.linalg.norm([
            self.ship.state['x'] - berth_pos[0],
            self.ship.state['y'] - berth_pos[1]
        ])

        if self.berthing_phase and u < 0.1:
            alpha = self.final_phase_smooth_alpha
        elif self.berthing_phase:
            alpha = 0.08
        else:
            alpha = 0.15

        target_velocity = self._get_target_velocity(dist_to_goal)

        if (abs(u - target_velocity) < self.velocity_deadzone and
                abs(u) < 0.05 and self.berthing_phase):
            if self.stable_rpm is None:
                self.stable_rpm = (self.last_action[0] + self.last_action[1]) / 2
            action_diff = np.abs(action - self.last_action)
            if np.max(action_diff) < self.rpm_deadzone:
                action = self.last_action.copy()
                self.in_deadzone = True
            else:
                self.in_deadzone = False
        else:
            self.in_deadzone = False
            self.stable_rpm = None

        if self.berthing_phase and abs(u) < 0.01:
            action = action * 0.9

        target_n_port = action[0] * self.max_n if action[0] >= 0 else action[0] * abs(self.min_n)
        target_n_starboard = action[1] * self.max_n if action[1] >= 0 else action[1] * abs(self.min_n)

        current_n_port = self.last_action[0] * self.max_n if self.last_action[0] >= 0 else self.last_action[0] * abs(self.min_n)
        current_n_starboard = self.last_action[1] * self.max_n if self.last_action[1] >= 0 else self.last_action[1] * abs(self.min_n)

        n_port = current_n_port + alpha * (target_n_port - current_n_port)
        n_starboard = current_n_starboard + alpha * (target_n_starboard - current_n_starboard)

        rate_limit = self.max_n_rate * self.dt * (0.3 if self.berthing_phase and u < 0.1 else 1.0)

        n_port = current_n_port + np.clip(n_port - current_n_port, -rate_limit, rate_limit)
        n_starboard = current_n_starboard + np.clip(n_starboard - current_n_starboard, -rate_limit, rate_limit)

        n_port = np.clip(n_port, self.min_n, self.max_n)
        n_starboard = np.clip(n_starboard, self.min_n, self.max_n)

        self.last_action[0] = n_port / self.max_n if n_port >= 0 else n_port / abs(self.min_n)
        self.last_action[1] = n_starboard / self.max_n if n_starboard >= 0 else n_starboard / abs(self.min_n)

        self.control_history.append([n_port, n_starboard])
        self.ship.step(n_port, n_starboard, self.dt,
                       V_c=self.current_V_c, beta_c=self.current_beta_c,
                       V_w=self.current_V_w, beta_w=self.current_beta_w)
        self._update_closest_path_point_improved()

        dist_to_goal = np.linalg.norm([
            self.ship.state['x'] - berth_pos[0],
            self.ship.state['y'] - berth_pos[1]
        ])

        if not self.berthing_phase and dist_to_goal < self.berthing_zone:
            self.berthing_phase = True

        reward = self._compute_simplified_reward(action)
        done = self._check_done()
        self.current_step += 1
        obs = self._get_observation()
        info = self._get_info()

        if info.get('success', False):
            self.episode_success = True

        if self.verbose and self.current_step % self.log_interval == 0:
            self._log_step_info(reward)

        return obs, reward, done, info

    def _compute_simplified_reward(self, action):
        state = self.ship.state
        x, y, psi = state['x'], state['y'], state['psi']
        u, vm, r = state['u'], state['vm'], state['r']

        if self.closest_path_idx < len(self.trajectory.s_arr):
            current_s = self.trajectory.s_arr[self.closest_path_idx]
            ref_x, ref_y, ref_psi, ref_velocity, ref_r = self._get_ref_full(current_s)
        else:
            ref_x, ref_y = self.trajectory.points[-1]
            ref_psi = self.trajectory.end_heading
            ref_velocity = 0.0
            ref_r = 0.0

        berth_pos = self.trajectory.points[-1]
        dist_to_goal = np.linalg.norm([x - berth_pos[0], y - berth_pos[1]])

        position_error = np.linalg.norm([x - ref_x, y - ref_y])
        heading_error = abs(self._normalize_angle(psi - ref_psi))
        r_track = -2.0 * position_error - 3.0 * heading_error

        target_velocity = ref_velocity if self.use_paper_velocity else self._get_target_velocity(dist_to_goal)
        speed_error = abs(u - target_velocity)

        if self.berthing_phase and u < 0.1:
            r_speed = -1.0 * speed_error + (3.0 if speed_error < 0.03 else 0)
        else:
            r_speed = -5.0 * speed_error + (2.0 if speed_error < 0.05 else 0)

        action_change = np.linalg.norm(action - self.last_action)
        if self.berthing_phase and u < 0.1:
            r_smooth = 2.0 if action_change < 0.02 else -3.0 * action_change
        else:
            r_smooth = -0.5 * action_change

        r_deadzone = 1.0 if self.in_deadzone else 0.0

        r_zero_rpm = 0.0
        if self.berthing_phase and abs(u) < 0.001:
            avg_action = (abs(self.last_action[0]) + abs(self.last_action[1])) / 2
            r_zero_rpm = np.clip(3.0 * (1.0 - avg_action / 0.1), -2.0, 3.0)

        cross_track_error = self._calculate_cross_track_error(x, y)
        r_safe = -15.0 * (cross_track_error - 0.8) if cross_track_error > 0.8 else 0.0
        if dist_to_goal < self.final_approach_zone:
            resultant_velocity = np.sqrt(u ** 2 + vm ** 2)
            if resultant_velocity > 0.1:
                r_safe -= 20.0 * (resultant_velocity - 0.1)

        r_sparse = 0.0
        berth_heading = self.trajectory.end_heading
        if self._check_berthing_success(x, y, psi, np.sqrt(u ** 2 + vm ** 2), r, berth_heading):
            r_sparse = 100.0
        elif cross_track_error > self.max_cross_track:
            r_sparse = -100.0

        progress_delta = self.path_progress - self.last_progress
        r_progress = 50.0 * progress_delta if progress_delta > 0 else -20.0 * abs(progress_delta)
        self.last_progress = self.path_progress

        # 使用路径规划的参考偏航角速度
        target_r = ref_r
        r_yaw = -2.0 * abs(r - target_r)

        if dist_to_goal > self.coasting_distance:
            w_track, w_speed, w_smooth, w_safe, w_progress, w_yaw = 1.0, 1.5, 0.3, 1.0, 2.0, 1.0
        elif dist_to_goal > self.berthing_zone:
            w_track, w_speed, w_smooth, w_safe, w_progress, w_yaw = 1.5, 1.5, 0.5, 1.5, 1.0, 1.5
        else:
            w_track, w_speed, w_smooth, w_safe, w_progress, w_yaw = 2.0, 1.0, 2.5, 2.0, 0.3, 1.5

        total_reward = (w_track * r_track + w_speed * r_speed + w_smooth * r_smooth +
                        w_safe * r_safe + w_progress * r_progress + r_sparse +
                        w_yaw * r_yaw + r_deadzone + r_zero_rpm)

        return np.clip(total_reward, -50, 100)

    def _calculate_cross_track_error(self, x, y):
        if self.closest_path_idx >= len(self.trajectory.points) - 1:
            # 到达轨迹末端时，计算垂直于终点航向的横向误差（而非到终点的距离）
            berth_pos = self.trajectory.points[-1]
            berth_heading = self.trajectory.end_heading

            # 从终点到船舶的向量
            dx = x - berth_pos[0]
            dy = y - berth_pos[1]

            # 横向误差 = 垂直于航向方向的分量
            # 航向方向 = (cos(psi), sin(psi))
            # 横向方向 = (-sin(psi), cos(psi))
            lateral_error = abs(-np.sin(berth_heading) * dx + np.cos(berth_heading) * dy)
            return lateral_error

        p1 = self.trajectory.points[self.closest_path_idx]
        p2 = self.trajectory.points[self.closest_path_idx + 1]
        ship_pos = np.array([x, y])

        vec_segment = p2 - p1
        vec_to_ship = ship_pos - p1
        proj = np.clip(np.dot(vec_to_ship, vec_segment) / np.dot(vec_segment, vec_segment), 0, 1)
        closest_on_segment = p1 + proj * vec_segment
        return np.linalg.norm(ship_pos - closest_on_segment)

    def _update_closest_path_point_improved(self):
        ship_pos = np.array([self.ship.state['x'], self.ship.state['y']])

        search_start = max(0, self.closest_path_idx - 5)
        search_end = min(len(self.trajectory.points), self.closest_path_idx + 20)

        search_points = self.trajectory.points[search_start:search_end]
        if len(search_points) > 0:
            distances = np.linalg.norm(search_points - ship_pos, axis=1)
            local_closest_idx = np.argmin(distances)
            new_closest_idx = search_start + local_closest_idx

            if new_closest_idx > self.closest_path_idx or \
                    (new_closest_idx >= self.closest_path_idx - 2 and distances[local_closest_idx] < 1.0):
                self.closest_path_idx = new_closest_idx

        if self.closest_path_idx < len(self.trajectory.s_arr):
            if self.closest_path_idx < len(self.trajectory.points) - 1:
                p1 = self.trajectory.points[self.closest_path_idx]
                p2 = self.trajectory.points[self.closest_path_idx + 1]
                segment_vec = p2 - p1
                segment_length = np.linalg.norm(segment_vec)

                if segment_length > 0:
                    segment_unit = segment_vec / segment_length
                    ship_vec = ship_pos - p1
                    projection = np.clip(np.dot(ship_vec, segment_unit), 0, segment_length)

                    s1 = self.trajectory.s_arr[self.closest_path_idx]
                    if self.closest_path_idx + 1 < len(self.trajectory.s_arr):
                        s2 = self.trajectory.s_arr[self.closest_path_idx + 1]
                        current_s = s1 + (s2 - s1) * (projection / segment_length)
                    else:
                        current_s = s1
                    self.path_progress = current_s / self.trajectory.arc_length
                else:
                    self.path_progress = self.trajectory.s_arr[self.closest_path_idx] / self.trajectory.arc_length
            else:
                self.path_progress = 1.0
        else:
            self.path_progress = 1.0

    def _check_done(self):
        state = self.ship.state
        x, y, psi = state['x'], state['y'], state['psi']
        u, vm, r = state['u'], state['vm'], state['r']

        resultant_velocity = np.sqrt(u ** 2 + vm ** 2)
        berth_pos = self.trajectory.points[-1]
        berth_heading = self.trajectory.end_heading
        dist_to_goal = np.linalg.norm([x - berth_pos[0], y - berth_pos[1]])

        if self._check_berthing_success(x, y, psi, resultant_velocity, r, berth_heading):
            if self.verbose:
                print(f"成功靠泊! 位置误差: {dist_to_goal:.3f}m")
            return True

        cross_track_error = self._calculate_cross_track_error(x, y)
        if cross_track_error > self.max_cross_track * 2:
            if self.verbose:
                print(f"超出边界! 横向误差: {cross_track_error:.3f}m")
            return True

        if self.current_step >= self.max_episode_steps:
            if self.verbose:
                print(f"超时! 步数: {self.current_step}")
            return True

        return False

    def _check_berthing_success(self, x, y, psi, velocity, yaw_rate, berth_heading):
        berth_pos = self.trajectory.points[-1]
        pos_error = np.linalg.norm([x - berth_pos[0], y - berth_pos[1]])
        if pos_error > self.position_tolerance:
            return False
        heading_error = abs(self._normalize_angle(psi - berth_heading))
        if heading_error > self.heading_tolerance:
            return False
        if velocity > self.velocity_tolerance:
            return False
        if abs(yaw_rate) > self.yaw_rate_tolerance:
            return False
        return True

    def _normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _get_observation(self):
        state = self.ship.state
        x, y, psi = state['x'], state['y'], state['psi']
        u, vm, r = state['u'], state['vm'], state['r']

        if self.closest_path_idx < len(self.trajectory.s_arr):
            current_s = self.trajectory.s_arr[self.closest_path_idx]
            ref_x, ref_y, ref_psi, ref_velocity, ref_r = self._get_ref_full(current_s)
        else:
            ref_x, ref_y = self.trajectory.points[-1]
            ref_psi = self.trajectory.end_heading
            ref_velocity = 0.0
            ref_r = 0.0

        pos_error_x = x - ref_x
        pos_error_y = y - ref_y
        heading_error = self._normalize_angle(psi - ref_psi)
        cross_track_error = self._calculate_cross_track_error(x, y)

        berth_pos = self.trajectory.points[-1]
        dist_to_goal = np.linalg.norm([x - berth_pos[0], y - berth_pos[1]])
        angle_to_goal = np.arctan2(berth_pos[1] - y, berth_pos[0] - x)
        relative_angle_to_goal = self._normalize_angle(angle_to_goal - psi)

        lookahead_heading_change = 0.0
        if self.closest_path_idx + 10 < len(self.trajectory.s_arr):
            future_psi = self.trajectory.get_ref(self.trajectory.s_arr[self.closest_path_idx + 10])[2]
            lookahead_heading_change = self._normalize_angle(future_psi - ref_psi)

        velocity_error = u - ref_velocity

        # 扩展观测（30维）- 匹配vectorized_env的格式
        # 干扰归一化
        V_c_norm = self.current_V_c / 0.3  # 归一化海流速度
        V_w_norm = self.current_V_w / 5.0  # 归一化风速

        obs = [
            pos_error_x, pos_error_y, cross_track_error,
            np.sin(psi), np.cos(psi), heading_error,
            u, vm, r, velocity_error, ref_velocity,
            np.sin(ref_psi), np.cos(ref_psi), lookahead_heading_change,
            dist_to_goal * 0.01,  # 距离缩放 (100m -> 1.0)
            np.sin(relative_angle_to_goal), np.cos(relative_angle_to_goal),
            self.path_progress, self.last_action[0], self.last_action[1],
            float(dist_to_goal < self.coasting_distance),
            float(dist_to_goal < self.berthing_zone),
            float(self.berthing_phase),
            ref_r,  # 参考偏航角速度
            # 扰动信息 (6维) - 与vectorized_env完全一致
            V_c_norm,                      # 24: 归一化海流速度
            np.sin(self.current_beta_c),   # 25: 海流方向正弦
            np.cos(self.current_beta_c),   # 26: 海流方向余弦
            V_w_norm,                      # 27: 归一化风速
            np.sin(self.current_beta_w),   # 28: 风向正弦
            np.cos(self.current_beta_w),   # 29: 风向余弦
        ]

        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        state = self.ship.state
        x, y, psi = state['x'], state['y'], state['psi']
        u, vm, r = state['u'], state['vm'], state['r']

        berth_pos = self.trajectory.points[-1]
        berth_heading = self.trajectory.end_heading
        dist_to_goal = np.linalg.norm([x - berth_pos[0], y - berth_pos[1]])
        resultant_velocity = np.sqrt(u ** 2 + vm ** 2)

        ref_r = 0.0
        if self.closest_path_idx < len(self.trajectory.s_arr):
            current_s = self.trajectory.s_arr[self.closest_path_idx]
            ref_r = self.trajectory.get_target_yaw_rate(current_s)

        current_avg_rpm = (abs(self.last_action[0]) + abs(self.last_action[1])) / 2 * self.max_n * 60 / (2 * np.pi)

        return {
            'ship_position': (x, y),
            'ship_heading': np.rad2deg(psi),
            'ship_velocity': (u, vm),
            'ship_yaw_rate': r,
            'distance_to_goal': dist_to_goal,
            'path_progress': self.path_progress,
            'cross_track_error': self._calculate_cross_track_error(x, y),
            'target_velocity': self._get_target_velocity(dist_to_goal),
            'target_yaw_rate': ref_r,
            'resultant_velocity': resultant_velocity,
            'success': self._check_berthing_success(x, y, psi, resultant_velocity, r, berth_heading),
            'curriculum_level': self.curriculum_level,
            'in_deadzone': self.in_deadzone,
            'current_avg_rpm': current_avg_rpm,
            'trajectory_info': {
                'kc': self.kc,
                'DockingLine_x': self.DockingLine_x,
                'DockingLine_y': self.DockingLine_y,
                'turnpoint': self.turnpoint,
                'arc_length': self.trajectory.arc_length,
                'safe_distance': self.safe_distance,
                'target_ratio': self.target_ratio
            }
        }

    def _log_step_info(self, reward):
        state = self.ship.state
        x, y, u, r = state['x'], state['y'], state['u'], state['r']

        berth_pos = self.trajectory.points[-1]
        dist_to_goal = np.linalg.norm([x - berth_pos[0], y - berth_pos[1]])
        cte = self._calculate_cross_track_error(x, y)
        target_velocity = self._get_target_velocity(dist_to_goal)

        ref_r = 0.0
        if self.closest_path_idx < len(self.trajectory.s_arr):
            current_s = self.trajectory.s_arr[self.closest_path_idx]
            ref_r = self.trajectory.get_target_yaw_rate(current_s)

        current_avg_rpm = (abs(self.last_action[0]) + abs(self.last_action[1])) / 2 * self.max_n * 60 / (2 * np.pi)

        print(f"Step {self.current_step}: progress={self.path_progress*100:.1f}%, "
              f"cte={cte:.3f}, u={u:.3f}(target={target_velocity:.3f}), "
              f"r={np.rad2deg(r):.2f}°/s(target={np.rad2deg(ref_r):.2f}°/s), "
              f"dist={dist_to_goal:.2f}, rpm={current_avg_rpm:.0f}")

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    print("=" * 60)
    print("船舶靠泊环境测试 - berthing_sim_smoothest.py版本")
    print("=" * 60)

    env = OtterBerthingTrackingEnv(curriculum_level=1, verbose=True)
    obs = env.reset()
    print(f"观测空间维度: {obs.shape}")
    env.trajectory.print_summary()
