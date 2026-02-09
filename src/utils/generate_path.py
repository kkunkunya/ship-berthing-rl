"""
船舶靠泊路径生成器 - 完全基于berthing_sim_smoothest.py

直接使用berthing_sim_smoothest.py中的:
1. 路径定义: 直线段 + 指数衰减段
2. 速度规划: ud = u_max * tanh(k * |y - DockingLine_y|)
3. ODE求解: dθ/dt = ud / |dr/dθ| - ws, dws/dt = -ζ * ws
4. 航向计算: ψ_d = π/2 - arctan2(dx/dθ, dy/dθ)
5. 偏航角速度: r = dψ/dt
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class PathPoint:
    """路径点数据结构"""
    theta: float      # 路径参数
    x: float          # X位置
    y: float          # Y位置
    psi: float        # 航向角 (弧度)
    u_d: float        # 期望速度
    r_d: float        # 期望偏航角速度 (弧度/秒)
    s: float          # 累计弧长
    kappa: float      # 曲率
    t: float          # 时间


def calculate_speed_tanh(y_pos, DockingLine_y, safe_distance, u_max=1.5, target_ratio=0.95):
    """
    统一Tanh速度计算函数（无分段，全程连续）
    
    公式: ud = u_max * tanh(k * |y - DockingLine_y|)
    """
    dy = abs(y_pos - DockingLine_y)
    k = np.arctanh(target_ratio) / safe_distance
    return u_max * np.tanh(k * dy)


def full_coupled_odes(t, Init_value, params):
    """
    耦合常微分方程组 - 完全来自berthing_sim_smoothest.py
    
    状态变量:
        Init_value[0] = theta: 路径参数
        Init_value[1] = ws: 速度设定误差函数
    
    微分方程:
        dθ/dt = ud / |dr/dθ| - ws
        dws/dt = -ζ * ws
    """
    zeta = params['zeta']
    kc = params['kc']
    turnpoint = params['turnpoint']
    kd = params['kd']
    Td = params['Td']
    DockingLine_x = params['DockingLine_x']
    DockingLine_y = params['DockingLine_y']
    safe_distance = params['safe_distance']
    u_max = params['u_max']
    target_ratio = params['target_ratio']

    theta = Init_value[0]
    ws = Init_value[1]

    # 根据theta计算当前位置和路径导数
    if theta <= turnpoint:
        x_pos = theta
        y_pos = kc * theta
        dx_dtheta = 1.0
        dy_dtheta = kc
    else:
        x_pos = DockingLine_x - kd * np.exp(-(theta - turnpoint) / Td)
        y_pos = theta - turnpoint + kc * turnpoint
        dx_dtheta = (kd / Td) * np.exp(-(theta - turnpoint) / Td)
        dy_dtheta = 1.0

    # 使用Tanh函数计算期望速度
    ud = calculate_speed_tanh(y_pos, DockingLine_y, safe_distance, u_max, target_ratio)

    # 计算路径弧长微元
    denominator = np.sqrt(dx_dtheta**2 + dy_dtheta**2)

    # 微分方程组
    dtheta_dt = ud / denominator - ws
    dws_dt = -zeta * ws

    return [dtheta_dt, dws_dt]


class ExponentialTrajectoryGenerator:
    """
    船舶靠泊路径生成器 - 完全基于berthing_sim_smoothest.py
    
    使用ODE求解生成轨迹，保持与原始代码完全一致
    """

    def __init__(self,
                 # 路径参数 (berthing_sim_smoothest.py默认值)
                 kc: float = 0.45,
                 DockingLine_x: float = 70.0,
                 DockingLine_y: float = 95.0,
                 turnpoint: float = 35.0,
                 
                 # 速度参数
                 u_max: float = 1.5,
                 safe_distance: float = 60.0,
                 target_ratio: float = 0.95,
                 
                 # ODE参数
                 zeta: float = 0.2,
                 ws_0: float = 1.5,
                 t_final: float = 200.0,
                 dt: float = 0.1,
                 
                 # 起始位置
                 start_pos: Tuple[float, float] = (0, 0),
                 
                 # 船舶尺寸
                 object_length: float = 2.0,
                 object_width: float = 1.08,
                 
                 # 目标区域
                 target_length: float = None,
                 target_width: float = None,
                 target_pos: Tuple[float, float] = None,
                 
                 # 兼容性参数（忽略）
                 total_length: float = None,
                 straight_ratio: float = None,
                 straight_slope: float = None,
                 initial_velocity: float = None,
                 final_velocity: float = None,
                 end_heading: float = np.pi / 2,
                 extend_to_target_heading: bool = True,
                 num_points: int = 2000):
        
        # 路径参数
        self.kc = kc
        self.DockingLine_x = DockingLine_x
        self.DockingLine_y = DockingLine_y
        self.turnpoint = turnpoint
        self.kd = DockingLine_x - turnpoint
        self.Td = self.kd * kc
        
        # 速度参数
        self.u_max = u_max
        self.initial_velocity = u_max
        self.safe_distance = safe_distance
        self.target_ratio = target_ratio
        
        # ODE参数
        self.zeta = zeta
        self.ws_0 = ws_0
        self.t_final = t_final
        self.dt = dt
        
        # 起始位置
        self.start_pos = np.array(start_pos)
        
        # 船舶尺寸
        self.object_length = object_length
        self.object_width = object_width
        
        # 兼容性属性
        self.total_length = turnpoint + (DockingLine_y - kc * turnpoint)
        self.straight_ratio = turnpoint / self.total_length if self.total_length > 0 else 0.6
        self.straight_slope = kc
        self.end_heading = end_heading
        self.target_end_heading = end_heading
        
        # 生成路径
        self.path_points: List[PathPoint] = []
        self.points: np.ndarray = None
        self.s_arr: np.ndarray = None
        self.arc_length: float = 0.0
        
        # 运行ODE求解生成轨迹
        self._generate_path_via_ode()
        
        # 目标区域
        if target_length is None:
            self.target_length = object_length * 1.2
        else:
            self.target_length = target_length

        if target_width is None:
            self.target_width = object_width * 2
        else:
            self.target_width = target_width

        if target_pos is None:
            self.target_pos = (self.DockingLine_x, self.DockingLine_y)
        else:
            self.target_pos = target_pos

    def _generate_path_via_ode(self):
        """使用ODE求解生成路径 - 完全来自berthing_sim_smoothest.py"""
        
        # 时间数组
        tspan = np.arange(0, self.t_final + self.dt, self.dt)
        
        # 初始条件
        theta_0 = 0
        Init_value = [theta_0, self.ws_0]
        
        # 参数字典
        params = {
            'zeta': self.zeta,
            'kc': self.kc,
            'turnpoint': self.turnpoint,
            'kd': self.kd,
            'Td': self.Td,
            'DockingLine_x': self.DockingLine_x,
            'DockingLine_y': self.DockingLine_y,
            'safe_distance': self.safe_distance,
            'u_max': self.u_max,
            'target_ratio': self.target_ratio
        }
        
        # 求解ODE
        solution = solve_ivp(
            fun=lambda t, y: full_coupled_odes(t, y, params),
            t_span=(0, self.t_final),
            y0=Init_value,
            method='RK45',
            t_eval=tspan,
            rtol=1e-6,
            atol=1e-8
        )
        
        # 提取结果
        t = solution.t
        theta_sol = solution.y[0, :]
        ws_sol = solution.y[1, :]
        
        n = len(t)
        x_sol = np.zeros(n)
        y_sol = np.zeros(n)
        ud_sol = np.zeros(n)
        psi_d_sol = np.zeros(n)
        kappa_sol = np.zeros(n)
        
        # 计算所有派生变量
        for i in range(n):
            theta_current = theta_sol[i]
            
            if theta_current <= self.turnpoint:
                x_sol[i] = theta_current
                y_sol[i] = self.kc * theta_current
                dx_dtheta = 1.0
                dy_dtheta = self.kc
                d2x_dtheta2 = 0.0
            else:
                exp_term = np.exp(-(theta_current - self.turnpoint) / self.Td)
                x_sol[i] = self.DockingLine_x - self.kd * exp_term
                y_sol[i] = theta_current - self.turnpoint + self.kc * self.turnpoint
                dx_dtheta = (self.kd / self.Td) * exp_term
                dy_dtheta = 1.0
                d2x_dtheta2 = -(self.kd / (self.Td ** 2)) * exp_term
            
            # 航向角: ψ_d = π/2 - arctan2(dx/dθ, dy/dθ)
            psi_d_sol[i] = np.pi/2 - np.arctan2(dx_dtheta, dy_dtheta)
            
            # 期望速度
            ud_sol[i] = calculate_speed_tanh(
                y_sol[i], self.DockingLine_y, 
                self.safe_distance, self.u_max, self.target_ratio
            )
            
            # 曲率
            numerator = abs(dx_dtheta * 0 - dy_dtheta * d2x_dtheta2)
            denominator = (dx_dtheta ** 2 + dy_dtheta ** 2) ** 1.5
            kappa_sol[i] = numerator / denominator if denominator > 1e-10 else 0.0
        
        # 添加起始位置偏移
        x_sol += self.start_pos[0]
        y_sol += self.start_pos[1]
        
        # 计算累计弧长
        dx = np.diff(x_sol)
        dy = np.diff(y_sol)
        ds = np.sqrt(dx ** 2 + dy ** 2)
        s_arr = np.concatenate([[0.0], np.cumsum(ds)])
        
        # 计算偏航角速度 r = dψ/dt
        r_sol = np.zeros(n)
        for i in range(n):
            if i == 0:
                r_sol[i] = (psi_d_sol[1] - psi_d_sol[0]) / (t[1] - t[0])
            elif i == n - 1:
                r_sol[i] = (psi_d_sol[n-1] - psi_d_sol[n-2]) / (t[n-1] - t[n-2])
            else:
                r_sol[i] = (psi_d_sol[i+1] - psi_d_sol[i-1]) / (t[i+1] - t[i-1])
        
        # 存储
        self.points = np.column_stack((x_sol, y_sol))
        self.s_arr = s_arr
        self.arc_length = s_arr[-1]
        self.num_points = n
        
        # 创建PathPoint列表
        for i in range(n):
            self.path_points.append(PathPoint(
                theta=theta_sol[i],
                x=x_sol[i],
                y=y_sol[i],
                psi=psi_d_sol[i],
                u_d=ud_sol[i],
                r_d=r_sol[i],
                s=s_arr[i],
                kappa=kappa_sol[i],
                t=t[i]
            ))
        
        # 记录关键信息
        self.turn_point_coords = self._find_turn_point()
        self.start_heading = psi_d_sol[0]
        self.actual_end_heading = psi_d_sol[-1]
        
        # 保存数组
        self.t_arr = t
        self.theta_arr = theta_sol
        self.ws_arr = ws_sol
        self.velocity_profile = ud_sol
        self.heading_profile = psi_d_sol
        self.yaw_rate_profile = r_sol
        
        # 保存原始xd格式数据（兼容berthing_sim_smoothest.py）
        self.xd = {
            'x': x_sol,
            'y': y_sol,
            'psi': psi_d_sol,
            'psi_deg': np.rad2deg(psi_d_sol),
            'u': ud_sol,
            'v': np.zeros(n),
            'r': r_sol,
            't': t,
            'theta': theta_sol,
            'ws': ws_sol,
            'ud': ud_sol,
            'pe': np.sqrt((x_sol - self.DockingLine_x)**2 + (y_sol - self.DockingLine_y)**2)
        }

    def _find_turn_point(self) -> np.ndarray:
        """找到转向点坐标"""
        for pt in self.path_points:
            if pt.theta >= self.turnpoint:
                return np.array([pt.x, pt.y])
        return self.points[len(self.points) // 2]

    def get_ref(self, s: float) -> Tuple[float, float, float]:
        """根据弧长获取参考点"""
        if s <= 0:
            return self.path_points[0].x, self.path_points[0].y, self.path_points[0].psi
        if s >= self.arc_length:
            return self.path_points[-1].x, self.path_points[-1].y, self.path_points[-1].psi

        idx = np.searchsorted(self.s_arr, s)
        idx = min(idx, len(self.path_points) - 1)
        return self.path_points[idx].x, self.path_points[idx].y, self.path_points[idx].psi

    def get_ref_with_velocity(self, s: float) -> Tuple[float, float, float, float]:
        """根据弧长获取参考点（包含速度）"""
        if s <= 0:
            pt = self.path_points[0]
            return pt.x, pt.y, pt.psi, pt.u_d
        if s >= self.arc_length:
            pt = self.path_points[-1]
            return pt.x, pt.y, pt.psi, pt.u_d

        idx = np.searchsorted(self.s_arr, s)
        idx = min(idx, len(self.path_points) - 1)
        pt = self.path_points[idx]
        return pt.x, pt.y, pt.psi, pt.u_d

    def get_ref_full(self, s: float) -> Tuple[float, float, float, float, float]:
        """根据弧长获取完整参考点（包含速度和偏航角速度）"""
        if s <= 0:
            pt = self.path_points[0]
            return pt.x, pt.y, pt.psi, pt.u_d, pt.r_d
        if s >= self.arc_length:
            pt = self.path_points[-1]
            return pt.x, pt.y, pt.psi, pt.u_d, pt.r_d

        idx = np.searchsorted(self.s_arr, s)
        idx = min(idx, len(self.path_points) - 1)
        pt = self.path_points[idx]
        return pt.x, pt.y, pt.psi, pt.u_d, pt.r_d

    def get_target_velocity(self, distance_to_goal: float) -> float:
        """根据到目标的距离获取目标速度"""
        k = np.arctanh(self.target_ratio) / self.safe_distance
        return self.u_max * np.tanh(k * distance_to_goal)

    def get_target_yaw_rate(self, s: float) -> float:
        """根据弧长获取目标偏航角速度"""
        if s <= 0:
            return self.path_points[0].r_d
        if s >= self.arc_length:
            return self.path_points[-1].r_d

        idx = np.searchsorted(self.s_arr, s)
        idx = min(idx, len(self.path_points) - 1)
        return self.path_points[idx].r_d

    def get_full_profile(self) -> dict:
        """获取完整的路径剖面数据"""
        return {
            'theta': np.array([pt.theta for pt in self.path_points]),
            'x': np.array([pt.x for pt in self.path_points]),
            'y': np.array([pt.y for pt in self.path_points]),
            'psi': np.array([pt.psi for pt in self.path_points]),
            'psi_deg': np.rad2deg([pt.psi for pt in self.path_points]),
            'u_d': np.array([pt.u_d for pt in self.path_points]),
            'r_d': np.array([pt.r_d for pt in self.path_points]),
            'r_d_deg': np.rad2deg([pt.r_d for pt in self.path_points]),
            's': np.array([pt.s for pt in self.path_points]),
            'kappa': np.array([pt.kappa for pt in self.path_points]),
            't': np.array([pt.t for pt in self.path_points])
        }

    def get_curvature_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取曲率剖面"""
        s_points = np.array([pt.s for pt in self.path_points])
        curvatures = np.array([pt.kappa for pt in self.path_points])
        return s_points, curvatures

    def get_target_corners(self) -> np.ndarray:
        """获取目标区域四个角"""
        half_length = self.target_length / 2
        half_width = self.target_width / 2

        corners_local = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ])

        cos_h = np.cos(self.end_heading)
        sin_h = np.sin(self.end_heading)
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        corners_global = corners_local @ rotation.T + np.array(self.target_pos)
        return corners_global

    def print_summary(self):
        """打印路径摘要信息"""
        print("\n" + "=" * 60)
        print("船舶靠泊路径参数摘要 (berthing_sim_smoothest.py)")
        print("=" * 60)

        print(f"\n【路径参数】")
        print(f"  直线段斜率 (kc): {self.kc:.3f}")
        print(f"  转向点位置: θ = {self.turnpoint:.2f}")
        print(f"  泊位坐标: ({self.DockingLine_x:.2f}, {self.DockingLine_y:.2f}) m")
        print(f"  kd = {self.kd:.3f}")
        print(f"  Td = {self.Td:.3f}")
        print(f"  实际弧长: {self.arc_length:.2f} m")

        print(f"\n【速度规划 (Tanh连续函数)】")
        print(f"  公式: u_d = u_max * tanh(k * |y - DockingLine_y|)")
        print(f"  最大速度 (u_max): {self.u_max:.3f} m/s")
        print(f"  安全距离 (safe_distance): {self.safe_distance:.1f} m")
        print(f"  目标比例 (target_ratio): {self.target_ratio:.2f}")

        print(f"\n【ODE参数】")
        print(f"  阻尼系数 (zeta): {self.zeta}")
        print(f"  初始ws: {self.ws_0}")
        print(f"  仿真时间: {self.t_final} s")
        print(f"  时间步长: {self.dt} s")

        print(f"\n【航向信息】")
        print(f"  起始航向: {np.rad2deg(self.start_heading):.2f}°")
        print(f"  终点航向: {np.rad2deg(self.actual_end_heading):.2f}°")

        print(f"\n【偏航角速度】")
        max_r = max([abs(pt.r_d) for pt in self.path_points])
        print(f"  最大偏航角速度: {np.rad2deg(max_r):.4f} °/s")

        print(f"\n【位置信息】")
        print(f"  起点: ({self.points[0, 0]:.2f}, {self.points[0, 1]:.2f}) m")
        print(f"  终点: ({self.points[-1, 0]:.2f}, {self.points[-1, 1]:.2f}) m")
        print(f"  转向点: ({self.turn_point_coords[0]:.2f}, {self.turn_point_coords[1]:.2f}) m")
        print(f"  终点位置误差: {self.xd['pe'][-1]:.4f} m")

        # 计算加速度
        accel = np.gradient(self.velocity_profile, self.t_arr)
        print(f"\n【平滑性验证】")
        print(f"  初始速度: {self.velocity_profile[0]:.4f} m/s")
        print(f"  终止速度: {self.velocity_profile[-1]:.6f} m/s")
        print(f"  最大加速度: {np.max(np.abs(accel)):.4f} m/s²")

        print("=" * 60 + "\n")


# 别名
DockingTrajectoryGenerator = ExponentialTrajectoryGenerator


def create_berthing_trajectory(
        kc: float = 0.45,
        DockingLine_x: float = 70.0,
        DockingLine_y: float = 95.0,
        turnpoint: float = 35.0,
        u_max: float = 1.5,
        safe_distance: float = 60.0,
        target_ratio: float = 0.95,
        zeta: float = 0.2,
        ws_0: float = 1.5,
        t_final: float = 200.0,
        dt: float = 0.1,
        ship_length: float = 2.0,
        ship_beam: float = 1.08,
        # 兼容性参数
        u0: float = None,
        **kwargs
) -> ExponentialTrajectoryGenerator:
    """创建靠泊轨迹的便捷函数"""
    if u0 is not None:
        u_max = u0
        
    return ExponentialTrajectoryGenerator(
        kc=kc,
        DockingLine_x=DockingLine_x,
        DockingLine_y=DockingLine_y,
        turnpoint=turnpoint,
        u_max=u_max,
        safe_distance=safe_distance,
        target_ratio=target_ratio,
        zeta=zeta,
        ws_0=ws_0,
        t_final=t_final,
        dt=dt,
        start_pos=(0, 0),
        object_length=ship_length,
        object_width=ship_beam
    )


# 测试代码
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("船舶靠泊轨迹生成器测试")
    print("完全基于 berthing_sim_smoothest.py")
    print("=" * 60)

    # 创建轨迹生成器 - 使用berthing_sim_smoothest.py的默认参数
    trajectory = create_berthing_trajectory(
        kc=0.45,
        DockingLine_x=70.0,
        DockingLine_y=95.0,
        turnpoint=35.0,
        u_max=1.5,
        safe_distance=60.0,
        target_ratio=0.95,
        zeta=0.2,
        ws_0=1.5,
        t_final=200.0,
        dt=0.1
    )

    # 打印摘要
    trajectory.print_summary()

    # 测试接口
    print("\n【接口测试】")
    print("-" * 60)

    print("\n1. get_ref(s) - 获取参考点:")
    for s in [0, 20, 40, 60, 80, trajectory.arc_length]:
        x, y, psi = trajectory.get_ref(s)
        print(f"   s={s:5.1f}m -> x={x:6.2f}m, y={y:6.2f}m, ψ={np.rad2deg(psi):6.1f}°")

    print("\n2. get_ref_with_velocity(s) - 获取参考点(含速度):")
    for s in [0, 20, 40, 60, 80, trajectory.arc_length]:
        x, y, psi, u_d = trajectory.get_ref_with_velocity(s)
        print(f"   s={s:5.1f}m -> x={x:6.2f}m, y={y:6.2f}m, ψ={np.rad2deg(psi):6.1f}°, v={u_d:.3f}m/s")

    print("\n3. get_ref_full(s) - 获取完整参考点(含偏航角速度):")
    for s in [0, 20, 40, 60, 80, trajectory.arc_length]:
        x, y, psi, u_d, r_d = trajectory.get_ref_full(s)
        print(f"   s={s:5.1f}m -> ψ={np.rad2deg(psi):6.1f}°, v={u_d:.3f}m/s, r={np.rad2deg(r_d):.4f}°/s")

    print("\n4. get_target_velocity(dist) - 根据距离获取目标速度:")
    for dist in [100, 60, 40, 20, 10, 5, 2, 0]:
        v = trajectory.get_target_velocity(dist)
        print(f"   dist={dist:5.1f}m -> target_v={v:.3f}m/s")

    print("\n" + "=" * 60)

    # 绘制图形
    profile = trajectory.get_full_profile()
    xd = trajectory.xd

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 速度曲线
    axes[0, 0].plot(profile['t'], profile['u_d'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('$u_d$ (m/s)')
    axes[0, 0].set_title('Desired Speed')
    axes[0, 0].grid(True, alpha=0.3)

    # 航向角
    axes[0, 1].plot(profile['t'], profile['psi_deg'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('$\\psi$ (deg)')
    axes[0, 1].set_title('Heading Angle')
    axes[0, 1].grid(True, alpha=0.3)

    # 偏航角速度
    axes[0, 2].plot(profile['t'], profile['r_d_deg'], 'r-', linewidth=2)
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('r (deg/s)')
    axes[0, 2].set_title('Yaw Rate')
    axes[0, 2].grid(True, alpha=0.3)

    # 路径
    axes[1, 0].plot(profile['x'], profile['y'], 'b-', linewidth=2)
    axes[1, 0].plot(profile['x'][0], profile['y'][0], 'go', markersize=10, label='Start')
    axes[1, 0].plot(profile['x'][-1], profile['y'][-1], 'ro', markersize=10, label='End')
    axes[1, 0].plot(70, 95, 'k*', markersize=15, label='Target (70, 95)')
    axes[1, 0].axhline(y=95, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].axvline(x=70, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('X (m)')
    axes[1, 0].set_ylabel('Y (m)')
    axes[1, 0].set_title('Berthing Path')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 位置误差
    axes[1, 1].plot(profile['t'], xd['pe'], 'k-', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('$p_e$ (m)')
    axes[1, 1].set_title('Position Error')
    axes[1, 1].grid(True, alpha=0.3)

    # 速度-距离曲线
    axes[1, 2].plot(xd['pe'], profile['u_d'], 'b-', linewidth=2)
    axes[1, 2].axvline(x=60, color='k', linestyle='--', alpha=0.5, label='Safe Distance')
    axes[1, 2].set_xlabel('Distance to Berth (m)')
    axes[1, 2].set_ylabel('Speed (m/s)')
    axes[1, 2].set_title('Speed vs Distance')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].invert_xaxis()

    plt.tight_layout()
    plt.savefig('trajectory_profile.png', dpi=150)
    print("图形已保存: trajectory_profile.png")

    print("\n测试完成!")
    print("=" * 60)
