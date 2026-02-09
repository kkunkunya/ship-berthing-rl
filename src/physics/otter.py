import numpy as np
from scipy.integrate import solve_ivp
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ==================== 辅助函数 ====================

def Smtrx(a):
    """反对称矩阵 (skew-symmetric matrix)"""
    return np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])


def Hmtrx(r):
    """6x6变换矩阵"""
    H = np.eye(6)
    H[0:3, 3:6] = Smtrx(r)
    return H


def Rzyx(phi, theta, psi):
    """ZYX欧拉角旋转矩阵"""
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    return np.array([
        [cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth],
        [spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
        [-sth, cth * sphi, cth * cphi]
    ])


def eulerang(phi, theta, psi):
    """欧拉角运动学矩阵"""
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    tth = np.tan(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    J = np.array([
        [cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth, 0, 0, 0],
        [spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi, 0, 0, 0],
        [-sth, cth * sphi, cth * cphi, 0, 0, 0],
        [0, 0, 0, 1, sphi * tth, cphi * tth],
        [0, 0, 0, 0, cphi, -sphi],
        [0, 0, 0, 0, sphi / cth, cphi / cth]
    ])
    return J


def addedMassSurge(m, L, rho):
    """纵荡附加质量"""
    return 0.1 * m


def m2c(MA, nu_r):
    """附加质量矩阵转为科氏矩阵"""
    CA = np.zeros((6, 6))
    CA[0:3, 3:6] = -Smtrx(MA[0:3, 0:3] @ nu_r[0:3] + MA[0:3, 3:6] @ nu_r[3:6])
    CA[3:6, 0:3] = -Smtrx(MA[0:3, 0:3] @ nu_r[0:3] + MA[0:3, 3:6] @ nu_r[3:6])
    CA[3:6, 3:6] = -Smtrx(MA[3:6, 0:3] @ nu_r[0:3] + MA[3:6, 3:6] @ nu_r[3:6])
    return CA


def crossFlowDrag(L, B, T, nu_r):
    """横流阻力（简化版本）"""
    tau = np.zeros(6)
    Cd = 1.0
    rho = 1025
    v = np.clip(nu_r[1], -10, 10)
    r = np.clip(nu_r[5], -5, 5)
    tau[1] = -0.5 * rho * Cd * B * T * L * abs(v) * v
    tau[5] = -0.5 * rho * Cd * B * T * L ** 2 * abs(r) * r / 12
    return tau


def windForce(L, B, H_freeboard, nu, psi, V_w, beta_w):
    """
    计算风力干扰

    参数:
        L: 船长 (m)
        B: 船宽 (m)
        H_freeboard: 干舷高度（水面以上部分）(m)
        nu: 船舶速度向量 [u, v, w, p, q, r]
        psi: 航向角 (rad)
        V_w: 风速 (m/s)
        beta_w: 风向 (rad)，相对于NED坐标系，0表示北风

    返回:
        tau_wind: 6维风力向量 [X, Y, Z, K, M, N]

    物理模型:
        相对风速: V_rw = V_w - V_ship（在风坐标系下）
        风力 = 0.5 * rho_air * V_rw^2 * C * A
    """
    tau_wind = np.zeros(6)

    if V_w < 0.01:  # 风速太小，忽略
        return tau_wind

    rho_air = 1.225  # 空气密度 (kg/m³)

    # 计算相对风速和风向（船体坐标系）
    # 风向相对于船体的角度
    gamma_w = beta_w - psi  # 相对风向角

    # 风在船体坐标系的分量
    u_w = V_w * np.cos(gamma_w)  # 纵向风速分量
    v_w = V_w * np.sin(gamma_w)  # 横向风速分量

    # 相对风速（考虑船速）
    u_rw = u_w - nu[0]  # 相对纵向风速
    v_rw = v_w - nu[1]  # 相对横向风速

    V_rw = np.sqrt(u_rw**2 + v_rw**2)  # 相对风速大小

    if V_rw < 0.01:
        return tau_wind

    # 相对风向角
    gamma_rw = np.arctan2(v_rw, u_rw)

    # 投影面积
    A_Fw = B * H_freeboard  # 正面投影面积（船宽 × 干舷）
    A_Lw = L * H_freeboard  # 侧面投影面积（船长 × 干舷）

    # 风力系数（简化模型，基于相对风向角）
    # C_X: 纵向力系数（迎风为负，顺风为正）
    # C_Y: 横向力系数
    # C_N: 艏摇力矩系数
    cos_gamma = np.cos(gamma_rw)
    sin_gamma = np.sin(gamma_rw)

    C_X = -0.9 * cos_gamma  # 纵向系数
    C_Y = 0.95 * sin_gamma  # 横向系数
    C_N = 0.20 * sin_gamma * cos_gamma  # 艏摇系数（船体不对称时产生力矩）

    # 风力计算
    q = 0.5 * rho_air * V_rw**2  # 动压

    X_wind = q * C_X * A_Fw  # 纵向风力 (N)
    Y_wind = q * C_Y * A_Lw  # 横向风力 (N)
    N_wind = q * C_N * A_Lw * L  # 艏摇力矩 (N·m)

    tau_wind[0] = X_wind
    tau_wind[1] = Y_wind
    tau_wind[5] = N_wind

    return tau_wind


# ==================== Otter USV 核心模型 ====================

class OtterUSVCore:
    """Otter USV核心动力学模型"""

    def __init__(self):
        # 主要参数（官方数据）
        self.g = 9.81
        self.rho = 1025
        self.L = 2.0  # 船长 (m)
        self.B = 1.08  # 船宽 (m)
        self.m = 55.0  # 质量 (kg, 无负载)
        self.draft = 0.134  # 吃水 (m, 无负载)
        self.rg = np.array([0.2, 0, -0.2])
        self.R44 = 0.4 * self.B
        self.R55 = 0.25 * self.L
        self.R66 = 0.25 * self.L
        self.T_sway = 1.0
        self.T_yaw = 1.0
        self.Umax = 3.0  # 最大速度 (m/s, 官方数据)

        # 浮体参数
        self.B_pont = 0.25
        self.y_pont = 0.395
        self.Cw_pont = 0.75
        self.Cb_pont = 0.4

        # 螺旋桨参数（官方数据）
        self.l1 = -self.y_pont  # 左舷螺旋桨横向位置
        self.l2 = self.y_pont  # 右舷螺旋桨横向位置

        # 官方螺旋桨转速限制
        # 正转：993 RPM = 993 * 2π/60 ≈ 104.0 rad/s
        # 反转：972 RPM = 972 * 2π/60 ≈ 101.8 rad/s
        self.n_max = 993.0 * 2 * np.pi / 60.0  # ≈ 104.0 rad/s
        self.n_min = -972.0 * 2 * np.pi / 60.0  # ≈ -101.8 rad/s

        # 推力系数（保持原有值，可能需要根据实际调整）
        self.k_pos = 0.02216 / 2
        self.k_neg = 0.01289 / 2

    def dynamics(self, x, n, mp=0, rp=None, V_c=0, beta_c=0, V_w=0, beta_w=0):
        """
        计算状态导数

        Args:
            x: 12维状态向量 [u,v,w,p,q,r,x,y,z,phi,theta,psi]
            n: 2维控制输入 [n_port, n_starboard] (rad/s)
            mp: 额外负载质量 (kg)
            rp: 额外负载位置 (m)
            V_c: 海流速度 (m/s)
            beta_c: 海流方向 (rad)
            V_w: 风速 (m/s)
            beta_w: 风向 (rad)，相对于NED坐标系

        注：官方数据
            - 无负载吃水: 13.4 cm
            - 25kg负载吃水: 19.5 cm
            当mp>0时会自动调整吃水深度
        """
        if rp is None:
            rp = np.zeros(3)

        # 数值保护：限制状态量范围
        x = np.clip(x, -1e6, 1e6)

        # 状态变量
        nu = x[0:6].copy()
        nu1 = x[0:3]
        nu2 = x[3:6]
        eta = x[6:12].copy()

        # 限制速度和角速度范围
        nu[0] = np.clip(nu[0], -10, 10)
        nu[1] = np.clip(nu[1], -5, 5)
        nu[2] = np.clip(nu[2], -5, 5)
        nu[3] = np.clip(nu[3], -2, 2)
        nu[4] = np.clip(nu[4], -2, 2)
        nu[5] = np.clip(nu[5], -2, 2)

        # 限制姿态角范围
        eta[3] = np.clip(eta[3], -np.pi / 3, np.pi / 3)
        eta[4] = np.clip(eta[4], -np.pi / 3, np.pi / 3)

        U = np.sqrt(nu[0] ** 2 + nu[1] ** 2 + nu[2] ** 2)
        u_c = V_c * np.cos(beta_c - eta[5])
        v_c = V_c * np.sin(beta_c - eta[5])
        nu_c = np.array([u_c, v_c, 0, 0, 0, 0])
        nu_r = nu - nu_c
        nu_c_dot = np.concatenate([
            -Smtrx(nu2) @ nu_c[0:3],
            np.zeros(3)
        ])

        # 质量和惯性
        nabla = (self.m + mp) / self.rho
        T = nabla / (2 * self.Cb_pont * self.B_pont * self.L)
        Ig_CG = self.m * np.diag([self.R44 ** 2, self.R55 ** 2, self.R66 ** 2])
        rg = (self.m * self.rg + mp * rp) / (self.m + mp)
        Ig = Ig_CG - self.m * Smtrx(rg) @ Smtrx(rg) - mp * Smtrx(rp) @ Smtrx(rp)

        # 刚体质量矩阵
        I3 = np.eye(3)
        O3 = np.zeros((3, 3))
        MRB_CG = np.block([
            [(self.m + mp) * I3, O3],
            [O3, Ig]
        ])
        CRB_CG = np.block([
            [(self.m + mp) * Smtrx(nu2), O3],
            [O3, -Smtrx(Ig @ nu2)]
        ])

        H = Hmtrx(rg)
        MRB = H.T @ MRB_CG @ H
        CRB = H.T @ CRB_CG @ H

        # 附加质量
        Xudot = -addedMassSurge(self.m, self.L, self.rho)
        Yvdot = -1.5 * self.m
        Zwdot = -1.0 * self.m
        Kpdot = -0.2 * Ig[0, 0]
        Mqdot = -0.8 * Ig[1, 1]
        Nrdot = -1.7 * Ig[2, 2]

        MA = -np.diag([Xudot, Yvdot, Zwdot, Kpdot, Mqdot, Nrdot])
        CA = m2c(MA, nu_r)

        M = MRB + MA
        C = CRB + CA

        # 静水力
        Aw_pont = self.Cw_pont * self.L * self.B_pont
        I_T = 2 * (1 / 12) * self.L * self.B_pont ** 3 * \
              (6 * self.Cw_pont ** 3 / ((1 + self.Cw_pont) * (1 + 2 * self.Cw_pont))) + \
              2 * Aw_pont * self.y_pont ** 2
        I_L = 0.8 * 2 * (1 / 12) * self.B_pont * self.L ** 3
        KB = (1 / 3) * (5 * T / 2 - 0.5 * nabla / (self.L * self.B_pont))
        BM_T = I_T / nabla
        BM_L = I_L / nabla
        KM_T = KB + BM_T
        KM_L = KB + BM_L
        KG = T - rg[2]
        GM_T = KM_T - KG
        GM_L = KM_L - KG

        G33 = self.rho * self.g * (2 * Aw_pont)
        G44 = self.rho * self.g * nabla * GM_T
        G55 = self.rho * self.g * nabla * GM_L

        G_CF = np.diag([0, 0, G33, G44, G55, 0])
        LCF = -0.2
        H_CF = Hmtrx(np.array([LCF, 0, 0]))
        G = H_CF.T @ G_CF @ H_CF

        # 阻尼
        w3 = np.sqrt(G33 / M[2, 2])
        w4 = np.sqrt(G44 / M[3, 3])
        w5 = np.sqrt(G55 / M[4, 4])

        Xu = -24.4 * self.g / self.Umax
        Yv = -M[1, 1] / self.T_sway
        Zw = -2 * 0.3 * w3 * M[2, 2]
        Kp = -2 * 0.2 * w4 * M[3, 3]
        Mq = -2 * 0.4 * w5 * M[4, 4]
        Nr = -M[5, 5] / self.T_yaw

        # 推力计算
        Thrust = np.zeros(2)
        for i in range(2):
            n_sat = np.clip(n[i], self.n_min, self.n_max)
            if n_sat > 0:
                Thrust[i] = self.k_pos * n_sat * abs(n_sat)
            else:
                Thrust[i] = self.k_neg * n_sat * abs(n_sat)

        tau = np.array([
            Thrust[0] + Thrust[1],
            0, 0, 0, 0,
            -self.l1 * Thrust[0] - self.l2 * Thrust[1]
        ])

        # 阻尼力
        Xh = Xu * np.clip(nu_r[0], -10, 10)
        Yh = Yv * np.clip(nu_r[1], -5, 5)
        Zh = Zw * np.clip(nu_r[2], -5, 5)
        Kh = Kp * np.clip(nu_r[3], -2, 2)
        Mh = Mq * np.clip(nu_r[4], -2, 2)
        r_clip = np.clip(nu_r[5], -2, 2)
        Nh = Nr * (1 + 10 * abs(r_clip)) * r_clip
        tau_damp = np.array([Xh, Yh, Zh, Kh, Mh, Nh])

        # 横流阻力
        tau_crossflow = crossFlowDrag(self.L, self.B_pont, T, nu_r)

        # 风力干扰
        H_freeboard = 0.3  # 估计干舷高度 (m)，Otter USV 约30cm
        tau_wind = windForce(self.L, self.B, H_freeboard, nu, eta[5], V_w, beta_w)

        # 载荷
        f_payload = Rzyx(eta[3], eta[4], eta[5]).T @ np.array([0, 0, mp * self.g])
        m_payload = Smtrx(rp) @ f_payload
        g_0 = np.concatenate([f_payload, m_payload])

        # 平衡位置
        eta_0 = np.zeros(6)
        eta_0[2:5] = np.linalg.inv(G[2:5, 2:5]) @ g_0[2:5]
        eta_shifted = eta - eta_0

        # 运动学变换
        J = eulerang(eta[3], eta[4], eta[5])

        # 状态导数
        try:
            force = tau + tau_damp + tau_crossflow + tau_wind - C @ nu_r - G @ eta_shifted
            force = np.clip(force, -1e5, 1e5)
            nu_dot = nu_c_dot + np.linalg.solve(M, force)
            nu_dot = np.clip(nu_dot, -10, 10)
        except np.linalg.LinAlgError:
            nu_dot = np.zeros(6)

        eta_dot = J @ nu
        eta_dot = np.clip(eta_dot, -50, 50)

        return np.concatenate([nu_dot, eta_dot])


# ==================== 包装类：双螺旋桨控制 ====================

class OtterUSV_DualPropeller:
    """
    Otter USV包装类 - 原生双螺旋桨控制

    控制输入：
    - n_port: 左舷螺旋桨转速 (rad/s)
    - n_starboard: 右舷螺旋桨转速 (rad/s)

    状态输出（简化为3自由度）：
    - u: 纵向速度 (surge)
    - vm: 船中处横向速度 (sway at midship)
    - r: 转艏角速度 (yaw rate)
    - x, y: 位置
    - psi: 航向角
    """

    def __init__(self, verbose=True):
        self.core = OtterUSVCore()
        self.verbose = verbose

        # 船舶参数
        self.L = self.core.L
        self.B = self.core.B
        self.xG = self.core.rg[0]  # 重心纵向位置

        # 创建伪参数对象p（用于环境访问）
        class PseudoParams:
            def __init__(self, xG):
                self.xG = xG

        self.p = PseudoParams(self.xG)

        # 12维完整状态
        self.full_state = np.zeros(12)

        # 简化状态字典
        self.state = {
            'u': 0.0,  # surge velocity
            'vm': 0.0,  # sway velocity at midship
            'r': 0.0,  # yaw rate
            'x': 0.0,  # x position
            'y': 0.0,  # y position
            'psi': 0.0,  # heading
        }

        # 控制历史（用于记录）
        self.last_n_port = 0.0
        self.last_n_starboard = 0.0

        if verbose:
            self._print_info()

    def _print_info(self):
        """打印船舶信息"""
        print(f"[OtterUSV_DualPropeller] 初始化完成")
        print(f"  船长: {self.L:.2f} m")
        print(f"  船宽: {self.B:.2f} m")
        print(f"  质量: {self.core.m:.1f} kg (无负载)")
        print(f"  吃水: {self.core.draft:.3f} m")
        print(f"  最大速度: {self.core.Umax:.1f} m/s")
        print(f"  重心位置xG: {self.xG:.3f} m")
        print(f"  螺旋桨转速范围: [{self.core.n_min:.1f}, {self.core.n_max:.1f}] rad/s")
        print(
            f"  螺旋桨转速范围: [{self.core.n_min * 60 / (2 * np.pi):.0f}, {self.core.n_max * 60 / (2 * np.pi):.0f}] RPM")
        print(f"  控制方式: 双螺旋桨独立控制（无舵）")
        print(f"  左舷螺旋桨位置: y = {-self.core.y_pont:.3f} m")
        print(f"  右舷螺旋桨位置: y = {self.core.y_pont:.3f} m")

    def reset(self, initial_state=None):
        """重置到初始状态"""
        if initial_state is None:
            # 默认初始状态
            self.full_state = np.zeros(12)
            self.full_state[0] = 0.1  # 初始纵向速度

            self.state = {
                'u': 0.1,
                'vm': 0.0,
                'r': 0.0,
                'x': 0.0,
                'y': 0.0,
                'psi': 0.0,
            }
        else:
            # 从简化状态初始化完整状态
            self.full_state = np.zeros(12)
            self.full_state[0] = initial_state.get('u', 0.0)  # u
            self.full_state[1] = initial_state.get('vm', 0.0)  # v (近似为vm)
            self.full_state[5] = initial_state.get('r', 0.0)  # r
            self.full_state[6] = initial_state.get('x', 0.0)  # x
            self.full_state[7] = initial_state.get('y', 0.0)  # y
            self.full_state[11] = initial_state.get('psi', 0.0)  # psi

            self.state = initial_state.copy()

        # 重置控制历史
        self.last_n_port = 0.0
        self.last_n_starboard = 0.0

        return self.state.copy()

    def step(self, n_port, n_starboard, dt=0.1, V_c=0.0, beta_c=0.0, V_w=0.0, beta_w=0.0):
        """
        执行一步仿真

        Args:
            n_port: 左舷螺旋桨转速 (rad/s)
            n_starboard: 右舷螺旋桨转速 (rad/s)
            dt: 时间步长 (s)
            V_c: 海流速度 (m/s)，默认0表示无海流
            beta_c: 海流方向 (rad)，相对于NED坐标系
            V_w: 风速 (m/s)，默认0表示无风
            beta_w: 风向 (rad)，相对于NED坐标系

        Returns:
            简化的状态字典
        """
        # 记录控制输入
        self.last_n_port = n_port
        self.last_n_starboard = n_starboard

        # 限制在物理范围内
        n_port = np.clip(n_port, self.core.n_min, self.core.n_max)
        n_starboard = np.clip(n_starboard, self.core.n_min, self.core.n_max)

        n = np.array([n_port, n_starboard])

        # 使用RK4积分（传入海流和风参数）
        x = self.full_state.copy()

        k1 = self.core.dynamics(x, n, V_c=V_c, beta_c=beta_c, V_w=V_w, beta_w=beta_w)
        k2 = self.core.dynamics(x + 0.5 * dt * k1, n, V_c=V_c, beta_c=beta_c, V_w=V_w, beta_w=beta_w)
        k3 = self.core.dynamics(x + 0.5 * dt * k2, n, V_c=V_c, beta_c=beta_c, V_w=V_w, beta_w=beta_w)
        k4 = self.core.dynamics(x + dt * k3, n, V_c=V_c, beta_c=beta_c, V_w=V_w, beta_w=beta_w)

        self.full_state = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # 更新简化状态
        self.state['u'] = self.full_state[0]
        self.state['vm'] = self.full_state[1]  # Otter的v近似为vm
        self.state['r'] = self.full_state[5]
        self.state['x'] = self.full_state[6]
        self.state['y'] = self.full_state[7]
        self.state['psi'] = self.full_state[11]

        # 归一化航向角到[-pi, pi]
        while self.state['psi'] > np.pi:
            self.state['psi'] -= 2 * np.pi
        while self.state['psi'] < -np.pi:
            self.state['psi'] += 2 * np.pi

        return self.state.copy()

    def get_state_vector(self):
        """获取状态向量（6维）"""
        return np.array([
            self.state['x'],
            self.state['y'],
            self.state['psi'],
            self.state['u'],
            self.state['vm'],
            self.state['r']
        ])

    def get_control_vector(self):
        """获取控制向量"""
        return np.array([self.last_n_port, self.last_n_starboard])