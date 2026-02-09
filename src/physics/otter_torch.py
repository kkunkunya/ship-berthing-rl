"""
Otter USV 物理仿真 - PyTorch GPU 向量化版本

将原始 otter.py 的 NumPy 实现转换为 PyTorch，支持：
1. GPU 加速计算
2. 批量并行仿真（同时计算 N 个环境）
3. JIT 编译优化

核心改动：
- 所有 numpy 操作 → torch tensor 操作
- 所有函数支持 batch dimension (B, ...)
- 使用 @torch.jit.script 加速
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import math
import numpy as np

# 默认设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


# ==================== 辅助函数（批量版本）====================

@torch.jit.script
def smtrx_batch(a: torch.Tensor) -> torch.Tensor:
    """
    批量反对称矩阵 (skew-symmetric matrix)

    Args:
        a: (B, 3) 向量

    Returns:
        (B, 3, 3) 反对称矩阵
    """
    B = a.shape[0]
    zeros = torch.zeros(B, device=a.device, dtype=a.dtype)

    # 构建反对称矩阵
    S = torch.stack([
        torch.stack([zeros, -a[:, 2], a[:, 1]], dim=1),
        torch.stack([a[:, 2], zeros, -a[:, 0]], dim=1),
        torch.stack([-a[:, 1], a[:, 0], zeros], dim=1)
    ], dim=1)

    return S


@torch.jit.script
def hmtrx_batch(r: torch.Tensor) -> torch.Tensor:
    """
    批量 6x6 变换矩阵

    Args:
        r: (B, 3) 向量

    Returns:
        (B, 6, 6) 变换矩阵
    """
    B = r.shape[0]
    H = torch.eye(6, device=r.device, dtype=r.dtype).unsqueeze(0).expand(B, -1, -1).clone()
    H[:, 0:3, 3:6] = smtrx_batch(r)
    return H


@torch.jit.script
def eulerang_batch(phi: torch.Tensor, theta: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """
    批量欧拉角运动学矩阵

    Args:
        phi, theta, psi: (B,) 欧拉角

    Returns:
        (B, 6, 6) 运动学矩阵
    """
    B = phi.shape[0]

    cphi = torch.cos(phi)
    sphi = torch.sin(phi)
    cth = torch.cos(theta)
    sth = torch.sin(theta)
    tth = torch.tan(theta)
    cpsi = torch.cos(psi)
    spsi = torch.sin(psi)

    zeros = torch.zeros(B, device=phi.device, dtype=phi.dtype)
    ones = torch.ones(B, device=phi.device, dtype=phi.dtype)

    # 构建 6x6 矩阵
    J = torch.zeros(B, 6, 6, device=phi.device, dtype=phi.dtype)

    # 上 3x3 块（位置）
    J[:, 0, 0] = cpsi * cth
    J[:, 0, 1] = -spsi * cphi + cpsi * sth * sphi
    J[:, 0, 2] = spsi * sphi + cpsi * cphi * sth
    J[:, 1, 0] = spsi * cth
    J[:, 1, 1] = cpsi * cphi + sphi * sth * spsi
    J[:, 1, 2] = -cpsi * sphi + sth * spsi * cphi
    J[:, 2, 0] = -sth
    J[:, 2, 1] = cth * sphi
    J[:, 2, 2] = cth * cphi

    # 下 3x3 块（姿态）
    J[:, 3, 3] = ones
    J[:, 3, 4] = sphi * tth
    J[:, 3, 5] = cphi * tth
    J[:, 4, 4] = cphi
    J[:, 4, 5] = -sphi
    J[:, 5, 4] = sphi / cth
    J[:, 5, 5] = cphi / cth

    return J


@torch.jit.script
def m2c_batch(MA: torch.Tensor, nu_r: torch.Tensor) -> torch.Tensor:
    """
    批量附加质量矩阵转为科氏矩阵

    Args:
        MA: (B, 6, 6) 附加质量矩阵
        nu_r: (B, 6) 相对速度

    Returns:
        (B, 6, 6) 科氏矩阵
    """
    B = nu_r.shape[0]
    CA = torch.zeros(B, 6, 6, device=nu_r.device, dtype=nu_r.dtype)

    # MA @ nu_r 的前3个和后3个分量
    temp = torch.bmm(MA[:, 0:3, 0:3], nu_r[:, 0:3].unsqueeze(-1)).squeeze(-1) + \
           torch.bmm(MA[:, 0:3, 3:6], nu_r[:, 3:6].unsqueeze(-1)).squeeze(-1)

    S_temp = smtrx_batch(temp)

    CA[:, 0:3, 3:6] = -S_temp
    CA[:, 3:6, 0:3] = -S_temp

    temp2 = torch.bmm(MA[:, 3:6, 0:3], nu_r[:, 0:3].unsqueeze(-1)).squeeze(-1) + \
            torch.bmm(MA[:, 3:6, 3:6], nu_r[:, 3:6].unsqueeze(-1)).squeeze(-1)
    CA[:, 3:6, 3:6] = -smtrx_batch(temp2)

    return CA


@torch.jit.script
def cross_flow_drag_batch(L: float, B: float, T: torch.Tensor, nu_r: torch.Tensor) -> torch.Tensor:
    """
    批量横流阻力计算

    Args:
        L, B: 船长、船宽（标量）
        T: (B_batch,) 吃水
        nu_r: (B_batch, 6) 相对速度

    Returns:
        (B_batch, 6) 阻力
    """
    batch_size = nu_r.shape[0]
    tau = torch.zeros(batch_size, 6, device=nu_r.device, dtype=nu_r.dtype)

    Cd = 1.0
    rho = 1025.0

    v = torch.clamp(nu_r[:, 1], -10.0, 10.0)
    r = torch.clamp(nu_r[:, 5], -5.0, 5.0)

    tau[:, 1] = -0.5 * rho * Cd * B * T * L * torch.abs(v) * v
    tau[:, 5] = -0.5 * rho * Cd * B * T * L * L * torch.abs(r) * r / 12.0

    return tau


@torch.jit.script
def wind_force_batch(
    L: float, B: float, H_freeboard: float,
    nu: torch.Tensor, psi: torch.Tensor,
    V_w: torch.Tensor, beta_w: torch.Tensor
) -> torch.Tensor:
    """
    批量风力计算

    Args:
        L, B, H_freeboard: 船舶几何参数（标量）
        nu: (B, 6) 速度向量
        psi: (B,) 航向角
        V_w: (B,) 风速
        beta_w: (B,) 风向

    Returns:
        (B, 6) 风力向量
    """
    batch_size = nu.shape[0]
    tau_wind = torch.zeros(batch_size, 6, device=nu.device, dtype=nu.dtype)

    # 有风的 mask
    has_wind = V_w > 1e-6
    if not has_wind.any():
        return tau_wind

    # 空气密度
    rho_air = 1.225

    # 投影面积
    A_Fw = L * H_freeboard  # 正面
    A_Lw = B * H_freeboard  # 侧面

    # 船舶速度分量
    u = nu[:, 0]
    v = nu[:, 1]

    # 相对风向角（船体坐标系）
    gamma_w = beta_w - psi

    # 相对风速分量
    u_w = V_w * torch.cos(gamma_w)
    v_w = V_w * torch.sin(gamma_w)

    # 相对风速
    u_rw = u_w - u
    v_rw = v_w - v
    V_rw = torch.sqrt(u_rw * u_rw + v_rw * v_rw + 1e-8)

    # 相对风向
    gamma_rw = torch.atan2(v_rw, u_rw)

    # 风力系数
    cos_gamma = torch.cos(gamma_rw)
    sin_gamma = torch.sin(gamma_rw)

    Cx = -0.5 * cos_gamma
    Cy = 0.7 * sin_gamma
    Cn = 0.15 * sin_gamma

    # 动压
    q = 0.5 * rho_air * V_rw * V_rw

    # 风力
    tau_wind[:, 0] = q * Cx * A_Fw * has_wind.float()
    tau_wind[:, 1] = q * Cy * A_Lw * has_wind.float()
    tau_wind[:, 5] = q * Cn * A_Lw * L * has_wind.float()

    return tau_wind


class OtterTorchCore:
    """
    Otter USV 物理核心 - PyTorch 批量版本

    支持同时计算 N 个环境的物理仿真
    """

    def __init__(self, batch_size: int = 1, device: torch.device = DEVICE):
        self.batch_size = batch_size
        self.device = device

        # ========== 船舶参数（与原版相同）==========
        self.L = 2.0  # 船长 (m)
        self.B = 1.08  # 船宽 (m)
        self.B_pont = 0.25  # 浮筒宽度
        self.y_pont = 0.395  # 浮筒中心到船中距离
        self.Cw_pont = 0.75  # 浮筒水线面系数
        self.Cb_pont = 0.4  # 浮筒方形系数

        self.m = 55.0  # 质量 (kg)
        self.rho = 1025.0  # 海水密度

        # 重心位置
        self.rg = torch.tensor([0.2, 0.0, -0.2], device=device, dtype=DTYPE)

        # 回转半径 (与原版一致: R44=0.4*B, R55=R66=0.25*L)
        self.R44 = 0.4 * self.B   # = 0.432
        self.R55 = 0.25 * self.L  # = 0.5
        self.R66 = 0.25 * self.L  # = 0.5

        # 干舷高度（用于风力计算）
        self.H_freeboard = 0.3

        # 螺旋桨参数 (与原版一致)
        self.n_max = 993.0 * 2 * math.pi / 60.0  # 最大转速 ≈104 rad/s
        self.n_min = -972.0 * 2 * math.pi / 60.0  # 最小转速 ≈-102 rad/s
        self.k_pos = 0.02216 / 2  # 正向推力系数
        self.k_neg = 0.01289 / 2  # 反向推力系数

        # 默认吃水
        self.draft = 0.134

        # 预计算不变量
        self._precompute_constants()

    def _precompute_constants(self):
        """预计算不随状态变化的常量"""
        # 惯性矩阵对角元 (CG 坐标系)
        self.Ig_diag = self.m * torch.tensor(
            [self.R44**2, self.R55**2, self.R66**2],
            device=self.device, dtype=DTYPE
        )

        # 完整惯性矩阵 Ig = Ig_CG - m * S(rg) @ S(rg)
        # 原版: Ig = Ig_CG - self.m * Smtrx(rg) @ Smtrx(rg)
        rg_np = self.rg.cpu().numpy()
        S_rg = np.array([
            [0, -rg_np[2], rg_np[1]],
            [rg_np[2], 0, -rg_np[0]],
            [-rg_np[1], rg_np[0], 0]
        ])
        Ig_CG = self.m * np.diag([self.R44**2, self.R55**2, self.R66**2])
        Ig_np = Ig_CG - self.m * (S_rg @ S_rg)
        self.Ig = torch.tensor(Ig_np, device=self.device, dtype=DTYPE)

        # 附加质量系数（使用完整 Ig 的对角元素）
        self.Xudot = -0.1 * self.m
        self.Yvdot = -1.5 * self.m
        self.Zwdot = -1.0 * self.m
        self.Kpdot = -0.2 * self.Ig[0, 0].item()  # 使用 Ig 而不是 Ig_diag
        self.Mqdot = -0.8 * self.Ig[1, 1].item()
        self.Nrdot = -1.7 * self.Ig[2, 2].item()

        # 附加质量矩阵（对角，不随状态变化）
        self.MA = -torch.diag(torch.tensor(
            [self.Xudot, self.Yvdot, self.Zwdot, self.Kpdot, self.Mqdot, self.Nrdot],
            device=self.device, dtype=DTYPE
        ))

        # ========== 静水力常量（与原版一致）==========
        g = 9.81
        rho = self.rho

        # 浮体参数（无负载）
        mp = 0  # 额外负载
        nabla = (self.m + mp) / rho
        T = nabla / (2 * self.Cb_pont * self.B_pont * self.L)

        # 重心（无负载）
        rg = self.rg.cpu().numpy()

        # 水线面积和惯性矩
        Aw_pont = self.Cw_pont * self.L * self.B_pont
        I_T = 2 * (1/12) * self.L * self.B_pont**3 * \
              (6 * self.Cw_pont**3 / ((1 + self.Cw_pont) * (1 + 2*self.Cw_pont))) + \
              2 * Aw_pont * self.y_pont**2
        I_L = 0.8 * 2 * (1/12) * self.B_pont * self.L**3

        # 稳心计算
        KB = (1/3) * (5*T/2 - 0.5 * nabla / (self.L * self.B_pont))
        BM_T = I_T / nabla
        BM_L = I_L / nabla
        KM_T = KB + BM_T
        KM_L = KB + BM_L
        KG = T - rg[2]
        GM_T = KM_T - KG
        GM_L = KM_L - KG

        # 静水力系数
        self.G33 = rho * g * (2 * Aw_pont)  # heave
        self.G44 = rho * g * nabla * GM_T   # roll
        self.G55 = rho * g * nabla * GM_L   # pitch

        # 静水力矩阵（浮心坐标系）
        G_CF = np.diag([0, 0, self.G33, self.G44, self.G55, 0])

        # LCF 变换
        LCF = -0.2
        H_CF = np.eye(6)
        H_CF[0:3, 3:6] = np.array([
            [0, 0, 0],
            [0, 0, -LCF],
            [0, LCF, 0]
        ])
        G_np = H_CF.T @ G_CF @ H_CF
        self.G = torch.tensor(G_np, device=self.device, dtype=DTYPE)

        # 预计算 MRB（用于计算阻尼系数需要的 M 矩阵对角元素）
        # 这里需要完整计算一次 M 矩阵来获取对角元素
        Ig_np = self.Ig.cpu().numpy()
        MRB_CG = np.block([
            [self.m * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), Ig_np]
        ])
        S_rg = np.array([
            [0, -rg[2], rg[1]],
            [rg[2], 0, -rg[0]],
            [-rg[1], rg[0], 0]
        ])
        H = np.eye(6)
        H[0:3, 3:6] = S_rg
        MRB_np = H.T @ MRB_CG @ H
        MA_np = -np.diag([self.Xudot, self.Yvdot, self.Zwdot, self.Kpdot, self.Mqdot, self.Nrdot])
        M_np = MRB_np + MA_np

        # 自然频率和阻尼系数（与原版一致）
        self.w3 = math.sqrt(self.G33 / M_np[2, 2])  # heave
        self.w4 = math.sqrt(self.G44 / M_np[3, 3])  # roll
        self.w5 = math.sqrt(self.G55 / M_np[4, 4])  # pitch

        # 预计算阻尼系数（与原版一致）
        self.Zw_coef = -2 * 0.3 * self.w3  # heave: Zw = Zw_coef * M[2,2]
        self.Kp_coef = -2 * 0.2 * self.w4  # roll:  Kp = Kp_coef * M[3,3]
        self.Mq_coef = -2 * 0.4 * self.w5  # pitch: Mq = Mq_coef * M[4,4]

    def dynamics_batch(
        self,
        x: torch.Tensor,  # (B, 12) 状态
        n: torch.Tensor,  # (B, 2) 控制输入 [n_port, n_starboard]
        V_c: torch.Tensor,  # (B,) 海流速度
        beta_c: torch.Tensor,  # (B,) 海流方向
        V_w: torch.Tensor,  # (B,) 风速
        beta_w: torch.Tensor  # (B,) 风向
    ) -> torch.Tensor:
        """
        批量计算状态导数

        Returns:
            (B, 12) 状态导数
        """
        B = x.shape[0]

        # 数值保护
        x = torch.clamp(x, -1e6, 1e6)

        # 状态解包
        nu = x[:, 0:6].clone()
        eta = x[:, 6:12].clone()

        # 限制速度和角速度
        nu[:, 0] = torch.clamp(nu[:, 0], -10.0, 10.0)
        nu[:, 1] = torch.clamp(nu[:, 1], -5.0, 5.0)
        nu[:, 2] = torch.clamp(nu[:, 2], -5.0, 5.0)
        nu[:, 3] = torch.clamp(nu[:, 3], -2.0, 2.0)
        nu[:, 4] = torch.clamp(nu[:, 4], -2.0, 2.0)
        nu[:, 5] = torch.clamp(nu[:, 5], -2.0, 2.0)

        # 限制姿态角
        eta[:, 3] = torch.clamp(eta[:, 3], -math.pi/3, math.pi/3)
        eta[:, 4] = torch.clamp(eta[:, 4], -math.pi/3, math.pi/3)

        psi = eta[:, 5]

        # 海流影响
        u_c = V_c * torch.cos(beta_c - psi)
        v_c = V_c * torch.sin(beta_c - psi)
        nu_c = torch.zeros(B, 6, device=self.device, dtype=DTYPE)
        nu_c[:, 0] = u_c
        nu_c[:, 1] = v_c

        # 相对速度
        nu_r = nu - nu_c

        # 海流加速度项（与原版一致）
        # nu_c_dot = [-Smtrx(nu2) @ nu_c[0:3], zeros(3)]
        # 当船舶转向时，海流的相对速度会变化
        nu2 = nu[:, 3:6]  # 角速度 [p, q, r]
        S_nu2 = smtrx_batch(nu2)  # (B, 3, 3) 反对称矩阵
        nu_c_dot = torch.zeros(B, 6, device=self.device, dtype=DTYPE)
        # -S(nu2) @ nu_c[0:3]
        nu_c_dot[:, 0:3] = -torch.bmm(S_nu2, nu_c[:, 0:3].unsqueeze(-1)).squeeze(-1)

        # ========== 质量和惯性 ==========
        # 简化：假设无额外负载
        T = self.draft * torch.ones(B, device=self.device, dtype=DTYPE)

        # 刚体质量矩阵（完整 H 变换，与原版一致）
        # 原版: MRB_CG = [[m*I, O], [O, Ig]], 然后 MRB = H.T @ MRB_CG @ H
        # H 变换会产生非对角耦合项

        # 构建 MRB_CG
        MRB_CG = torch.zeros(B, 6, 6, device=self.device, dtype=DTYPE)
        MRB_CG[:, 0, 0] = self.m
        MRB_CG[:, 1, 1] = self.m
        MRB_CG[:, 2, 2] = self.m
        # 使用完整的 Ig 矩阵（包含非对角项）
        MRB_CG[:, 3:6, 3:6] = self.Ig.unsqueeze(0).expand(B, -1, -1)

        # H 变换
        rg_tensor = self.rg.clone().detach().unsqueeze(0).expand(B, -1)
        H = hmtrx_batch(rg_tensor)
        H_T = H.transpose(1, 2)
        MRB = torch.bmm(torch.bmm(H_T, MRB_CG), H)

        # 总质量矩阵（现在包含完整的非对角耦合项）
        MA_expanded = self.MA.unsqueeze(0).expand(B, -1, -1)
        M = MRB + MA_expanded

        # ========== 科氏力矩阵 ==========
        # 附加质量科氏力
        CA = m2c_batch(MA_expanded, nu_r)

        # 刚体科氏力矩阵 CRB (与原版一致)
        # 原版: CRB_CG = [[m*S(nu2), O3], [O3, -S(Ig@nu2)]]
        #       CRB = H.T @ CRB_CG @ H
        nu2 = nu[:, 3:6]  # 角速度 [p, q, r]

        # Ig @ nu2 (使用完整惯性矩阵，包含非对角项)
        Ig_expanded = self.Ig.unsqueeze(0).expand(B, -1, -1)  # (B, 3, 3)
        Ig_nu2 = torch.bmm(Ig_expanded, nu2.unsqueeze(-1)).squeeze(-1)  # (B, 3)

        # 构建 CRB_CG 的两个 3x3 块
        S_nu2 = smtrx_batch(nu2)  # S(nu2)
        S_Ig_nu2 = smtrx_batch(Ig_nu2)  # S(Ig @ nu2)

        CRB_CG = torch.zeros(B, 6, 6, device=self.device, dtype=DTYPE)
        CRB_CG[:, 0:3, 0:3] = self.m * S_nu2  # 左上块: m * S(nu2)
        CRB_CG[:, 3:6, 3:6] = -S_Ig_nu2  # 右下块: -S(Ig @ nu2)

        # H 变换
        rg_tensor = self.rg.clone().detach().unsqueeze(0).expand(B, -1)
        H = hmtrx_batch(rg_tensor)
        H_T = H.transpose(1, 2)
        CRB = torch.bmm(torch.bmm(H_T, CRB_CG), H)

        # 总科氏力矩阵 C = CRB + CA
        C_total = CRB + CA

        # ========== 阻尼 (与原版一致的非线性模型) ==========
        # 原版参数
        g = 9.81
        Umax = 3.0
        T_sway = 1.0
        T_yaw = 1.0

        # 线性阻尼系数 (与原版一致)
        Xu = -24.4 * g / Umax  # surge
        Yv = -M[:, 1, 1] / T_sway  # sway
        Nr = -M[:, 5, 5] / T_yaw  # yaw

        # 阻尼力 (直接计算，不使用矩阵)
        tau_damp = torch.zeros(B, 6, device=self.device, dtype=DTYPE)

        # 限制速度范围（数值稳定性）
        u_r = torch.clamp(nu_r[:, 0], -10.0, 10.0)
        v_r = torch.clamp(nu_r[:, 1], -5.0, 5.0)
        w_r = torch.clamp(nu_r[:, 2], -5.0, 5.0)
        p_r = torch.clamp(nu_r[:, 3], -2.0, 2.0)
        q_r = torch.clamp(nu_r[:, 4], -2.0, 2.0)
        r_r = torch.clamp(nu_r[:, 5], -2.0, 2.0)

        # 垂荡和横摇/纵摇阻尼（使用基于自然频率的系数，与原版一致）
        Zw = self.Zw_coef * M[:, 2, 2]  # heave
        Kp = self.Kp_coef * M[:, 3, 3]  # roll
        Mq = self.Mq_coef * M[:, 4, 4]  # pitch

        tau_damp[:, 0] = Xu * u_r
        tau_damp[:, 1] = Yv * v_r
        tau_damp[:, 2] = Zw * w_r
        tau_damp[:, 3] = Kp * p_r
        tau_damp[:, 4] = Mq * q_r
        # 非线性 yaw 阻尼 (原版: Nr * (1 + 10*|r|) * r)
        tau_damp[:, 5] = Nr * (1.0 + 10.0 * torch.abs(r_r)) * r_r

        # 横流阻力
        tau_crossflow = cross_flow_drag_batch(self.L, self.B_pont, T, nu_r)  # 使用 B_pont 与原版一致

        # ========== 推力 ==========
        n_port = n[:, 0]
        n_starboard = n[:, 1]

        # 推力计算（考虑正反向不同系数）
        T_port = torch.where(
            n_port >= 0,
            self.k_pos * n_port * torch.abs(n_port),
            self.k_neg * n_port * torch.abs(n_port)
        )
        T_starboard = torch.where(
            n_starboard >= 0,
            self.k_pos * n_starboard * torch.abs(n_starboard),
            self.k_neg * n_starboard * torch.abs(n_starboard)
        )

        # 推力向量
        # 原版: tau[5] = -l1*T_port - l2*T_starboard, l1=-y_pont, l2=y_pont
        # = y_pont * (T_port - T_starboard)
        tau_thrust = torch.zeros(B, 6, device=self.device, dtype=DTYPE)
        tau_thrust[:, 0] = T_port + T_starboard  # 纵向力
        tau_thrust[:, 5] = (T_port - T_starboard) * self.y_pont  # 转艏力矩 (修复符号)

        # ========== 风力 ==========
        tau_wind = wind_force_batch(
            self.L, self.B, self.H_freeboard,
            nu, psi, V_w, beta_w
        )

        # ========== 总力 ==========
        tau_total = tau_thrust + tau_crossflow + tau_wind

        # ========== 静水力（恢复力）==========
        # 原版: force = tau + tau_damp + tau_crossflow + tau_wind - C @ nu_r - G @ eta_shifted
        # eta_shifted = eta - eta_0, 无负载时 eta_0 = 0, 所以 eta_shifted = eta
        # G 只影响 z, phi, theta (索引 2, 3, 4)
        G_expanded = self.G.unsqueeze(0).expand(B, -1, -1)
        G_eta = torch.bmm(G_expanded, eta.unsqueeze(-1)).squeeze(-1)

        # ========== 动力学方程 ==========
        # M @ nu_dot + C @ nu_r + G @ eta = tau + tau_damp
        # nu_dot = M^{-1} @ (tau + tau_damp - C @ nu_r - G @ eta)

        C_nu = torch.bmm(C_total, nu_r.unsqueeze(-1)).squeeze(-1)  # 使用 C = CRB + CA

        rhs = tau_total + tau_damp - C_nu - G_eta  # 添加静水力项

        # 求解 nu_dot（使用完整矩阵求解，与原版一致）
        # M @ nu_dot = rhs => nu_dot = M^{-1} @ rhs
        # 原版: nu_dot = nu_c_dot + np.linalg.solve(M, force)
        nu_dot_body = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)
        nu_dot = nu_c_dot + nu_dot_body  # 添加海流加速度项

        # ========== 运动学 ==========
        J = eulerang_batch(eta[:, 3], eta[:, 4], eta[:, 5])
        eta_dot = torch.bmm(J, nu.unsqueeze(-1)).squeeze(-1)

        # 组装状态导数
        x_dot = torch.cat([nu_dot, eta_dot], dim=1)

        return x_dot

    def step_batch(
        self,
        x: torch.Tensor,  # (B, 12)
        n: torch.Tensor,  # (B, 2)
        dt: float,
        V_c: torch.Tensor,
        beta_c: torch.Tensor,
        V_w: torch.Tensor,
        beta_w: torch.Tensor
    ) -> torch.Tensor:
        """
        批量 RK4 积分
        """
        # RK4 积分
        k1 = self.dynamics_batch(x, n, V_c, beta_c, V_w, beta_w)
        k2 = self.dynamics_batch(x + 0.5 * dt * k1, n, V_c, beta_c, V_w, beta_w)
        k3 = self.dynamics_batch(x + 0.5 * dt * k2, n, V_c, beta_c, V_w, beta_w)
        k4 = self.dynamics_batch(x + dt * k3, n, V_c, beta_c, V_w, beta_w)

        x_new = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # 归一化航向角到 [-pi, pi]
        x_new[:, 11] = torch.remainder(x_new[:, 11] + math.pi, 2*math.pi) - math.pi

        return x_new


class OtterTorchUSV:
    """
    Otter USV 向量化接口

    提供与原版 OtterUSV_DualPropeller 兼容的接口，
    但支持批量操作
    """

    def __init__(self, batch_size: int = 1, device: torch.device = DEVICE):
        self.batch_size = batch_size
        self.device = device
        self.core = OtterTorchCore(batch_size, device)

        # 船舶参数
        self.L = self.core.L
        self.B = self.core.B
        self.n_max = self.core.n_max
        self.n_min = self.core.n_min

        # 状态: (B, 12)
        self.full_state = torch.zeros(batch_size, 12, device=device, dtype=DTYPE)

    def reset(self, initial_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        重置环境

        Args:
            initial_states: (B, 6) [u, v, r, x, y, psi] 或 None

        Returns:
            (B, 6) 简化状态
        """
        if initial_states is None:
            self.full_state = torch.zeros(self.batch_size, 12, device=self.device, dtype=DTYPE)
            self.full_state[:, 0] = 0.1  # 初始纵向速度
        else:
            self.full_state = torch.zeros(self.batch_size, 12, device=self.device, dtype=DTYPE)
            self.full_state[:, 0] = initial_states[:, 0]  # u
            self.full_state[:, 1] = initial_states[:, 1]  # v
            self.full_state[:, 5] = initial_states[:, 2]  # r
            self.full_state[:, 6] = initial_states[:, 3]  # x
            self.full_state[:, 7] = initial_states[:, 4]  # y
            self.full_state[:, 11] = initial_states[:, 5]  # psi

        return self.get_state()

    def step(
        self,
        n: torch.Tensor,  # (B, 2) [n_port, n_starboard]
        dt: float = 0.1,
        V_c: Optional[torch.Tensor] = None,
        beta_c: Optional[torch.Tensor] = None,
        V_w: Optional[torch.Tensor] = None,
        beta_w: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        批量执行一步仿真

        Returns:
            (B, 6) 简化状态 [u, v, r, x, y, psi]
        """
        B = self.batch_size

        # 默认值
        if V_c is None:
            V_c = torch.zeros(B, device=self.device, dtype=DTYPE)
        if beta_c is None:
            beta_c = torch.zeros(B, device=self.device, dtype=DTYPE)
        if V_w is None:
            V_w = torch.zeros(B, device=self.device, dtype=DTYPE)
        if beta_w is None:
            beta_w = torch.zeros(B, device=self.device, dtype=DTYPE)

        # 限制转速范围
        n = torch.clamp(n, self.n_min, self.n_max)

        # RK4 积分
        self.full_state = self.core.step_batch(
            self.full_state, n, dt, V_c, beta_c, V_w, beta_w
        )

        return self.get_state()

    def get_state(self) -> torch.Tensor:
        """
        获取简化状态

        Returns:
            (B, 6) [u, v, r, x, y, psi]
        """
        state = torch.zeros(self.batch_size, 6, device=self.device, dtype=DTYPE)
        state[:, 0] = self.full_state[:, 0]  # u
        state[:, 1] = self.full_state[:, 1]  # v
        state[:, 2] = self.full_state[:, 5]  # r
        state[:, 3] = self.full_state[:, 6]  # x
        state[:, 4] = self.full_state[:, 7]  # y
        state[:, 5] = self.full_state[:, 11]  # psi
        return state


# ==================== 测试代码 ====================

def test_vectorized_physics():
    """测试向量化物理仿真"""
    print("=" * 60)
    print("测试 OtterTorchUSV 向量化物理仿真")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    batch_size = 64
    print(f"批量大小: {batch_size}")

    # 创建向量化 USV
    usv = OtterTorchUSV(batch_size=batch_size, device=device)

    # 初始状态
    initial = torch.zeros(batch_size, 6, device=device, dtype=DTYPE)
    initial[:, 0] = 1.5  # u = 1.5 m/s
    initial[:, 5] = torch.linspace(0, math.pi/4, batch_size, device=device)  # 不同航向

    usv.reset(initial)

    # 干扰参数
    V_c = torch.ones(batch_size, device=device) * 0.1
    beta_c = torch.zeros(batch_size, device=device)
    V_w = torch.ones(batch_size, device=device) * 2.0
    beta_w = torch.ones(batch_size, device=device) * math.pi / 4

    # 控制输入（满油门前进）
    n = torch.ones(batch_size, 2, device=device) * usv.n_max * 0.7

    # 预热
    print("\n预热...")
    for _ in range(10):
        usv.step(n, dt=0.1, V_c=V_c, beta_c=beta_c, V_w=V_w, beta_w=beta_w)

    # 性能测试
    import time

    usv.reset(initial)

    n_steps = 1000

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(n_steps):
        usv.step(n, dt=0.1, V_c=V_c, beta_c=beta_c, V_w=V_w, beta_w=beta_w)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_env_steps = batch_size * n_steps
    steps_per_sec = total_env_steps / elapsed

    print(f"\n性能测试结果:")
    print(f"  总步数: {total_env_steps:,}")
    print(f"  耗时: {elapsed:.3f} 秒")
    print(f"  吞吐量: {steps_per_sec:,.0f} steps/sec")

    # 对比原版（单环境）估算
    # 假设原版 ~500 steps/sec
    original_estimate = 500
    speedup = steps_per_sec / original_estimate
    print(f"  相比单环境估计加速: {speedup:.1f}x")

    # 检查状态
    state = usv.get_state()
    print(f"\n最终状态 (batch 0):")
    print(f"  u = {state[0, 0]:.3f} m/s")
    print(f"  v = {state[0, 1]:.3f} m/s")
    print(f"  r = {state[0, 2]:.3f} rad/s")
    print(f"  x = {state[0, 3]:.1f} m")
    print(f"  y = {state[0, 4]:.1f} m")
    print(f"  psi = {state[0, 5]:.3f} rad")


if __name__ == "__main__":
    test_vectorized_physics()
