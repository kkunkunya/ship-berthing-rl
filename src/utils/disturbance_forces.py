import numpy as np

from src.physics.otter import (
    OtterUSVCore,
    Smtrx,
    Hmtrx,
    Rzyx,
    addedMassSurge,
    m2c,
    crossFlowDrag,
    windForce,
)


def _compute_dynamics_terms(core, x, n, V_c, beta_c, V_w, beta_w):
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    if x.shape[0] != 12:
        raise ValueError("state x must be length 12")
    if n.shape[0] != 2:
        raise ValueError("control n must be length 2")

    # 数值保护：保持与动力学一致
    x = np.clip(x, -1e6, 1e6)
    nu = x[0:6].copy()
    nu2 = x[3:6]
    eta = x[6:12].copy()

    nu[0] = np.clip(nu[0], -10, 10)
    nu[1] = np.clip(nu[1], -5, 5)
    nu[2] = np.clip(nu[2], -5, 5)
    nu[3] = np.clip(nu[3], -2, 2)
    nu[4] = np.clip(nu[4], -2, 2)
    nu[5] = np.clip(nu[5], -2, 2)
    eta[3] = np.clip(eta[3], -np.pi / 3, np.pi / 3)
    eta[4] = np.clip(eta[4], -np.pi / 3, np.pi / 3)

    u_c = V_c * np.cos(beta_c - eta[5])
    v_c = V_c * np.sin(beta_c - eta[5])
    nu_c = np.array([u_c, v_c, 0, 0, 0, 0])
    nu_r = nu - nu_c
    nu_c_dot = np.concatenate([
        -Smtrx(nu2) @ nu_c[0:3],
        np.zeros(3),
    ])

    # 质量和惯性
    mp = 0.0
    rp = np.zeros(3)
    nabla = (core.m + mp) / core.rho
    T = nabla / (2 * core.Cb_pont * core.B_pont * core.L)
    Ig_CG = core.m * np.diag([core.R44**2, core.R55**2, core.R66**2])
    rg = (core.m * core.rg + mp * rp) / (core.m + mp)
    Ig = Ig_CG - core.m * Smtrx(rg) @ Smtrx(rg) - mp * Smtrx(rp) @ Smtrx(rp)

    I3 = np.eye(3)
    O3 = np.zeros((3, 3))
    MRB_CG = np.block([[(core.m + mp) * I3, O3], [O3, Ig]])
    CRB_CG = np.block([
        [(core.m + mp) * Smtrx(nu2), O3],
        [O3, -Smtrx(Ig @ nu2)],
    ])

    H = Hmtrx(rg)
    MRB = H.T @ MRB_CG @ H
    CRB = H.T @ CRB_CG @ H

    Xudot = -addedMassSurge(core.m, core.L, core.rho)
    Yvdot = -1.5 * core.m
    Zwdot = -1.0 * core.m
    Kpdot = -0.2 * Ig[0, 0]
    Mqdot = -0.8 * Ig[1, 1]
    Nrdot = -1.7 * Ig[2, 2]

    MA = -np.diag([Xudot, Yvdot, Zwdot, Kpdot, Mqdot, Nrdot])
    CA = m2c(MA, nu_r)
    M = MRB + MA
    C = CRB + CA

    # 静水力
    Aw_pont = core.Cw_pont * core.L * core.B_pont
    I_T = (
        2
        * (1 / 12)
        * core.L
        * core.B_pont**3
        * (6 * core.Cw_pont**3 / ((1 + core.Cw_pont) * (1 + 2 * core.Cw_pont)))
        + 2 * Aw_pont * core.y_pont**2
    )
    I_L = 0.8 * 2 * (1 / 12) * core.B_pont * core.L**3
    KB = (1 / 3) * (5 * T / 2 - 0.5 * nabla / (core.L * core.B_pont))
    BM_T = I_T / nabla
    BM_L = I_L / nabla
    KM_T = KB + BM_T
    KM_L = KB + BM_L
    KG = T - rg[2]
    GM_T = KM_T - KG
    GM_L = KM_L - KG

    G33 = core.rho * core.g * (2 * Aw_pont)
    G44 = core.rho * core.g * nabla * GM_T
    G55 = core.rho * core.g * nabla * GM_L

    G_CF = np.diag([0, 0, G33, G44, G55, 0])
    LCF = -0.2
    H_CF = Hmtrx(np.array([LCF, 0, 0]))
    G = H_CF.T @ G_CF @ H_CF

    # 阻尼力
    Xu = -24.4 * core.g / core.Umax
    Yv = -M[1, 1] / core.T_sway
    Zw = -2 * 0.3 * np.sqrt(G33 / M[2, 2]) * M[2, 2]
    Kp = -2 * 0.2 * np.sqrt(G44 / M[3, 3]) * M[3, 3]
    Mq = -2 * 0.4 * np.sqrt(G55 / M[4, 4]) * M[4, 4]
    Nr = -M[5, 5] / core.T_yaw

    # 推力计算
    Thrust = np.zeros(2)
    for i in range(2):
        n_sat = np.clip(n[i], core.n_min, core.n_max)
        if n_sat > 0:
            Thrust[i] = core.k_pos * n_sat * abs(n_sat)
        else:
            Thrust[i] = core.k_neg * n_sat * abs(n_sat)

    tau = np.array([
        Thrust[0] + Thrust[1],
        0,
        0,
        0,
        0,
        -core.l1 * Thrust[0] - core.l2 * Thrust[1],
    ])

    Xh = Xu * np.clip(nu_r[0], -10, 10)
    Yh = Yv * np.clip(nu_r[1], -5, 5)
    Zh = Zw * np.clip(nu_r[2], -5, 5)
    Kh = Kp * np.clip(nu_r[3], -2, 2)
    Mh = Mq * np.clip(nu_r[4], -2, 2)
    r_clip = np.clip(nu_r[5], -2, 2)
    Nh = Nr * (1 + 10 * abs(r_clip)) * r_clip
    tau_damp = np.array([Xh, Yh, Zh, Kh, Mh, Nh])

    tau_crossflow = crossFlowDrag(core.L, core.B_pont, T, nu_r)

    H_freeboard = 0.3
    tau_wind = windForce(core.L, core.B, H_freeboard, nu, eta[5], V_w, beta_w)

    f_payload = Rzyx(eta[3], eta[4], eta[5]).T @ np.array([0, 0, mp * core.g])
    m_payload = Smtrx(rp) @ f_payload
    g_0 = np.concatenate([f_payload, m_payload])

    eta_0 = np.zeros(6)
    eta_0[2:5] = np.linalg.inv(G[2:5, 2:5]) @ g_0[2:5]
    eta_shifted = eta - eta_0

    force = tau + tau_damp + tau_crossflow + tau_wind - C @ nu_r - G @ eta_shifted

    return {
        "M": M,
        "force": force,
        "nu_c_dot": nu_c_dot,
        "tau_wind": tau_wind,
    }


def compute_equivalent_current_tau(core, x, n, V_c, beta_c, V_w, beta_w):
    """
    计算等效海流扰动力（广义力）与风力干扰。

    说明：海流通过相对速度进入动力学，因此用“等效广义力”形式表达。
    """
    terms = _compute_dynamics_terms(core, x, n, V_c, beta_c, V_w, beta_w)
    terms_no_current = _compute_dynamics_terms(core, x, n, 0.0, 0.0, V_w, beta_w)

    tau_current = terms["M"] @ terms["nu_c_dot"] + terms["force"] - terms_no_current["force"]
    tau_wind = terms["tau_wind"]
    tau_combined = tau_current + tau_wind

    return tau_current, tau_wind, tau_combined


def compute_combined_disturbance_tau(core, x, n, V_c, beta_c, V_w, beta_w):
    tau_current, tau_wind, tau_combined = compute_equivalent_current_tau(
        core=core,
        x=x,
        n=n,
        V_c=V_c,
        beta_c=beta_c,
        V_w=V_w,
        beta_w=beta_w,
    )
    return {
        "tau_current": tau_current,
        "tau_wind": tau_wind,
        "tau_combined": tau_combined,
    }
