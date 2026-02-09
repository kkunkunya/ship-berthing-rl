import numpy as np


def compute_body_frame_components(speed, beta, psi):
    speed_arr = np.asarray(speed, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)
    psi_arr = np.asarray(psi, dtype=float)
    u = speed_arr * np.cos(beta_arr - psi_arr)
    v = speed_arr * np.sin(beta_arr - psi_arr)
    return u, v


def _expand_to_series(value, length):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(length, arr, dtype=float)
    return arr


def compute_disturbance_series(psi_series, V_c, beta_c, V_w, beta_w):
    psi = np.asarray(psi_series, dtype=float)
    if psi.ndim != 1:
        raise ValueError("psi_series must be a 1D array")

    V_c_series = _expand_to_series(V_c, psi.shape[0])
    beta_c_series = _expand_to_series(beta_c, psi.shape[0])
    V_w_series = _expand_to_series(V_w, psi.shape[0])
    beta_w_series = _expand_to_series(beta_w, psi.shape[0])

    u_c, v_c = compute_body_frame_components(V_c_series, beta_c_series, psi)
    u_w, v_w = compute_body_frame_components(V_w_series, beta_w_series, psi)

    return {
        "u_c": u_c,
        "v_c": v_c,
        "u_w": u_w,
        "v_w": v_w,
    }


def override_curriculum_disturbance_config(env, level, V_c_max, V_w_max):
    if not hasattr(env, "curriculum_current_config") or not hasattr(env, "curriculum_wind_config"):
        raise AttributeError("env lacks curriculum disturbance config")
    if level not in env.curriculum_current_config:
        env.curriculum_current_config[level] = {}
    if level not in env.curriculum_wind_config:
        env.curriculum_wind_config[level] = {}
    env.curriculum_current_config[level]["V_c_max"] = float(V_c_max)
    env.curriculum_wind_config[level]["V_w_max"] = float(V_w_max)
