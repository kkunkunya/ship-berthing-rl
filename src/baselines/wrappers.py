from src.utils.disturbance_metrics import override_curriculum_disturbance_config


class GymnasiumCompatWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = getattr(env, "action_space", None)
        self.observation_space = getattr(env, "observation_space", None)
        self.metadata = getattr(env, "metadata", {})

    def reset(self, *, seed=None, options=None):
        if seed is not None and hasattr(self.env, "seed"):
            self.env.seed(seed)
        obs = self.env.reset()
        if isinstance(obs, tuple) and len(obs) == 2:
            return obs
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = bool(done)
        truncated = False
        if done and hasattr(self.env, "current_step") and hasattr(self.env, "max_episode_steps"):
            truncated = self.env.current_step >= self.env.max_episode_steps
            if truncated:
                terminated = False
        return obs, float(reward), terminated, truncated, info

    def render(self, *args, **kwargs):
        if hasattr(self.env, "render"):
            return self.env.render(*args, **kwargs)
        return None

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None


def prepare_baseline_env(env, level, V_c_max, V_w_max):
    override_curriculum_disturbance_config(env, level, V_c_max, V_w_max)
    return env
