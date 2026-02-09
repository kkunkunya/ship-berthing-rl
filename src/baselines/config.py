from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineConfig:
    algorithm: str
    train_timesteps: int
    eval_episodes: int
    seed: int
    learning_rate: float
    gamma: float
    batch_size: int
    n_steps: int | None = None
    n_epochs: int | None = None
    clip_range: float | None = None
    tau: float | None = None
    policy_delay: int | None = None
    target_policy_noise: float | None = None
    target_noise_clip: float | None = None
    buffer_size: int | None = None


def get_baseline_config(algorithm: str) -> BaselineConfig:
    algo = algorithm.lower()
    if algo == "ppo":
        return BaselineConfig(
            algorithm="ppo",
            train_timesteps=150_000,
            eval_episodes=100,
            seed=42,
            learning_rate=3e-4,
            gamma=0.99,
            batch_size=64,
            n_steps=2048,
            n_epochs=10,
            clip_range=0.2,
        )
    if algo == "td3":
        return BaselineConfig(
            algorithm="td3",
            train_timesteps=150_000,
            eval_episodes=100,
            seed=42,
            learning_rate=1e-3,
            gamma=0.99,
            batch_size=256,
            tau=0.005,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            buffer_size=100_000,
        )
    raise ValueError(f"Unsupported baseline algorithm: {algorithm}")
