"""
向量化 PER-SAC Otter USV 训练 - GPU 加速版本

利用 VectorizedOtterEnv 在 GPU 上并行运行 N 个环境，
显著提升训练速度（预计 5-10x 加速）。

核心改进：
1. 并行环境交互 - N 个环境同时 step
2. 批量经验收集 - 每步收集 N 条 transition
3. 异步重置 - 单个环境 done 后立即重置
4. 保持原有功能 - PER, 成功轨迹回放, 课程学习
5. 修复：动态 UTD (Update-To-Data) 更新频率
6. 修复：自适应熵调整 (Adaptive Entropy)
"""

import argparse
import os
import numpy as np
import time
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque

# 导入向量化环境
from ..envs.vectorized_env import VectorizedOtterEnv
from ..utils.per_buffer import PrioritizedReplayBuffer, GpuPrioritizedReplayBuffer, SuccessTrajectoryBuffer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============== 网络架构（与 train.py 相同）==============
class ImprovedActorNetwork(nn.Module):
    """改进的Actor网络"""

    def __init__(self, input_dim, action_dim, fc1_dim=256, fc2_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, fc2_dim)
        self.ln3 = nn.LayerNorm(fc2_dim)
        self.mean = nn.Linear(fc2_dim, action_dim)
        self.log_std = nn.Linear(fc2_dim, action_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.uniform_(self.mean.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std.weight, -3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def get_action(self, state, evaluate=False):
        """批量获取动作 (保持在 GPU 上)"""
        if evaluate:
            mean, _ = self.forward(state)
            return torch.tanh(mean)
        else:
            action, _ = self.sample(state)
            return action


class ImprovedCriticNetwork(nn.Module):
    """改进的Critic网络"""

    def __init__(self, input_dim, action_dim, fc1_dim=256, fc2_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, fc2_dim)
        self.ln3 = nn.LayerNorm(fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.uniform_(self.q.weight, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.q(x)


# ============== 课程学习管理器 ==============
class CurriculumManager:
    """课程学习管理器 - 根据训练成功率自动调整难度"""

    def __init__(
        self,
        level_thresholds=None,
        window_size=100,
        patience=200,
        initial_level=1,
    ):
        self.level_thresholds = level_thresholds or [0.5, 0.6, 0.7, 0.8]
        self.window_size = window_size
        self.patience = patience
        self.current_level = initial_level
        self.max_level = len(self.level_thresholds) + 1
        self.success_history = deque(maxlen=window_size)
        self.level_stable_episodes = 0

    def update(self, success):
        """更新课程状态，返回新等级（如果升级）"""
        self.success_history.append(1 if success else 0)
        self.level_stable_episodes += 1

        if len(self.success_history) < self.window_size:
            return None

        current_success_rate = float(np.mean(self.success_history))

        if self.current_level < self.max_level:
            threshold = self.level_thresholds[self.current_level - 1]
            if current_success_rate >= threshold and self.level_stable_episodes >= self.patience:
                self.current_level += 1
                self.level_stable_episodes = 0
                self.success_history.clear()
                logger.info(
                    f"[Curriculum] 升级到等级 {self.current_level}! 成功率: {current_success_rate:.2%}"
                )
                return self.current_level
        return None

    def get_current_level(self):
        return self.current_level

    def get_success_rate(self):
        return float(np.mean(self.success_history)) if len(self.success_history) > 0 else 0.0


# ============== 干扰强度爬坡器 ==============
class DisturbanceRamp:
    """按成功率窗口逐步提升干扰上限"""

    def __init__(
        self,
        vc_steps,
        vw_steps,
        window_size=50,
        success_threshold=0.6,
    ):
        if not vc_steps or not vw_steps:
            raise ValueError("vc_steps/vw_steps 不能为空")

        if len(vc_steps) != len(vw_steps):
            # 对齐长度，短的用最后一个值补齐
            max_len = max(len(vc_steps), len(vw_steps))
            vc_steps = (vc_steps + [vc_steps[-1]] * max_len)[:max_len]
            vw_steps = (vw_steps + [vw_steps[-1]] * max_len)[:max_len]

        self.vc_steps = vc_steps
        self.vw_steps = vw_steps
        self.window_size = window_size
        self.success_threshold = success_threshold
        self.step_index = 0
        self.success_history = deque(maxlen=window_size)

    def current(self):
        return self.vc_steps[self.step_index], self.vw_steps[self.step_index]

    def update(self, success):
        self.success_history.append(1 if success else 0)
        if len(self.success_history) < self.window_size:
            return None

        success_rate = float(np.mean(self.success_history))
        if success_rate >= self.success_threshold and self.step_index < len(self.vc_steps) - 1:
            self.step_index += 1
            self.success_history.clear()
            return self.current()
        return None


# ============== 自适应熵调整器 (移植自 train.py) ==============
class AdaptiveEntropyManager:
    """自适应熵系数管理器 - 根据训练状态动态调整target_entropy"""

    def __init__(self, action_dim, initial_target_entropy=None,
                 min_entropy_ratio=0.5, max_entropy_ratio=1.5, adjustment_rate=0.01):
        self.action_dim = action_dim
        self.base_target_entropy = initial_target_entropy if initial_target_entropy else -action_dim
        self.current_target_entropy = self.base_target_entropy
        self.min_entropy = self.base_target_entropy * max_entropy_ratio
        self.max_entropy = self.base_target_entropy * min_entropy_ratio
        self.adjustment_rate = adjustment_rate
        self.success_rate_history = deque(maxlen=50)

    def update(self, success_rate):
        """根据成功率更新目标熵"""
        self.success_rate_history.append(success_rate)

        if len(self.success_rate_history) < 10:
            return self.current_target_entropy

        recent_rate = np.mean(list(self.success_rate_history)[-10:])
        older_rate = np.mean(list(self.success_rate_history)[:10]) if len(
            self.success_rate_history) >= 20 else recent_rate

        if recent_rate < older_rate - 0.1:
            adjustment = self.adjustment_rate * (older_rate - recent_rate) * 10
            self.current_target_entropy = max(self.min_entropy, self.current_target_entropy - adjustment)
            logger.info(f"[Entropy] 成功率下降，增加探索: target_entropy={self.current_target_entropy:.3f}")
        elif recent_rate > older_rate + 0.1:
            adjustment = self.adjustment_rate * (recent_rate - older_rate) * 5
            self.current_target_entropy = min(self.max_entropy, self.current_target_entropy + adjustment)
            logger.info(f"[Entropy] 成功率上升，减少探索: target_entropy={self.current_target_entropy:.3f}")

        return self.current_target_entropy

    def get_target_entropy(self):
        return self.current_target_entropy


# ============== 向量化 SAC Agent ==============
class VectorizedSacAgent:
    """
    向量化 SAC Agent

    与原版 EnhancedPerSacAgent 兼容，但优化了批量操作。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_envs: int = 64,
        alpha: float = 0.2,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 1e-5,
        critic_lr: float = 1e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        fc1_dim: int = 256,
        fc2_dim: int = 256,
        automatic_entropy_tuning: bool = True,
        debug_timing: bool = False,
        pin_memory: bool = False,
        use_gpu_replay: bool = False,
        checkpoint_dir: str = './checkpoints_vec/',
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.debug_timing = debug_timing
        self.pin_memory = pin_memory
        self.use_gpu_replay = use_gpu_replay
        self.last_update_timing = {}

        # 网络
        self.actor = ImprovedActorNetwork(state_dim, action_dim, fc1_dim, fc2_dim).to(DEVICE)
        self.critic1 = ImprovedCriticNetwork(state_dim, action_dim, fc1_dim, fc2_dim).to(DEVICE)
        self.critic2 = ImprovedCriticNetwork(state_dim, action_dim, fc1_dim, fc2_dim).to(DEVICE)
        self.target_critic1 = ImprovedCriticNetwork(state_dim, action_dim, fc1_dim, fc2_dim).to(DEVICE)
        self.target_critic2 = ImprovedCriticNetwork(state_dim, action_dim, fc1_dim, fc2_dim).to(DEVICE)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # 自适应熵
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if automatic_entropy_tuning:
            self.entropy_manager = AdaptiveEntropyManager(action_dim)
            self.target_entropy = self.entropy_manager.get_target_entropy()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
            self.alpha = self.log_alpha.exp().item()
            # 修复：防止 Adaptive Entropy 在并行环境下重复触发
            self.last_entropy_update_episode = -1
        else:
            self.alpha = alpha

        # PER 缓冲区
        if use_gpu_replay:
            self.memory = GpuPrioritizedReplayBuffer(
                buffer_size, state_dim, action_dim, batch_size,
                alpha=0.6, success_priority_boost=2.0, success_buffer_ratio=0.1,
                device=DEVICE
            )
        else:
            self.memory = PrioritizedReplayBuffer(
                buffer_size, state_dim, action_dim, batch_size,
                alpha=0.6, success_priority_boost=2.0, success_buffer_ratio=0.1
            )

        # 成功轨迹缓冲区
        self.success_buffer = SuccessTrajectoryBuffer(max_trajectories=100)

        # 每个环境的当前轨迹
        self.current_trajectories = [[] for _ in range(num_envs)]
        self.episode_rewards = [0.0 for _ in range(num_envs)]

        # 统计
        self.training_stats = {
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'alpha': deque(maxlen=100),
            'q_values': deque(maxlen=100),
            'target_entropy': deque(maxlen=100),
        }

        os.makedirs(checkpoint_dir, exist_ok=True)

    def select_action_batch(self, states: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        """
        批量选择动作（全部在 GPU 上）

        Args:
            states: (B, state_dim) tensor on GPU
            evaluate: 是否评估模式

        Returns:
            actions: (B, action_dim) tensor on GPU
        """
        with torch.no_grad():
            return self.actor.get_action(states, evaluate)

    def store_transitions_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ):
        """
        批量存储 transitions
        """
        if self.use_gpu_replay:
            if not torch.is_tensor(states):
                states_t = torch.as_tensor(states, device=DEVICE, dtype=torch.float32)
                actions_t = torch.as_tensor(actions, device=DEVICE, dtype=torch.float32)
                rewards_t = torch.as_tensor(rewards, device=DEVICE, dtype=torch.float32)
                next_states_t = torch.as_tensor(next_states, device=DEVICE, dtype=torch.float32)
                dones_t = torch.as_tensor(dones, device=DEVICE, dtype=torch.bool)
            else:
                states_t = states.detach()
                actions_t = actions.detach()
                rewards_t = rewards.detach()
                next_states_t = next_states.detach()
                dones_t = dones.detach()

            # 存入 PER 缓冲区（奖励缩放 /10）
            self.memory.store_transitions_batch(
                states_t, actions_t, rewards_t / 10.0, next_states_t, dones_t, is_success=False
            )

            # 记录到当前轨迹（尽量保持在 GPU）
            for i in range(states_t.shape[0]):
                self.current_trajectories[i].append((
                    states_t[i].detach(), actions_t[i].detach(), rewards_t[i].detach(),
                    next_states_t[i].detach(), dones_t[i].detach()
                ))
                self.episode_rewards[i] += float(rewards_t[i].item())
        else:
            for i in range(len(states)):
                # 存入 PER 缓冲区（奖励缩放 /10）
                self.memory.store_transition(
                    states[i], actions[i], rewards[i] / 10.0,
                    next_states[i], dones[i], is_success=False
                )

                # 记录到当前轨迹
                self.current_trajectories[i].append((
                    states[i].copy(), actions[i].copy(), rewards[i],
                    next_states[i].copy(), dones[i]
                ))
                self.episode_rewards[i] += rewards[i]

    def handle_episode_ends(self, dones: np.ndarray, infos: dict) -> list:
        """
        处理 episode 结束
        """
        completed = []
        successes = infos.get('success', np.zeros(self.num_envs, dtype=bool))

        for i in range(self.num_envs):
            if dones[i]:
                success = successes[i] if isinstance(successes, np.ndarray) else False
                reward = self.episode_rewards[i]

                # 存储成功轨迹
                if success and len(self.current_trajectories[i]) > 0:
                    self.success_buffer.store_trajectory(
                        self.current_trajectories[i].copy(), reward
                    )

                completed.append((i, reward, success))

                # 重置该环境的轨迹
                self.current_trajectories[i] = []
                self.episode_rewards[i] = 0.0

        return completed

    def replay_success_trajectories(self, n_trajectories: int = 1):
        """回放成功轨迹"""
        trajectories = self.success_buffer.sample_trajectory(
            n=n_trajectories, prioritize_by_reward=True
        )
        for traj in trajectories:
            for state, action, reward, next_state, done in traj:
                self.memory.store_transition(
                    state, action, reward / 10.0, next_state, done, is_success=True
                )
        if len(trajectories) > 0:
            logger.debug(f"[Replay] 回放了 {len(trajectories)} 条成功轨迹")

    def update_target_entropy(self, success_rate):
        """更新目标熵"""
        if self.automatic_entropy_tuning:
            self.target_entropy = self.entropy_manager.update(success_rate)

    def update(self, beta: float = 0.4, success_sample_ratio: float = 0.2):
        """SAC 更新"""
        if not self.memory.ready():
            return

        timing = {} if self.debug_timing else None

        def sync_cuda():
            if self.debug_timing and torch.cuda.is_available():
                torch.cuda.synchronize()

        sample_start = time.time()
        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample_buffer(beta, success_sample_ratio)
        if timing is not None:
            timing["sample"] = time.time() - sample_start

        sync_cuda()
        transfer_start = time.time()
        if self.use_gpu_replay:
            states = states.to(DEVICE).float()
            actions = actions.to(DEVICE).float()
            rewards = rewards.to(DEVICE).float().unsqueeze(1)
            next_states = next_states.to(DEVICE).float()
            dones = dones.to(DEVICE).float().unsqueeze(1)
            weights = weights.to(DEVICE).float().unsqueeze(1)
        else:
            def to_tensor(array):
                tensor = torch.from_numpy(array)
                if tensor.dtype != torch.float32:
                    tensor = tensor.float()
                if self.pin_memory and DEVICE.type == "cuda":
                    tensor = tensor.pin_memory()
                return tensor.to(DEVICE, non_blocking=self.pin_memory and DEVICE.type == "cuda")

            states = to_tensor(states)
            actions = to_tensor(actions)
            rewards = to_tensor(rewards).unsqueeze(1)
            next_states = to_tensor(next_states)
            dones = to_tensor(dones).unsqueeze(1)
            weights = to_tensor(weights).unsqueeze(1)
        sync_cuda()
        if timing is not None:
            timing["transfer"] = time.time() - transfer_start

        # Critic 更新
        sync_cuda()
        critic_start = time.time()
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        td_errors = ((q1 - target_q).abs() + (q2 - target_q).abs()) / 2
        priority_start = time.time()
        if self.use_gpu_replay:
            self.memory.update_priorities(indices, td_errors.detach().squeeze())
        else:
            td_errors_np = td_errors.detach().cpu().numpy().squeeze()
            self.memory.update_priorities(indices, td_errors_np)
        if timing is not None:
            timing["priority"] = time.time() - priority_start

        critic1_loss = (weights * F.mse_loss(q1, target_q, reduction='none')).mean()
        critic2_loss = (weights * F.mse_loss(q2, target_q, reduction='none')).mean()

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        sync_cuda()
        if timing is not None:
            timing["critic"] = time.time() - critic_start

        # Actor 更新
        sync_cuda()
        actor_start = time.time()
        actions_pred, log_probs = self.actor.sample(states)
        q1_pred = self.critic1(states, actions_pred)
        q2_pred = self.critic2(states, actions_pred)
        q_pred = torch.min(q1_pred, q2_pred)

        actor_loss = (self.alpha * log_probs - q_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        sync_cuda()
        if timing is not None:
            timing["actor"] = time.time() - actor_start

        # Alpha 更新
        if self.automatic_entropy_tuning:
            sync_cuda()
            alpha_start = time.time()
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            sync_cuda()
            if timing is not None:
                timing["alpha"] = time.time() - alpha_start

        sync_cuda()
        soft_start = time.time()
        self._soft_update()
        sync_cuda()
        if timing is not None:
            timing["soft_update"] = time.time() - soft_start

        # 记录统计
        self.training_stats['actor_loss'].append(actor_loss.item())
        self.training_stats['critic_loss'].append((critic1_loss.item() + critic2_loss.item()) / 2)
        self.training_stats['alpha'].append(self.alpha)
        self.training_stats['q_values'].append(q_pred.mean().item())
        if timing is not None:
            timing["total"] = sum(timing.values())
            self.last_update_timing = timing
        if self.automatic_entropy_tuning:
            self.training_stats['target_entropy'].append(self.target_entropy)

    def _soft_update(self):
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_models(self, episode):
        """保存检查点"""
        checkpoint = {
            'episode': episode,
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
        }
        if self.automatic_entropy_tuning:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['alpha_optimizer'] = self.alpha_optimizer.state_dict()

        path = os.path.join(self.checkpoint_dir, f'per_sac_otter_checkpoint_{episode}.pth')
        torch.save(checkpoint, path)
        logger.info(f"模型已保存: {path}")

    def load_models(self, episode, reset_optimizer: bool = False):
        """加载检查点"""
        if isinstance(episode, str) and episode.lower() == 'best':
            path = os.path.join(self.checkpoint_dir, 'per_sac_otter_checkpoint_best.pth')
        elif isinstance(episode, str) and os.path.isfile(episode):
            path = episode
        else:
            path = os.path.join(self.checkpoint_dir, f'per_sac_otter_checkpoint_{episode}.pth')
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        
        if not reset_optimizer:
            actor_opt_state = checkpoint.get('actor_optimizer')
            critic1_opt_state = checkpoint.get('critic1_optimizer')
            critic2_opt_state = checkpoint.get('critic2_optimizer')

            if actor_opt_state is not None:
                self.actor_optimizer.load_state_dict(actor_opt_state)
            if critic1_opt_state is not None:
                self.critic1_optimizer.load_state_dict(critic1_opt_state)
            if critic2_opt_state is not None:
                self.critic2_optimizer.load_state_dict(critic2_opt_state)

            if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
                self.log_alpha = checkpoint['log_alpha']
                alpha_opt_state = checkpoint.get('alpha_optimizer')
                if alpha_opt_state is not None:
                    self.alpha_optimizer.load_state_dict(alpha_opt_state)
                self.alpha = self.log_alpha.exp().item()
            logger.info(f"模型及优化器已加载: {path}")
        else:
            # 即使重置优化器，也保留 log_alpha，防止 entropy 激增摧毁策略
            if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
                self.log_alpha = checkpoint['log_alpha']
                self.alpha = self.log_alpha.exp().item()
                logger.info(f"保留 Log Alpha: {self.log_alpha.item():.4f} (Alpha={self.alpha:.4f})")
            
            logger.info(f"模型已加载 (优化器重置): {path}")

    def load_from_original(self, checkpoint_dir: str, episode, reset_optimizer: bool = False):
        """从原始 train.py 的检查点加载"""
        if isinstance(episode, str) and episode.lower() == 'best':
            path = os.path.join(checkpoint_dir, 'per_sac_otter_checkpoint_best.pth')
        elif isinstance(episode, str) and os.path.isfile(episode):
            path = episode
        else:
            path = os.path.join(checkpoint_dir, f'per_sac_otter_checkpoint_{episode}.pth')
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])

        logger.info(f"从原始检查点加载: {path}")
        if reset_optimizer:
            logger.info("优化器已重置 (不加载原有状态)")


# ============== 训练监控 ==============
class VectorizedTrainingMonitor:
    """向量化训练监控器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_rewards = []
        self.success_episodes = []
        self.episode_lengths = []
        self.steps_per_second = []

    def update_episode(self, reward: float, success: bool, length: int = 0):
        self.episode_rewards.append(reward)
        self.success_episodes.append(1.0 if success else 0.0)
        self.episode_lengths.append(length)

    def update_speed(self, sps: float):
        self.steps_per_second.append(sps)

    def get_stats(self) -> dict:
        stats = {}
        if len(self.episode_rewards) > 0:
            recent = self.episode_rewards[-self.window_size:]
            stats['avg_reward'] = np.mean(recent)
            stats['std_reward'] = np.std(recent)

        if len(self.success_episodes) > 0:
            recent = self.success_episodes[-self.window_size:]
            stats['success_rate'] = np.mean(recent)

        if len(self.steps_per_second) > 0:
            recent = self.steps_per_second[-self.window_size:]
            stats['avg_sps'] = np.mean(recent)

        return stats

    def save_data(self, filepath: str):
        # 转换为原生 Python 类型以支持 JSON 序列化
        def to_native(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [to_native(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            return obj

        data = {
            'episode_rewards': to_native(self.episode_rewards),
            'success_episodes': to_native(self.success_episodes),
            'episode_lengths': to_native(self.episode_lengths),
            'steps_per_second': to_native(self.steps_per_second),
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# ============== 评估函数 ==============
def evaluate_agent(
    agent: VectorizedSacAgent,
    env: VectorizedOtterEnv,
    num_episodes: int = 10,
    max_steps: int | None = None,
    stochastic: bool = False,
) -> tuple:
    """
    评估 agent 性能（使用独立评估环境，避免干扰训练环境）
    """
    rewards = []
    successes = []

    eval_env = VectorizedOtterEnv(
        num_envs=1,
        device=env.device,
        curriculum_level=env.curriculum_level,
        enable_current=env.enable_current,
        V_c_max=env.V_c_max,
        current_mode=env.current_mode,
        enable_wind=env.enable_wind,
        V_w_max=env.V_w_max,
        wind_mode=env.wind_mode,
    )

    max_steps = max_steps if max_steps is not None else getattr(eval_env, "max_episode_steps", 2000)
    
    # 调试统计
    action_means = []
    pos_error_means = []

    for ep_i in range(num_episodes):
        obs = eval_env.reset()
        episode_reward = 0.0
        done = False
        step_count = 0
        reached_max_steps = False
        
        ep_actions = []
        ep_pos_errors = []

        while not done:
            # 评估模式：默认 deterministic (evaluate=True)，除非指定 stochastic
            use_eval_mode = not stochastic
            action = agent.select_action_batch(obs, evaluate=use_eval_mode)
            
            # 记录调试信息
            ep_actions.append(action[0].cpu().numpy())
            # obs[0, 0] 是归一化的 pos_error_x, [0, 1] 是 pos_error_y
            # scaling factor 为 0.2 (5m -> 1.0)
            raw_pos_x = obs[0, 0].item() / 0.2
            raw_pos_y = obs[0, 1].item() / 0.2
            ep_pos_errors.append(np.sqrt(raw_pos_x**2 + raw_pos_y**2))

            obs, reward, done_full, info = eval_env.step(action)
            episode_reward += reward[0].item()
            done = bool(done_full[0].item())
            step_count += 1

            if not done and step_count >= max_steps:
                reached_max_steps = True
                logger.warning(f"评估 Ep{ep_i} 超过最大步数({max_steps})，判定为失败。PosErr={ep_pos_errors[-1]:.1f}m")
                break

        success = 0.0 if reached_max_steps else info.get('success', np.zeros(1))[0]
        rewards.append(episode_reward)
        successes.append(success)
        
        action_means.append(np.mean(np.abs(ep_actions)))
        pos_error_means.append(np.mean(ep_pos_errors))

    # 打印详细评估日志
    avg_act = np.mean(action_means)
    avg_err = np.mean(pos_error_means)
    logger.info(f"[详细评估] AvgActionMag={avg_act:.3f}, AvgPosError={avg_err:.1f}m, Mode={'Stochastic' if stochastic else 'Deterministic'}")

    return np.mean(rewards), np.mean(successes)


def should_run_eval(total_episodes: int, eval_interval: int, last_eval_episode: int) -> bool:
    if eval_interval <= 0:
        return False
    if total_episodes <= 0:
        return False
    if total_episodes == last_eval_episode:
        return False
    return total_episodes % eval_interval == 0


def compute_num_updates(num_envs: int, updates_per_step, utd_ratio: float) -> int:
    """计算每次环境步进的更新次数（UTD 比例）"""
    if updates_per_step is not None:
        return max(1, int(updates_per_step))
    if utd_ratio is None or utd_ratio <= 0:
        return 1
    return max(1, int(round(num_envs * utd_ratio)))


def should_log_heartbeat(last_ts: float, now_ts: float, interval_sec: float) -> bool:
    """按时间间隔判断是否输出心跳日志"""
    if interval_sec is None or interval_sec <= 0:
        return False
    return (now_ts - last_ts) >= interval_sec


def detect_slow_stages(stage_times: dict, slow_threshold_sec: float):
    """检测超时阶段，用于发出警告"""
    if slow_threshold_sec is None or slow_threshold_sec <= 0:
        return []
    slow = []
    for name, value in stage_times.items():
        if value >= slow_threshold_sec:
            slow.append(name)
    return slow


# ============== 主训练函数 ==============
def train(args):
    """向量化训练主函数"""

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ----------------------------------------------------
    # 修复：动态更新频率 (Update-To-Data Ratio)
    # ----------------------------------------------------
    updates_per_step = compute_num_updates(args.num_envs, args.updates_per_step, args.utd_ratio)
    if args.updates_per_step is None:
        # 如果未指定固定更新次数，则根据 utd_ratio 计算
        # 例如: num_envs=64, utd_ratio=1.0 -> updates=64
        # 这保证了 gradient_steps / sample = 1.0，与单环境训练保持一致
        if args.utd_ratio < 1.0:
            logger.warning(f"⚠️ UTD ratio ({args.utd_ratio}) is low! Recommended >= 1.0 for sample efficiency.")
        
        logger.info(f"自动设置 updates_per_step = {updates_per_step} (num_envs={args.num_envs} * utd_ratio={args.utd_ratio})")
    else:
        logger.info(f"使用手动 updates_per_step = {updates_per_step}")

    # 创建向量化环境
    env = VectorizedOtterEnv(
        num_envs=args.num_envs,
        device=DEVICE,
        curriculum_level=args.initial_curriculum_level,
        enable_current=args.enable_current,
        V_c_max=args.V_c_max,
        current_mode=args.current_mode,
        enable_wind=args.enable_wind,
        V_w_max=args.V_w_max,
        wind_mode=args.wind_mode,
        pos_scale=args.initial_pos_scale,  # 初始缩放
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    logger.info("=" * 70)
    logger.info("向量化 PER-SAC Otter USV 训练")
    logger.info("=" * 70)
    logger.info(f"并行环境数: {args.num_envs}")
    logger.info(f"设备: {DEVICE}")
    logger.info(f"状态维度: {state_dim}")
    logger.info(f"动作维度: {action_dim}")
    logger.info(f"课程等级: {args.initial_curriculum_level}")
    logger.info(f"目标 episodes: {args.episodes}")
    logger.info(f"观测缩放: {args.initial_pos_scale} -> {args.final_pos_scale} (over {args.pos_scale_steps} eps)")
    logger.info("=" * 70)
    logger.info("干扰配置:")
    logger.info(f"  海流: {'启用' if args.enable_current else '禁用'} (V_c_max={args.V_c_max}m/s)")
    logger.info(f"  风: {'启用' if args.enable_wind else '禁用'} (V_w_max={args.V_w_max}m/s)")
    logger.info("=" * 70)

    # 课程学习管理器
    curriculum = None
    if args.curriculum_auto:
        curriculum = CurriculumManager(
            window_size=args.curriculum_window,
            patience=args.curriculum_patience,
            initial_level=args.initial_curriculum_level,
        )

    # 干扰强度爬坡器
    ramp = None
    if args.ramp_disturbance:
        vc_steps = [float(x) for x in args.ramp_vc_steps.split(",") if x.strip()]
        vw_steps = [float(x) for x in args.ramp_vw_steps.split(",") if x.strip()]
        ramp = DisturbanceRamp(
            vc_steps=vc_steps,
            vw_steps=vw_steps,
            window_size=args.ramp_window,
            success_threshold=args.ramp_success_threshold,
        )
        env.V_c_max, env.V_w_max = ramp.current()
        logger.info(
            f"[Ramp] 初始干扰上限: V_c_max={env.V_c_max:.3f}, V_w_max={env.V_w_max:.3f}"
        )

    # 创建 Agent
    agent = VectorizedSacAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_envs=args.num_envs,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=0.99,
        tau=0.005,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        debug_timing=args.debug_timing,
        pin_memory=args.pin_memory,
        use_gpu_replay=args.use_gpu_replay,
        checkpoint_dir=args.checkpoint_dir,
    )

    # 加载检查点
    if args.load_checkpoint:
        if args.load_from_original:
            agent.load_from_original(args.load_from_original, args.load_checkpoint, reset_optimizer=args.reset_optimizer)
        else:
            agent.load_models(args.load_checkpoint, reset_optimizer=args.reset_optimizer)
        
        # 智能调整: 如果加载了 Checkpoint 且 warmup_steps 为默认值，则自动禁用 Warmup
        if args.warmup_steps == 10000:
            logger.info("检测到 Checkpoint 加载，自动禁用 Warmup (warmup_steps=0) 以保护预训练策略。")
            args.warmup_steps = 0

    # 监控器
    monitor = VectorizedTrainingMonitor(window_size=100)

    # 训练循环
    logger.info("开始向量化训练...")
    start_time = time.time()

    states = env.reset()  # (B, state_dim) on GPU
    total_steps = 0
    total_episodes = 0
    beta = 0.4
    best_success_rate = 0.0
    last_eval_episode = -1
    
    # 成功率监控 (用于 Adaptive Entropy)
    recent_success_history = deque(maxlen=50)

    # 记录每个环境的 episode 步数
    env_steps = np.zeros(args.num_envs, dtype=int)
    timing_window = deque(maxlen=20)
    last_heartbeat = time.time()

    # 缩放渐变
    current_pos_scale = args.initial_pos_scale
    pos_scale_delta = 0.0
    if args.pos_scale_steps > 0:
        pos_scale_delta = (args.final_pos_scale - args.initial_pos_scale) / args.pos_scale_steps

    while total_episodes < args.episodes:
        # 更新观测缩放
        if args.pos_scale_steps > 0 and total_episodes < args.pos_scale_steps:
             # 基于 episode 进行线性插值
             new_scale = args.initial_pos_scale + pos_scale_delta * total_episodes
             # 确保在范围内
             if pos_scale_delta > 0:
                 new_scale = min(new_scale, args.final_pos_scale)
             else:
                 new_scale = max(new_scale, args.final_pos_scale)
             
             if abs(new_scale - env.pos_scale) > 1e-4:
                 env.set_pos_scale(new_scale)
                 current_pos_scale = new_scale
        elif args.pos_scale_steps > 0 and total_episodes >= args.pos_scale_steps:
             # 确保最终值正确
             if abs(env.pos_scale - args.final_pos_scale) > 1e-4:
                 env.set_pos_scale(args.final_pos_scale)
                 current_pos_scale = args.final_pos_scale

        step_start = time.time()

        # 选择动作
        if total_steps < args.warmup_steps:
            # Warmup: 随机动作
            actions = torch.rand(args.num_envs, action_dim, device=DEVICE) * 2 - 1
        else:
            actions = agent.select_action_batch(states, evaluate=False)

        # 环境步进
        env_start = time.time()
        next_states, rewards, dones, infos = env.step(actions)
        env_time = time.time() - env_start

        convert_start = time.time()
        if agent.use_gpu_replay:
            # 仅将 dones 转为 numpy（用于统计/成功判定）
            dones_np = dones.detach().cpu().numpy()
            convert_time = time.time() - convert_start
            agent.store_transitions_batch(
                states, actions, rewards, next_states, dones
            )
        else:
            # 转换为 numpy（用于存入 CPU buffer）
            states_np = states.cpu().numpy()
            actions_np = actions.cpu().numpy()
            rewards_np = rewards.cpu().numpy()
            next_states_np = next_states.cpu().numpy()
            dones_np = dones.cpu().numpy()
            convert_time = time.time() - convert_start

            # 存储 transitions
            agent.store_transitions_batch(
                states_np, actions_np, rewards_np, next_states_np, dones_np
            )

        # 更新步数
        total_steps += args.num_envs
        env_steps += 1

        # 处理 episode 结束
        completed = agent.handle_episode_ends(dones_np, infos)
        for env_idx, reward, success in completed:
            total_episodes += 1
            monitor.update_episode(reward, success, env_steps[env_idx])
            recent_success_history.append(1 if success else 0)
            env_steps[env_idx] = 0

            if curriculum is not None:
                new_level = curriculum.update(success)
                if new_level is not None:
                    env.set_curriculum_level(new_level)
                    logger.info(f"[Curriculum] Environment updated to level {new_level}")

            if ramp is not None:
                new_vals = ramp.update(success)
                if new_vals is not None:
                    env.V_c_max, env.V_w_max = new_vals
                    logger.info(
                        f"[Ramp] 提升干扰上限: V_c_max={env.V_c_max:.3f}, "
                        f"V_w_max={env.V_w_max:.3f}"
                    )

        # ⚠️ 关键修复：重置已完成的环境！
        if dones.any():
            done_indices = torch.where(dones)[0]
            reset_obs = env.reset(env_indices=done_indices)
            # reset_obs 返回所有环境的 obs，只取被重置的部分
            next_states[done_indices] = reset_obs[done_indices]

        # 修复：Adaptive Entropy Update（防止并行环境下重复触发）
        if total_steps >= args.warmup_steps and len(recent_success_history) >= 20:
            if total_episodes % 50 == 0 and total_episodes > agent.last_entropy_update_episode:
                current_success = np.mean(recent_success_history)
                agent.update_target_entropy(current_success)
                agent.last_entropy_update_episode = total_episodes

        # 日志（每50个episode打印一次）
        if completed and total_episodes % 50 == 0:
            last_reward = completed[-1][1]
            stats = monitor.get_stats()
            elapsed = time.time() - start_time
            sps = total_steps / elapsed
            
            ent_val = agent.alpha if hasattr(agent, 'alpha') else 0.0

            logger.info(
                f"Episode {total_episodes}: "
                f"R={last_reward:.1f}, "
                f"AvgR={stats.get('avg_reward', 0):.1f}, "
                f"Succ={stats.get('success_rate', 0):.1%}, "
                f"Alpha={ent_val:.3f}, "
                f"Lvl={curriculum.get_current_level() if curriculum else env.curriculum_level}, "
                f"Scale={current_pos_scale:.2f}, "
                f"SPS={sps:.0f}"
            )

        # 更新网络 (使用修复后的 updates_per_step)
        update_time = 0.0
        if total_steps >= args.warmup_steps:
            update_start = time.time()
            for _ in range(updates_per_step):
                agent.update(beta)
            update_time = time.time() - update_start
            beta = min(1.0, beta + 0.0001)

        # 定期评估
        if should_run_eval(total_episodes, args.eval_interval, last_eval_episode):
            eval_reward, eval_success = evaluate_agent(
                agent, env, num_episodes=10, stochastic=args.stochastic_eval
            )
            last_eval_episode = total_episodes
            logger.info(f"[评估] Episode {total_episodes}: Reward = {eval_reward:.1f}, Success = {eval_success:.1%}")

            if eval_success > best_success_rate:
                best_success_rate = eval_success
                agent.save_models('best')
                logger.info(f"新最佳模型: {best_success_rate:.1%}")

        # 定期保存
        if total_episodes > 0 and total_episodes % args.save_interval == 0:
            agent.save_models(total_episodes)
            monitor.save_data(os.path.join(args.output_dir, 'training_data.json'))

        # 更新状态
        states = next_states

        # 速度统计
        step_time = time.time() - step_start
        timing_window.append({
            "env": env_time,
            "convert": convert_time,
            "update": update_time,
            "loop": step_time,
        })
        if step_time > 0:
            monitor.update_speed(args.num_envs / step_time)

        # 心跳日志：用于定位卡住位置
        now_ts = time.time()
        if should_log_heartbeat(last_heartbeat, now_ts, args.heartbeat_secs):
            if len(timing_window) > 0:
                avg_env = float(np.mean([t["env"] for t in timing_window]))
                avg_convert = float(np.mean([t["convert"] for t in timing_window]))
                avg_update = float(np.mean([t["update"] for t in timing_window]))
                avg_loop = float(np.mean([t["loop"] for t in timing_window]))
            else:
                avg_env = avg_convert = avg_update = avg_loop = 0.0

            env_min = int(env_steps.min()) if env_steps.size > 0 else 0
            env_max = int(env_steps.max()) if env_steps.size > 0 else 0
            buffer_stats = agent.memory.get_stats() if hasattr(agent, "memory") else {}
            gpu_mem_mb = 0.0
            if torch.cuda.is_available():
                gpu_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)

            logger.info(
                "[Heartbeat] ep=%s steps=%s env_ms=%.1f update_ms=%.1f convert_ms=%.1f loop_ms=%.1f "
                "env_steps[min=%s max=%s] buffer=%s gpu_mem=%.0fMB",
                total_episodes,
                total_steps,
                avg_env * 1000,
                avg_update * 1000,
                avg_convert * 1000,
                avg_loop * 1000,
                env_min,
                env_max,
                buffer_stats.get("main_buffer_size", 0),
                gpu_mem_mb,
            )

            if args.debug_timing and agent.last_update_timing:
                detail = agent.last_update_timing
                logger.info(
                    "[UpdateTiming] sample=%.1fms transfer=%.1fms critic=%.1fms actor=%.1fms "
                    "alpha=%.1fms priority=%.1fms soft=%.1fms total=%.1fms",
                    detail.get("sample", 0.0) * 1000,
                    detail.get("transfer", 0.0) * 1000,
                    detail.get("critic", 0.0) * 1000,
                    detail.get("actor", 0.0) * 1000,
                    detail.get("alpha", 0.0) * 1000,
                    detail.get("priority", 0.0) * 1000,
                    detail.get("soft_update", 0.0) * 1000,
                    detail.get("total", 0.0) * 1000,
                )

            slow_stages = detect_slow_stages(
                {
                    "env": env_time,
                    "update": update_time,
                    "convert": convert_time,
                    "loop": step_time,
                },
                args.slow_step_threshold,
            )
            if slow_stages:
                logger.warning(
                    "步骤耗时过长: %s (env=%.2fs update=%.2fs convert=%.2fs loop=%.2fs)",
                    ",".join(slow_stages),
                    env_time,
                    update_time,
                    convert_time,
                    step_time,
                )
            if args.debug_timing and agent.last_update_timing:
                slow_updates = detect_slow_stages(agent.last_update_timing, args.slow_step_threshold)
                if slow_updates:
                    logger.warning(
                        "Update阶段耗时过长: %s (sample=%.2fs transfer=%.2fs critic=%.2fs actor=%.2fs alpha=%.2fs priority=%.2fs soft=%.2fs)",
                        ",".join(slow_updates),
                        detail.get("sample", 0.0),
                        detail.get("transfer", 0.0),
                        detail.get("critic", 0.0),
                        detail.get("actor", 0.0),
                        detail.get("alpha", 0.0),
                        detail.get("priority", 0.0),
                        detail.get("soft_update", 0.0),
                    )

            last_heartbeat = now_ts

    # 训练结束
    training_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"训练完成!")
    logger.info(f"总时间: {training_time:.1f}s ({training_time/3600:.2f}h)")
    logger.info(f"总步数: {total_steps:,}")
    logger.info(f"总 Episodes: {total_episodes}")
    logger.info(f"平均速度: {total_steps/training_time:.0f} steps/sec")
    logger.info(f"最佳成功率: {best_success_rate:.1%}")
    logger.info("=" * 70)

    # 最终保存
    agent.save_models(total_episodes)
    monitor.save_data(os.path.join(args.output_dir, 'training_data.json'))


def main():
    parser = argparse.ArgumentParser(description='向量化 PER-SAC Otter USV 训练')

    # 环境参数
    parser.add_argument('--num_envs', type=int, default=64, help='并行环境数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # 训练参数
    parser.add_argument('--episodes', type=int, default=1000, help='总 episode 数')
    parser.add_argument('--warmup_steps', type=int, default=10000, help='warmup 步数')
    
    # 修复: 默认 updates_per_step 改为 None，以便自动计算
    parser.add_argument('--updates_per_step', type=int, default=None, help='每步更新次数 (覆盖 UTD)')
    parser.add_argument('--utd_ratio', type=float, default=1.0, help='更新/数据比率 (Default: 1.0)')
    
    parser.add_argument('--batch_size', type=int, default=256, help='批大小')
    parser.add_argument('--buffer_size', type=int, default=1_000_000, help='缓冲区大小')
    parser.add_argument('--heartbeat_secs', type=float, default=30.0, help='心跳日志间隔（秒）')
    parser.add_argument('--slow_step_threshold', type=float, default=5.0, help='单步阶段耗时告警阈值（秒）')
    parser.add_argument('--debug_timing', action='store_true', default=False, help='启用更新细分计时日志')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='使用 pin_memory 加速 CPU->GPU 传输')
    parser.add_argument('--use_gpu_replay', action='store_true', default=False, help='使用 GPU 经验回放缓冲区')

    # 学习率
    parser.add_argument('--actor_lr', type=float, default=1e-5, help='Actor 学习率')
    parser.add_argument('--critic_lr', type=float, default=1e-4, help='Critic 学习率')

    # 课程学习
    parser.add_argument('--initial_curriculum_level', type=int, default=1, help='初始课程等级')
    parser.add_argument('--curriculum_patience', type=int, default=200, help='课程升级前需要稳定的episode数')
    parser.add_argument('--curriculum_window', type=int, default=100, help='课程成功率统计窗口')
    parser.add_argument('--curriculum_auto', action='store_true', default=True, help='启用课程自动升级')
    parser.add_argument('--no_curriculum_auto', action='store_false', dest='curriculum_auto', help='关闭课程自动升级')

    # 干扰爬坡
    parser.add_argument('--ramp_disturbance', action='store_true', default=False, help='启用干扰细步进')
    parser.add_argument('--ramp_vc_steps', type=str, default='0.03,0.05,0.07,0.10', help='海流步进列表')
    parser.add_argument('--ramp_vw_steps', type=str, default='0.5,0.8,1.1,1.5', help='风速步进列表')
    parser.add_argument('--ramp_success_threshold', type=float, default=0.6, help='步进触发成功率阈值')
    parser.add_argument('--ramp_window', type=int, default=50, help='步进成功率统计窗口')

    # 干扰参数
    parser.add_argument('--enable_current', action='store_true', default=True)
    parser.add_argument('--no_current', action='store_false', dest='enable_current')
    parser.add_argument('--V_c_max', type=float, default=0.3)
    parser.add_argument('--current_mode', type=str, default='curriculum')
    parser.add_argument('--enable_wind', action='store_true', default=True)
    parser.add_argument('--no_wind', action='store_false', dest='enable_wind')
    parser.add_argument('--V_w_max', type=float, default=5.0)
    parser.add_argument('--wind_mode', type=str, default='curriculum')

    # 检查点
    parser.add_argument('--load_checkpoint', type=str, default=None, help='加载检查点 (episode数字或"best")')
    parser.add_argument('--reset_optimizer', action='store_true', default=False, help='重置优化器状态 (用于微调)')
    
    # 观测缩放渐变
    parser.add_argument('--initial_pos_scale', type=float, default=1.0, help='初始位置误差缩放')
    parser.add_argument('--final_pos_scale', type=float, default=1.0, help='最终位置误差缩放')
    parser.add_argument('--pos_scale_steps', type=int, default=0, help='缩放渐变步数 (episodes)')

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    default_checkpoint_dir = os.path.join(root_dir, 'assets', 'checkpoints', 'checkpoints_vec')
    default_output_dir = os.path.join(root_dir, 'assets', 'results', 'results_vec')

    parser.add_argument('--load_from_original', type=str, default=None, help='从原始检查点目录加载')
    parser.add_argument('--checkpoint_dir', type=str, default=default_checkpoint_dir)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)

    # 日志
    parser.add_argument('--eval_interval', type=int, default=50, help='评估间隔')
    parser.add_argument('--save_interval', type=int, default=100, help='保存间隔')
    parser.add_argument('--stochastic_eval', action='store_true', default=False, help='使用随机策略进行评估 (调试用)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
