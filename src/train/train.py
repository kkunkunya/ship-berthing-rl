"""
PER-SAC Otter USV 靠泊训练 - 完整修复版本

修复内容：
1. 课程学习机制 - 逐步提高难度
2. 成功轨迹回放 - 防止遗忘好策略
3. 动态熵调整 - 根据成功率调整探索程度
4. 学习率调度 - 后期降低学习率稳定训练
5. 优先级衰减 - 防止PER过度偏向某些样本
6. 保留所有原有可视化功能
"""


import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from collections import deque
import time
import json
import logging
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# 导入修复后的环境和PER缓冲区
from ..envs.ship_berthing_env import OtterBerthingTrackingEnv
from ..utils.per_buffer import PrioritizedReplayBuffer, SuccessTrajectoryBuffer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


# ============== 顶刊论文级别的全局matplotlib设置 ==============
def setup_publication_style():
    """设置顶刊论文级别的图表样式"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia', 'STIXGeneral'],
        'font.size': 16,
        'mathtext.fontset': 'stix',
        'axes.labelsize': 24,
        'axes.titlesize': 26,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.minor.size': 5,
        'ytick.minor.size': 5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'xtick.top': True,
        'ytick.right': True,
        'legend.fontsize': 16,
        'legend.framealpha': 0.95,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        'legend.frameon': True,
        'lines.linewidth': 2.5,
        'lines.markersize': 10,
        'axes.linewidth': 1.8,
        'axes.edgecolor': 'black',
        'grid.alpha': 0.4,
        'grid.linewidth': 0.8,
        'grid.linestyle': '--',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'figure.figsize': (12, 8),
        'figure.autolayout': False,
    })


setup_publication_style()

# 专业配色方案
COLORS = {
    'primary': '#0072B2',
    'secondary': '#E69F00',
    'tertiary': '#009E73',
    'quaternary': '#D55E00',
    'purple': '#CC79A7',
    'cyan': '#56B4E9',
    'yellow': '#F0E442',
    'gray': '#999999',
    'black': '#000000',
}


# ============== 小船图标绘制函数 ==============
def draw_ship_icon(ax, x, y, psi, scale=3.5, color='#FFB6C1', alpha=0.8, zorder=8):
    """绘制船只图标"""
    ship_shape = np.array([
        [0, 1.0], [0.4, 0.3], [0.4, -0.7], [0.2, -1.0],
        [-0.2, -1.0], [-0.4, -0.7], [-0.4, 0.3], [0, 1.0]
    ])
    ship_shape = ship_shape * scale

    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    rotation_matrix = np.array([[cos_psi, -sin_psi], [sin_psi, cos_psi]])
    rotated_ship = ship_shape @ rotation_matrix.T
    rotated_ship[:, 0] += x
    rotated_ship[:, 1] += y

    ship_patch = plt.Polygon(rotated_ship, facecolor=color,
                             edgecolor=COLORS['primary'], linewidth=2.0,
                             linestyle='--', alpha=alpha, zorder=zorder)
    ax.add_patch(ship_patch)
    return ship_patch


# ============== 网络架构 ==============
class ImprovedActorNetwork(nn.Module):
    """改进的Actor网络"""

    def __init__(self, input_dim, action_dim, fc1_dim=256, fc2_dim=256):
        super(ImprovedActorNetwork, self).__init__()
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
        log_std = T.clamp(self.log_std(x), -20, 2)
        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = T.tanh(z)
        log_prob = normal.log_prob(z) - T.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob


class ImprovedCriticNetwork(nn.Module):
    """改进的Critic网络"""

    def __init__(self, input_dim, action_dim, fc1_dim=256, fc2_dim=256):
        super(ImprovedCriticNetwork, self).__init__()
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
        x = T.cat([state, action], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.q(x)


# ============== 新增：课程学习管理器 ==============
class CurriculumManager:
    """课程学习管理器 - 根据训练进度自动调整环境难度"""

    def __init__(self,
                 level_thresholds=[0.5, 0.6, 0.7, 0.8],
                 window_size=100,
                 patience=200,
                 initial_level=1):  # 新增：初始级别参数
        self.level_thresholds = level_thresholds
        self.window_size = window_size
        self.patience = patience
        self.current_level = initial_level  # 使用初始级别
        self.max_level = len(level_thresholds) + 1
        self.success_history = deque(maxlen=window_size)
        self.level_stable_episodes = 0
        self.level_history = []

    def update(self, success):
        """更新课程状态，返回新等级（如果升级）"""
        self.success_history.append(1 if success else 0)
        self.level_stable_episodes += 1

        if len(self.success_history) < self.window_size:
            return None

        current_success_rate = np.mean(self.success_history)

        if self.current_level < self.max_level:
            threshold = self.level_thresholds[self.current_level - 1]
            if current_success_rate >= threshold and self.level_stable_episodes >= self.patience:
                self.current_level += 1
                self.level_stable_episodes = 0
                self.success_history.clear()
                self.level_history.append({
                    'level': self.current_level,
                    'success_rate': current_success_rate
                })
                logger.info(f"[Curriculum] 升级到等级 {self.current_level}! 成功率: {current_success_rate:.2%}")
                return self.current_level
        return None

    def get_current_level(self):
        return self.current_level

    def get_success_rate(self):
        return np.mean(self.success_history) if len(self.success_history) > 0 else 0.0


# ============== 新增：动态熵系数调整器 ==============
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


# ============== 新增：学习率调度器 ==============
class WarmupCosineScheduler:
    """预热 + 余弦退火学习率调度器"""

    def __init__(self, optimizer, warmup_episodes, total_episodes, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_episodes = warmup_episodes
        self.total_episodes = total_episodes
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_episode = 0

    def step(self, episode=None):
        if episode is not None:
            self.current_episode = episode
        else:
            self.current_episode += 1

        if self.current_episode < self.warmup_episodes:
            warmup_factor = self.current_episode / self.warmup_episodes
            new_lrs = [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            progress = (self.current_episode - self.warmup_episodes) / (self.total_episodes - self.warmup_episodes)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor
            new_lrs = [base_lr * factor for base_lr in self.base_lrs]

        for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = new_lr

        return new_lrs[0]

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


# ============== 增强版 PER-SAC 智能体 ==============
class EnhancedPerSacAgent:
    """增强版 PER-SAC Agent - 包含课程学习、成功轨迹回放等功能"""

    def __init__(self, state_dim, action_dim,
                 alpha=0.2, gamma=0.99, tau=0.005,
                 actor_lr=5e-5, critic_lr=1e-4,
                 max_size=1000000, batch_size=256,
                 fc1_dim=256, fc2_dim=256,
                 automatic_entropy_tuning=True,
                 ckpt_dir='./checkpoints/',
                 total_episodes=2000,
                 warmup_episodes=200):

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.checkpoint_dir = ckpt_dir
        self.action_dim = action_dim

        # 网络
        self.actor = ImprovedActorNetwork(state_dim, action_dim, fc1_dim, fc2_dim).to(device)
        self.critic1 = ImprovedCriticNetwork(state_dim, action_dim, fc1_dim, fc2_dim).to(device)
        self.critic2 = ImprovedCriticNetwork(state_dim, action_dim, fc1_dim, fc2_dim).to(device)
        self.target_critic1 = ImprovedCriticNetwork(state_dim, action_dim, fc1_dim, fc2_dim).to(device)
        self.target_critic2 = ImprovedCriticNetwork(state_dim, action_dim, fc1_dim, fc2_dim).to(device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor_optimizer = T.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = T.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = T.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # 学习率调度器
        self.actor_scheduler = WarmupCosineScheduler(self.actor_optimizer, warmup_episodes, total_episodes, 0.1)
        self.critic1_scheduler = WarmupCosineScheduler(self.critic1_optimizer, warmup_episodes, total_episodes, 0.1)
        self.critic2_scheduler = WarmupCosineScheduler(self.critic2_optimizer, warmup_episodes, total_episodes, 0.1)

        # 自适应熵
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if automatic_entropy_tuning:
            self.entropy_manager = AdaptiveEntropyManager(action_dim)
            self.target_entropy = self.entropy_manager.get_target_entropy()
            self.log_alpha = T.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=3e-4)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

        # 增强版PER缓冲区
        self.memory = PrioritizedReplayBuffer(max_size, state_dim, action_dim, batch_size,
                                              alpha=0.6, success_priority_boost=2.0, success_buffer_ratio=0.1)

        # 成功轨迹缓冲区
        self.success_trajectory_buffer = SuccessTrajectoryBuffer(max_trajectories=100)
        self.current_trajectory = []

        # 统计
        self.training_stats = {
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'alpha': deque(maxlen=100),
            'q_values': deque(maxlen=100),
            'target_entropy': deque(maxlen=100),
            'actor_lr': deque(maxlen=100)
        }

        os.makedirs(ckpt_dir, exist_ok=True)

    def select_action(self, state, evaluate=False):
        state_tensor = T.FloatTensor(state).unsqueeze(0).to(device)
        with T.no_grad():
            if evaluate:
                mean, _ = self.actor(state_tensor)
                action = T.tanh(mean)
            else:
                action, _ = self.actor.sample(state_tensor)
        return action.cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done, is_success=False):
        self.memory.store_transition(state, action, reward, next_state, done, is_success)
        self.current_trajectory.append((state.copy(), action.copy(), reward, next_state.copy(), done))

    def end_episode(self, total_reward, success):
        """episode结束时调用"""
        if success and len(self.current_trajectory) > 0:
            self.success_trajectory_buffer.store_trajectory(self.current_trajectory.copy(), total_reward)
            logger.info(f"[Success Buffer] 存储成功轨迹，总数: {len(self.success_trajectory_buffer)}")
        self.current_trajectory = []

    def replay_success_trajectories(self, n_trajectories=1):
        """回放成功轨迹到经验缓冲区"""
        trajectories = self.success_trajectory_buffer.sample_trajectory(n=n_trajectories, prioritize_by_reward=True)
        for traj in trajectories:
            for state, action, reward, next_state, done in traj:
                self.memory.store_transition(state, action, reward, next_state, done, is_success=True)
        if len(trajectories) > 0:
            logger.info(f"[Replay] 回放了 {len(trajectories)} 条成功轨迹")

    def update_target_entropy(self, success_rate):
        if self.automatic_entropy_tuning:
            self.target_entropy = self.entropy_manager.update(success_rate)

    def step_schedulers(self, episode):
        actor_lr = self.actor_scheduler.step(episode)
        self.critic1_scheduler.step(episode)
        self.critic2_scheduler.step(episode)
        self.training_stats['actor_lr'].append(actor_lr)
        return actor_lr

    def update(self, beta=0.4, success_sample_ratio=0.2):
        if not self.memory.ready():
            return

        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample_buffer(beta, success_sample_ratio)

        states = T.FloatTensor(states).to(device)
        actions = T.FloatTensor(actions).to(device)
        rewards = T.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = T.FloatTensor(next_states).to(device)
        dones = T.FloatTensor(dones).unsqueeze(1).to(device)
        weights = T.FloatTensor(weights).unsqueeze(1).to(device)

        # Critic更新
        with T.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = T.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        td_errors = ((q1 - target_q).abs() + (q2 - target_q).abs()) / 2
        td_errors = td_errors.detach().cpu().numpy().squeeze()
        self.memory.update_priorities(indices, td_errors)

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

        # Actor更新
        actions_pred, log_probs = self.actor.sample(states)
        q1_pred = self.critic1(states, actions_pred)
        q2_pred = self.critic2(states, actions_pred)
        q_pred = T.min(q1_pred, q2_pred)

        actor_loss = (self.alpha * log_probs - q_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Alpha更新
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        self._soft_update()

        # 记录统计
        self.training_stats['actor_loss'].append(actor_loss.item())
        self.training_stats['critic_loss'].append((critic1_loss.item() + critic2_loss.item()) / 2)
        if self.automatic_entropy_tuning:
            self.training_stats['alpha'].append(self.alpha.item())
            self.training_stats['target_entropy'].append(self.target_entropy)
        self.training_stats['q_values'].append(q_pred.mean().item())

    def _soft_update(self):
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_models(self, episode):
        print(f'... saving models at episode {episode} ...')
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
        T.save(checkpoint, os.path.join(self.checkpoint_dir, f'per_sac_otter_checkpoint_{episode}.pth'))

    def load_models(self, episode_or_path):
        """加载模型检查点

        Args:
            episode_or_path: episode 数字，或完整的检查点文件路径
        """
        # 判断是 episode 数字还是完整路径
        if isinstance(episode_or_path, str) and os.path.exists(episode_or_path):
            checkpoint_path = episode_or_path
        elif isinstance(episode_or_path, str) and episode_or_path.endswith('.pth'):
            # 尝试作为路径处理
            checkpoint_path = episode_or_path
        else:
            # 作为 episode 数字处理
            checkpoint_path = os.path.join(self.checkpoint_dir, f'per_sac_otter_checkpoint_{episode_or_path}.pth')

        print(f'... loading models from {checkpoint_path} ...')
        checkpoint = T.load(checkpoint_path, map_location=device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])

        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])


# ============== 训练监控器 ==============
class TrainingMonitor:
    """训练监控器"""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_episodes = []
        self.cross_track_errors = []
        self.distance_to_goals = []
        self.final_velocities = []
        self.test_episodes = []
        self.test_rewards = []
        self.test_success_rates = []
        self.reward_components = []
        self.curriculum_levels = []  # 新增：记录课程等级

    def update_episode(self, reward, length, success, info, reward_history=None, curriculum_level=1):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.success_episodes.append(1.0 if success else 0.0)
        self.curriculum_levels.append(curriculum_level)

        if 'cross_track_error' in info:
            self.cross_track_errors.append(info['cross_track_error'])
        if 'distance_to_goal' in info:
            self.distance_to_goals.append(info['distance_to_goal'])
        if 'resultant_velocity' in info:
            self.final_velocities.append(info['resultant_velocity'])

        if reward_history:
            self.reward_components.append(reward_history)

    def update_test(self, episode, reward, success_rate):
        self.test_episodes.append(episode)
        self.test_rewards.append(reward)
        self.test_success_rates.append(success_rate)

    def get_recent_stats(self):
        stats = {}
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-self.window_size:]
            stats['avg_reward'] = np.mean(recent_rewards)
            stats['std_reward'] = np.std(recent_rewards)

        if len(self.success_episodes) > 0:
            recent_success = self.success_episodes[-self.window_size:]
            stats['success_rate'] = np.mean(recent_success)

        if len(self.cross_track_errors) > 0:
            recent_errors = self.cross_track_errors[-self.window_size:]
            stats['avg_cross_track_error'] = np.mean(recent_errors)

        if len(self.final_velocities) > 0:
            recent_velocities = self.final_velocities[-self.window_size:]
            stats['avg_final_velocity'] = np.mean(recent_velocities)

        return stats

    def save_data(self, filepath):
        def convert_to_native(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, bool):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            else:
                return obj

        data = {
            'episode_rewards': convert_to_native(self.episode_rewards),
            'episode_lengths': convert_to_native(self.episode_lengths),
            'success_episodes': convert_to_native(self.success_episodes),
            'test_episodes': convert_to_native(self.test_episodes),
            'test_rewards': convert_to_native(self.test_rewards),
            'test_success_rates': convert_to_native(self.test_success_rates),
            'cross_track_errors': convert_to_native(self.cross_track_errors),
            'distance_to_goals': convert_to_native(self.distance_to_goals),
            'final_velocities': convert_to_native(self.final_velocities),
            'curriculum_levels': convert_to_native(self.curriculum_levels),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def plot_training_progress_separate(self, save_dir, episode_num):
        """分别保存各个训练进度图"""
        os.makedirs(save_dir, exist_ok=True)

        # 1. 奖励曲线
        if len(self.episode_rewards) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            episodes = range(len(self.episode_rewards))

            ax.plot(episodes, self.episode_rewards, alpha=0.2, color=COLORS['gray'],
                    linewidth=0.8, label='Episode Reward')

            if len(self.episode_rewards) > self.window_size:
                moving_avg = np.convolve(self.episode_rewards,
                                         np.ones(self.window_size) / self.window_size,
                                         mode='valid')
                ax.plot(range(self.window_size - 1, len(self.episode_rewards)),
                        moving_avg, color=COLORS['quaternary'], linewidth=2.5,
                        label=f'Moving Average ({self.window_size} eps)')

            cumulative_avg = np.cumsum(self.episode_rewards) / np.arange(1, len(self.episode_rewards) + 1)
            ax.plot(episodes, cumulative_avg, color=COLORS['tertiary'], linewidth=2.5,
                    linestyle='--', label='Cumulative Average')

            ax.set_xlabel(r'$\mathbf{\mathit{Episode}}$', fontsize=24, fontweight='bold')
            ax.set_ylabel(r'$\mathbf{\mathit{Reward}}$', fontsize=24, fontweight='bold')
            ax.set_title('Training Rewards (PER-SAC)', fontsize=26, fontweight='bold', pad=15)
            ax.legend(loc='lower right', fontsize=16, framealpha=0.95, edgecolor='black')
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
            ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=8)

            for spine in ax.spines.values():
                spine.set_linewidth(1.6)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'rewards_{episode_num}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 2. 成功率
        if len(self.success_episodes) > self.window_size:
            fig, ax = plt.subplots(figsize=(12, 8))
            success_rate = np.convolve(self.success_episodes,
                                       np.ones(self.window_size) / self.window_size,
                                       mode='valid')
            ax.plot(range(self.window_size - 1, len(self.success_episodes)),
                    success_rate, color=COLORS['tertiary'], linewidth=2.5)
            ax.fill_between(range(self.window_size - 1, len(self.success_episodes)),
                            0, success_rate, alpha=0.2, color=COLORS['tertiary'])

            ax.set_xlabel(r'$\mathbf{\mathit{Episode}}$', fontsize=24, fontweight='bold')
            ax.set_ylabel(r'$\mathbf{\mathit{Success\ Rate}}$', fontsize=24, fontweight='bold')
            ax.set_title('Success Rate (Moving Average)', fontsize=26, fontweight='bold', pad=15)
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
            ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=8)

            for spine in ax.spines.values():
                spine.set_linewidth(1.6)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'success_rate_{episode_num}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 3. 最终速度
        if len(self.final_velocities) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(self.final_velocities, alpha=0.6, color=COLORS['purple'], linewidth=1.5)
            ax.axhline(y=0.03, color=COLORS['quaternary'], linestyle='--', linewidth=2.5,
                       alpha=0.9, label='Target (0.03 m/s)')

            ax.set_xlabel(r'$\mathbf{\mathit{Episode}}$', fontsize=24, fontweight='bold')
            ax.set_ylabel(r'$\mathbf{\mathit{Final\ Velocity\ (m/s)}}$', fontsize=24, fontweight='bold')
            ax.set_title('Final Velocity at Berthing', fontsize=26, fontweight='bold', pad=15)
            ax.legend(loc='upper right', fontsize=16, framealpha=0.95, edgecolor='black')
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
            ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=8)

            for spine in ax.spines.values():
                spine.set_linewidth(1.6)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'final_velocity_{episode_num}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 4. 测试结果
        if len(self.test_episodes) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(self.test_episodes, self.test_success_rates,
                    'o-', color=COLORS['purple'], linewidth=2.5, markersize=10,
                    markerfacecolor='white', markeredgewidth=2.5)

            ax.set_xlabel(r'$\mathbf{\mathit{Episode}}$', fontsize=24, fontweight='bold')
            ax.set_ylabel(r'$\mathbf{\mathit{Test\ Success\ Rate}}$', fontsize=24, fontweight='bold')
            ax.set_title('Test Performance', fontsize=26, fontweight='bold', pad=15)
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
            ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=8)

            for spine in ax.spines.values():
                spine.set_linewidth(1.6)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'test_performance_{episode_num}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 5. 平均奖励演化 - 已移除legend
        if len(self.episode_rewards) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            episodes = np.arange(len(self.episode_rewards))

            ax.plot(episodes, self.episode_rewards, alpha=0.12, color=COLORS['gray'], linewidth=0.5)

            windows = [20]
            colors = [COLORS['primary']]

            for window, color in zip(windows, colors):
                if len(self.episode_rewards) > window:
                    avg = np.convolve(self.episode_rewards, np.ones(window) / window, mode='valid')
                    ax.plot(range(window - 1, len(self.episode_rewards)), avg,
                            color=color, linewidth=2.5)

            ax.set_xlabel(r'$\mathbf{\mathit{Number\ of\ Episode}}$', fontsize=24, fontweight='bold')
            ax.set_ylabel(r'$\mathbf{\mathit{Reward}}$', fontsize=24, fontweight='bold')
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
            ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=8)

            for spine in ax.spines.values():
                spine.set_linewidth(1.6)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'avg_reward_evolution_{episode_num}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 6. 新增：课程等级变化图
        if len(self.curriculum_levels) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(range(len(self.curriculum_levels)), self.curriculum_levels,
                    color=COLORS['primary'], linewidth=2.5, drawstyle='steps-post')
            ax.fill_between(range(len(self.curriculum_levels)), 0, self.curriculum_levels,
                            alpha=0.2, color=COLORS['primary'], step='post')

            ax.set_xlabel(r'$\mathbf{\mathit{Episode}}$', fontsize=24, fontweight='bold')
            ax.set_ylabel(r'$\mathbf{\mathit{Curriculum\ Level}}$', fontsize=24, fontweight='bold')
            ax.set_title('Curriculum Learning Progress', fontsize=26, fontweight='bold', pad=15)
            ax.set_ylim([0, 5])
            ax.set_yticks([1, 2, 3, 4])
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
            ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=8)

            for spine in ax.spines.values():
                spine.set_linewidth(1.6)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'curriculum_level_{episode_num}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"Training progress plots saved to {save_dir}")


# ============== 轨迹可视化函数 ==============
def plot_trajectory_separate(actual_trajectory, reference_trajectory, berth_info,
                             episode, save_dir, env=None):
    """分别保存轨迹分析的各个子图"""
    os.makedirs(save_dir, exist_ok=True)

    # 提取数据
    actual_x = np.array([state['x'] for state in actual_trajectory])
    actual_y = np.array([state['y'] for state in actual_trajectory])
    actual_psi = np.array([state['psi'] for state in actual_trajectory])
    actual_u = np.array([state['u'] for state in actual_trajectory])

    if 'vm' in actual_trajectory[0]:
        actual_vm = np.array([state['vm'] for state in actual_trajectory])
        xG = env.ship.xG if hasattr(env.ship, 'xG') else 0.2
        actual_r = np.array([state['r'] for state in actual_trajectory])
        actual_v = actual_vm + xG * actual_r
    else:
        actual_v = np.array([state['v'] for state in actual_trajectory])
        actual_vm = actual_v
        actual_r = np.array([state['r'] for state in actual_trajectory])

    time_steps = np.arange(len(actual_trajectory))
    dt = 0.1
    time_seconds = time_steps * dt

    # ========== 1. 主轨迹对比图 ==========
    fig, ax = plt.subplots(figsize=(16, 11))

    if berth_info:
        berth_corners = berth_info['corners']
        berth_right_edge = np.max(berth_corners[:, 0])
        ship_beam = env.ship_beam if hasattr(env, 'ship_beam') else 1.08
        coast_offset_distance = ship_beam * 0.5
        coast_x_position = berth_right_edge - coast_offset_distance

        trajectory_length = env.trajectory.total_length
        coast_offset = 0.2 * trajectory_length

        min_y = min(np.min(actual_y), np.min(reference_trajectory[:, 1]))
        max_y = max(np.max(actual_y), np.max(reference_trajectory[:, 1]))
        coast_y = np.linspace(min_y - coast_offset, max_y + coast_offset, 100)
        coast_x = coast_x_position * np.ones_like(coast_y)

        x_min = min(np.min(actual_x), np.min(reference_trajectory[:, 0])) - 0.3 * trajectory_length
        x_max = coast_x_position + 0.3 * trajectory_length
        y_min = min(coast_y)
        y_max = max(coast_y)

        sea_color = '#d4e6f1'
        land_color = '#d5f5e3'

        ax.fill([x_min, x_min, coast_x_position, coast_x_position],
                [y_min, y_max, y_max, y_min],
                color=sea_color, alpha=0.6, edgecolor='none', zorder=0)
        ax.fill([coast_x_position, coast_x_position, x_max, x_max],
                [y_min, y_max, y_max, y_min],
                color=land_color, alpha=0.6, edgecolor='none', zorder=0)
        ax.plot(coast_x, coast_y, 'k-', linewidth=4, zorder=1)

        label_offset = 0.15 * trajectory_length
        ax.text(coast_x_position - label_offset, np.mean(coast_y), 'Water Area',
                fontsize=20, color='#1a5276', rotation=0,
                ha='center', va='center', fontweight='bold', fontstyle='italic', zorder=2)
        ax.text(coast_x_position + label_offset / 2, np.mean(coast_y), 'Dock Area',
                fontsize=20, color='#145a32', rotation=0,
                ha='center', va='center', fontweight='bold', fontstyle='italic', zorder=2)

    ref_line, = ax.plot(reference_trajectory[:, 0], reference_trajectory[:, 1],
                        color=COLORS['tertiary'], linestyle='--', linewidth=3.5,
                        label='Reference Trajectory', alpha=0.85, zorder=3)
    actual_line, = ax.plot(actual_x, actual_y, color=COLORS['primary'], linestyle='-',
                           linewidth=3, label='Actual Trajectory (PER-SAC)', zorder=4)

    berth_rect = None
    if berth_info and env:
        ship_length = env.ship_length if hasattr(env, 'ship_length') else 2.0
        ship_beam = env.ship_beam if hasattr(env, 'ship_beam') else 1.08
        berth_length = ship_length * 1.6
        berth_width = ship_beam * 1.3
        berth_center = berth_info['position']
        berth_offset_y = berth_length * 0.3
        adjusted_berth_y = berth_center[1] - berth_offset_y
        berth_x_max = coast_x_position
        berth_x_min = berth_x_max - berth_width

        berth_rect = plt.Rectangle(
            (berth_x_min, adjusted_berth_y - berth_length / 2),
            berth_width, berth_length,
            fill=True, facecolor='#fcf3cf', edgecolor=COLORS['quaternary'],
            linewidth=3.5, alpha=0.8, zorder=5
        )
        ax.add_patch(berth_rect)

    start_point = ax.scatter(actual_x[0], actual_y[0], c=COLORS['tertiary'], s=350, marker='*',
                             zorder=7, edgecolors='black', linewidth=2.5)
    end_point = ax.scatter(actual_x[-1], actual_y[-1], c=COLORS['quaternary'], s=350, marker='*',
                           zorder=7, edgecolors='black', linewidth=2.5)

    legend1_handles = [ref_line, actual_line]
    legend1_labels = ['Reference Trajectory', 'Actual Trajectory (PER-SAC)']
    if berth_rect is not None:
        legend1_handles.append(berth_rect)
        legend1_labels.append('Berth Area')

    legend1 = ax.legend(legend1_handles, legend1_labels,
                        loc='upper right', fontsize=16, framealpha=0.95, edgecolor='black')
    legend2_handles = [start_point, end_point]
    legend2_labels = ['Start Point', 'End Point']
    legend2 = ax.legend(legend2_handles, legend2_labels,
                        loc='lower left', fontsize=16, framealpha=0.95, edgecolor='black')
    ax.add_artist(legend1)

    # 时间标注和船只图标
    time_interval = 10.0
    time_label_steps = int(time_interval / dt)

    if berth_info and env:
        ship_scale = env.trajectory.total_length * 0.018
    else:
        ship_scale = 3.5

    total_time = time_seconds[-1]
    final_idx = len(actual_trajectory) - 1
    final_time = time_seconds[final_idx]

    time_labels_to_plot = []
    for t in np.arange(0, total_time, time_interval):
        idx = int(t / dt)
        if idx < len(actual_trajectory):
            time_labels_to_plot.append((idx, t))

    if len(time_labels_to_plot) > 0:
        last_time = time_labels_to_plot[-1][1]
        if final_time - last_time >= 5:
            time_labels_to_plot.append((final_idx, final_time))

    for idx, t in time_labels_to_plot:
        ship_heading = actual_psi[idx] - np.pi / 2
        draw_ship_icon(ax, actual_x[idx], actual_y[idx], ship_heading,
                       scale=ship_scale, color='#FFB6C1', alpha=0.85, zorder=8)

    for idx, t in time_labels_to_plot:
        is_first = (t == 0)
        if idx > 0 and idx < len(actual_x) - 1:
            dx = actual_x[min(idx + 10, len(actual_x) - 1)] - actual_x[max(idx - 10, 0)]
            dy = actual_y[min(idx + 10, len(actual_y) - 1)] - actual_y[max(idx - 10, 0)]
        elif idx == 0:
            dx = actual_x[min(10, len(actual_x) - 1)] - actual_x[0]
            dy = actual_y[min(10, len(actual_y) - 1)] - actual_y[0]
        else:
            dx = actual_x[-1] - actual_x[max(-10, -len(actual_x))]
            dy = actual_y[-1] - actual_y[max(-10, -len(actual_y))]

        angle = np.arctan2(dy, dx)
        if is_first:
            offset_x = 0
            offset_y = -6
        else:
            offset_magnitude = 7
            offset_x = offset_magnitude * np.sin(angle)
            offset_y = -offset_magnitude * np.cos(angle)

        time_label = f'{int(t)}s'
        ax.annotate(time_label, (actual_x[idx], actual_y[idx]),
                    xytext=(actual_x[idx] + offset_x, actual_y[idx] + offset_y),
                    fontsize=15, color='darkblue', weight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))

    # 小视窗
    if env and hasattr(env, 'trajectory') and hasattr(env.trajectory, 'turn_point_coords'):
        turn_point = env.trajectory.turn_point_coords
        inset_center_x = turn_point[0]
        inset_center_y = turn_point[1]
    else:
        curvatures = np.zeros(len(actual_x))
        for i in range(5, len(actual_x) - 5):
            v1 = np.array([actual_x[i] - actual_x[i - 5], actual_y[i] - actual_y[i - 5]])
            v2 = np.array([actual_x[i + 5] - actual_x[i], actual_y[i + 5] - actual_y[i]])
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                v1 = v1 / np.linalg.norm(v1)
                v2 = v2 / np.linalg.norm(v2)
                curvatures[i] = np.abs(np.cross(v1, v2))
        turn_idx = np.argmax(curvatures)
        inset_center_x = actual_x[turn_idx]
        inset_center_y = actual_y[turn_idx]

    window_half_size = 4.5
    inset_x_min = inset_center_x - window_half_size
    inset_x_max = inset_center_x + window_half_size
    inset_y_min = inset_center_y - window_half_size
    inset_y_max = inset_center_y + window_half_size

    if berth_info:
        rect_left_axes = (inset_y_min - y_min) / (y_max - y_min)
        rect_right_axes = (inset_y_max - y_min) / (y_max - y_min)
        rect_top_axes = (inset_x_max - x_min) / (x_max - x_min)
        inset_width = 0.13
        inset_height = 0.18
        rect_center_axes = (rect_left_axes + rect_right_axes) / 2
        inset_left = rect_center_axes - inset_width / 2
        inset_bottom = rect_top_axes + 0.005
    else:
        inset_left = 0.15
        inset_bottom = 0.55
        inset_width = 0.13
        inset_height = 0.18

    axins = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height])
    axins.set_facecolor('#f8f9fa')

    ref_mask = ((reference_trajectory[:, 0] >= inset_x_min - 1) &
                (reference_trajectory[:, 0] <= inset_x_max + 1) &
                (reference_trajectory[:, 1] >= inset_y_min - 1) &
                (reference_trajectory[:, 1] <= inset_y_max + 1))
    axins.plot(reference_trajectory[ref_mask, 0], reference_trajectory[ref_mask, 1],
               color=COLORS['tertiary'], linestyle='--', linewidth=2.5, alpha=0.85)

    actual_mask = ((actual_x >= inset_x_min - 1) & (actual_x <= inset_x_max + 1) &
                   (actual_y >= inset_y_min - 1) & (actual_y <= inset_y_max + 1))
    axins.plot(actual_x[actual_mask], actual_y[actual_mask],
               color=COLORS['primary'], linewidth=2.5)

    axins.set_xlim(inset_x_min, inset_x_max)
    axins.set_ylim(inset_y_min, inset_y_max)
    axins.set_aspect('equal')
    axins.tick_params(axis='both', labelsize=10)
    for spine in axins.spines.values():
        spine.set_linewidth(2)
        spine.set_color('#333333')

    rect = plt.Rectangle((inset_x_min, inset_y_min),
                         inset_x_max - inset_x_min, inset_y_max - inset_y_min,
                         fill=False, edgecolor='#333333', linewidth=1.5, zorder=10)
    ax.add_patch(rect)

    con1 = ConnectionPatch(xyA=(inset_x_min, inset_y_max), coordsA=ax.transData,
                           xyB=(0, 0), coordsB=axins.transAxes,
                           arrowstyle="-", color='#333333', lw=1.0)
    fig.add_artist(con1)
    con2 = ConnectionPatch(xyA=(inset_x_max, inset_y_max), coordsA=ax.transData,
                           xyB=(1, 0), coordsB=axins.transAxes,
                           arrowstyle="-", color='#333333', lw=1.0)
    fig.add_artist(con2)

    # 统计信息
    deviations = []
    for i in range(len(actual_trajectory)):
        point = np.array([actual_x[i], actual_y[i]])
        distances = np.linalg.norm(reference_trajectory - point, axis=1)
        deviations.append(np.min(distances))
    deviations = np.array(deviations)

    max_deviation = np.max(deviations)
    avg_deviation = np.mean(deviations)
    final_position_error = np.linalg.norm([
        actual_x[-1] - reference_trajectory[-1, 0],
        actual_y[-1] - reference_trajectory[-1, 1]
    ])
    final_velocity = np.sqrt(actual_u[-1] ** 2 + actual_vm[-1] ** 2)
    total_time = time_seconds[-1]

    ax.set_xlabel(r'$\mathbf{\mathit{Y\ (m)}}$', fontsize=24, fontweight='bold')
    ax.set_ylabel(r'$\mathbf{\mathit{X\ (m)}}$', fontsize=24, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=8)

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    if berth_info:
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
    else:
        margin_x = (np.max(actual_x) - np.min(actual_x)) * 0.1
        margin_y = (np.max(actual_y) - np.min(actual_y)) * 0.1
        ax.set_xlim([np.min(actual_x) - margin_x, np.max(actual_x) + margin_x])
        ax.set_ylim([np.min(actual_y) - margin_y, np.max(actual_y) + margin_y])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'trajectory_ep{episode}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ========== 2. Cross-track Error 跟踪误差对比图 ==========
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    phase_boundaries_trajectory = []
    if env and hasattr(env, 'trajectory'):
        total_arc = env.trajectory.arc_length
        boundaries = [0.15, 0.65, 0.80, 0.90, 0.95, 1.0]

        for boundary in boundaries:
            for i in range(len(actual_trajectory)):
                point = np.array([actual_x[i], actual_y[i]])
                distances_to_ref = np.linalg.norm(reference_trajectory - point, axis=1)
                nearest_idx = np.argmin(distances_to_ref)
                if hasattr(env.trajectory, 's_arr') and nearest_idx < len(env.trajectory.s_arr):
                    progress = env.trajectory.s_arr[nearest_idx] / total_arc
                else:
                    progress = nearest_idx / len(reference_trajectory)
                if progress >= boundary:
                    phase_boundaries_trajectory.append(i)
                    break
            else:
                phase_boundaries_trajectory.append(len(actual_trajectory) - 1)

    axes[0].plot(time_steps, deviations, color=COLORS['primary'], linewidth=2.5,
                 label='Cross-track Error')

    if len(deviations) > 50:
        window = 50
        moving_avg = np.convolve(deviations, np.ones(window) / window, mode='valid')
        axes[0].plot(range(window - 1, len(deviations)), moving_avg,
                     color=COLORS['quaternary'], linewidth=2.5, linestyle='--',
                     label=f'Moving Average ({window} steps)', alpha=0.9)

    if env:
        position_tol = env.position_tolerance if hasattr(env, 'position_tolerance') else 0.2
        axes[0].axhline(y=position_tol, color=COLORS['tertiary'], linestyle=':',
                        linewidth=2.5, alpha=0.8, label=f'Position Tolerance ({position_tol}m)')

    phase_colors = ['#90EE90', '#87CEEB', '#FFD700', '#FFA500', '#FF6347', '#FF4500']
    phase_names = ['Accel', 'Cruise', 'Decel-1', 'Decel-2', 'Decel-3', 'Berth']

    if len(phase_boundaries_trajectory) >= 6:
        last_step = 0
        for j, step_idx in enumerate(phase_boundaries_trajectory):
            axes[0].axvspan(last_step, step_idx, alpha=0.15, color=phase_colors[j])
            mid_step = (last_step + step_idx) // 2
            y_pos = axes[0].get_ylim()[1] * 0.95
            axes[0].text(mid_step, y_pos, phase_names[j], ha='center', va='top',
                         fontsize=16, fontweight='bold', color=phase_colors[j],
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
            last_step = step_idx

    axes[0].set_ylabel(r'$\mathbf{\mathit{Cross\text{-}track\ Error\ (m)}}$', fontsize=22, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=16, framealpha=0.95, edgecolor='black')
    axes[0].grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    axes[0].tick_params(axis='both', which='major', labelsize=18, width=1.5, length=8)
    axes[0].set_ylim(bottom=0)

    for spine in axes[0].spines.values():
        spine.set_linewidth(1.6)

    error_change = np.gradient(deviations)
    error_change_smooth = np.convolve(error_change, np.ones(20) / 20, mode='same')

    axes[1].plot(time_steps, error_change_smooth, color=COLORS['purple'], linewidth=2.0,
                 label='Error Rate of Change')
    axes[1].fill_between(time_steps, 0, error_change_smooth,
                         where=(error_change_smooth >= 0), alpha=0.3,
                         color=COLORS['quaternary'], label='Deviation Increasing')
    axes[1].fill_between(time_steps, 0, error_change_smooth,
                         where=(error_change_smooth < 0), alpha=0.3,
                         color=COLORS['tertiary'], label='Deviation Decreasing')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

    rmse_cte = np.sqrt(np.mean(np.array(deviations) ** 2))
    axes[1].text(0.98, 0.02,
                 f'CTE RMSE: {rmse_cte:.4f} m\nAvg CTE: {avg_deviation:.4f} m\nMax CTE: {max_deviation:.4f} m',
                 transform=axes[1].transAxes, va='bottom', ha='right', fontsize=15, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black'))

    axes[1].set_xlabel(r'$\mathbf{\mathit{Step}}$', fontsize=22, fontweight='bold')
    axes[1].set_ylabel(r'$\mathbf{\mathit{Error\ Rate\ (m/step)}}$', fontsize=22, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=16, framealpha=0.95, edgecolor='black')
    axes[1].grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    axes[1].tick_params(axis='both', which='major', labelsize=18, width=1.5, length=8)

    for spine in axes[1].spines.values():
        spine.set_linewidth(1.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'trajectory_tracking_ep{episode}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ========== 3. Velocity Profile ==========
    fig, ax = plt.subplots(figsize=(12, 8))

    target_velocities = []
    if env and hasattr(env, 'trajectory'):
        total_arc = env.trajectory.arc_length
        for i in range(len(actual_trajectory)):
            point = np.array([actual_x[i], actual_y[i]])
            distances = np.linalg.norm(reference_trajectory - point, axis=1)
            nearest_idx = np.argmin(distances)

            if hasattr(env.trajectory, 's_arr') and nearest_idx < len(env.trajectory.s_arr):
                current_s = env.trajectory.s_arr[nearest_idx]
            else:
                current_s = (nearest_idx / len(reference_trajectory)) * total_arc

            if hasattr(env.trajectory, 'get_ref_with_velocity'):
                _, _, _, target_vel = env.trajectory.get_ref_with_velocity(current_s)
            elif hasattr(env, '_get_target_velocity'):
                berth_pos = env.trajectory.points[-1]
                dist_to_goal = np.linalg.norm([actual_x[i] - berth_pos[0], actual_y[i] - berth_pos[1]])
                target_vel = env._get_target_velocity(dist_to_goal)
            else:
                u0 = env.initial_velocity if hasattr(env, 'initial_velocity') else 1.5
                progress = current_s / total_arc
                if progress < 0.4:
                    target_vel = u0
                else:
                    target_vel = u0 * (1.0 - (progress - 0.4) / 0.6)
                    target_vel = max(0, target_vel)

            target_velocities.append(target_vel)
        target_velocities = np.array(target_velocities)

    ax.plot(time_steps, actual_u, color=COLORS['tertiary'], linewidth=2.5, label='Surge Speed (u)')
    ax.plot(time_steps, actual_v, color=COLORS['quaternary'], linewidth=2.5, label='Sway Speed (v)')

    if len(target_velocities) > 0:
        ax.plot(time_steps, target_velocities, color=COLORS['purple'], linestyle='--',
                linewidth=3.0, label='Prescribed Surge Speed', alpha=0.9)

    ax.set_xlabel(r'$\mathbf{\mathit{Step}}$', fontsize=24, fontweight='bold')
    ax.set_ylabel(r'$\mathbf{\mathit{Speed\ (m/s)}}$', fontsize=24, fontweight='bold')
    ax.legend(loc='upper right', fontsize=15, framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=8)

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'velocity_profile_ep{episode}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ========== 4. Heading ==========
    fig, ax = plt.subplots(figsize=(12, 8))

    ref_psi = []
    for i in range(len(actual_trajectory)):
        point = np.array([actual_x[i], actual_y[i]])
        distances = np.linalg.norm(reference_trajectory - point, axis=1)
        nearest_idx = np.argmin(distances)
        if nearest_idx < len(reference_trajectory) - 1:
            dx = reference_trajectory[nearest_idx + 1, 0] - reference_trajectory[nearest_idx, 0]
            dy = reference_trajectory[nearest_idx + 1, 1] - reference_trajectory[nearest_idx, 1]
            ref_psi.append(np.arctan2(dy, dx))
        else:
            ref_psi.append(ref_psi[-1] if ref_psi else 0)
    ref_psi = np.array(ref_psi)

    ax.plot(time_steps, np.rad2deg(actual_psi), color=COLORS['quaternary'],
            linewidth=2.5, label='Actual Heading')
    ax.plot(time_steps, np.rad2deg(ref_psi), color=COLORS['primary'],
            linestyle='--', linewidth=2.5, label='Reference Heading')

    ax.set_xlabel(r'$\mathbf{\mathit{Step}}$', fontsize=24, fontweight='bold')
    ax.set_ylabel(r'$\mathbf{\mathit{Heading\ (degrees)}}$', fontsize=24, fontweight='bold')
    ax.legend(loc='best', fontsize=16, framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=8)

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'heading_ep{episode}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ========== 5. Yaw Rate ==========
    fig, ax = plt.subplots(figsize=(12, 8))
    yaw_rate_deg = np.rad2deg(actual_r)
    ax.plot(time_steps, yaw_rate_deg, color=COLORS['purple'], linewidth=2.5, label='Actual Yaw Rate')
    
    # 计算参考偏航角速度
    ref_yaw_rates = []
    if env and hasattr(env, 'trajectory') and hasattr(env.trajectory, 'get_target_yaw_rate'):
        total_arc = env.trajectory.arc_length
        for i in range(len(actual_trajectory)):
            point = np.array([actual_x[i], actual_y[i]])
            distances = np.linalg.norm(reference_trajectory - point, axis=1)
            nearest_idx = np.argmin(distances)
            
            if hasattr(env.trajectory, 's_arr') and nearest_idx < len(env.trajectory.s_arr):
                current_s = env.trajectory.s_arr[nearest_idx]
            else:
                current_s = (nearest_idx / len(reference_trajectory)) * total_arc
            
            ref_r = env.trajectory.get_target_yaw_rate(current_s)
            ref_yaw_rates.append(np.rad2deg(ref_r))
        
        ref_yaw_rates = np.array(ref_yaw_rates)
        ax.plot(time_steps, ref_yaw_rates, color=COLORS['primary'], linestyle='--', 
                linewidth=2.5, label='Reference Yaw Rate (Planned)', alpha=0.9)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    ax.set_xlabel(r'$\mathbf{\mathit{Step}}$', fontsize=24, fontweight='bold')
    ax.set_ylabel(r'$\mathbf{\mathit{r\ (deg/s)}}$', fontsize=24, fontweight='bold')
    ax.legend(loc='best', fontsize=16, framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=8)

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'yaw_rate_ep{episode}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ========== 6-8. 螺旋桨控制 ==========
    if env and hasattr(env, 'control_history') and len(env.control_history) > 0:
        control_hist = np.array(env.control_history)
        control_time = time_steps[:len(control_hist)]

        # 6. 左舷螺旋桨
        fig, ax = plt.subplots(figsize=(12, 8))
        n_port_rpm = control_hist[:, 0] * 60 / (2 * np.pi)
        ax.plot(control_time, n_port_rpm, color=COLORS['primary'], linewidth=2.5)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

        ax.set_xlabel(r'$\mathbf{\mathit{Step}}$', fontsize=24, fontweight='bold')
        ax.set_ylabel(r'$\mathbf{\mathit{n1\ (RPM)}}$', fontsize=24, fontweight='bold')
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=8)

        for spine in ax.spines.values():
            spine.set_linewidth(1.6)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'port_propeller_ep{episode}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 7. 右舷螺旋桨
        fig, ax = plt.subplots(figsize=(12, 8))
        n_starboard_rpm = control_hist[:, 1] * 60 / (2 * np.pi)
        ax.plot(control_time, n_starboard_rpm, color=COLORS['secondary'], linewidth=2.5)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

        ax.set_xlabel(r'$\mathbf{\mathit{Step}}$', fontsize=24, fontweight='bold')
        ax.set_ylabel(r'$\mathbf{\mathit{n2\ (RPM)}}$', fontsize=24, fontweight='bold')
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=8)

        for spine in ax.spines.values():
            spine.set_linewidth(1.6)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'starboard_propeller_ep{episode}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 8. 双螺旋桨对比
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(control_time, n_port_rpm, color=COLORS['primary'], linewidth=2.5,
                label='Port (Left)', alpha=0.9)
        ax.plot(control_time, n_starboard_rpm, color=COLORS['secondary'], linewidth=2.5,
                label='Starboard (Right)', alpha=0.9)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

        ax.set_xlabel(r'$\mathbf{\mathit{Step}}$', fontsize=24, fontweight='bold')
        ax.set_ylabel(r'$\mathbf{\mathit{Propeller\ Speed\ (RPM)}}$', fontsize=24, fontweight='bold')
        ax.legend(loc='best', fontsize=16, framealpha=0.95, edgecolor='black')
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=8)

        for spine in ax.spines.values():
            spine.set_linewidth(1.6)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'dual_propeller_comparison_ep{episode}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    logger.info(f"Trajectory analysis plots saved to {save_dir}")

    return {
        'max_deviation': max_deviation,
        'avg_deviation': avg_deviation,
        'final_error': final_position_error,
        'final_velocity': final_velocity,
        'final_yaw_rate': abs(actual_r[-1]),
        'trajectory_length': len(actual_trajectory)
    }


def get_berth_info(env):
    """获取泊位信息"""
    try:
        berth_corners = env.trajectory.get_target_corners()
        berth_pos = env.trajectory.target_pos
        return {'corners': berth_corners, 'position': berth_pos}
    except Exception as e:
        logger.warning(f"Failed to get berth info: {e}")
        return None


# ============== 评估函数 ==============
def evaluate_agent(agent, env, num_episodes=5, return_trajectory=False):
    """评估智能体"""
    test_env = OtterBerthingTrackingEnv(
        render_mode=None,
        verbose=False,
        curriculum_level=env.curriculum_level if hasattr(env, 'curriculum_level') else 1,
        initial_velocity=env.initial_velocity if hasattr(env, 'initial_velocity') else 1.5,
        use_paper_velocity=env.use_paper_velocity if hasattr(env, 'use_paper_velocity') else True,
        # 干扰参数
        enable_current=env.enable_current if hasattr(env, 'enable_current') else False,
        V_c_max=env.V_c_max if hasattr(env, 'V_c_max') else 0.3,
        current_mode=env.current_mode if hasattr(env, 'current_mode') else 'curriculum',
        enable_wind=env.enable_wind if hasattr(env, 'enable_wind') else False,
        V_w_max=env.V_w_max if hasattr(env, 'V_w_max') else 5.0,
        wind_mode=env.wind_mode if hasattr(env, 'wind_mode') else 'curriculum',
    )

    total_rewards = []
    success_count = 0
    best_trajectory = None
    best_reward = -float('inf')

    for i in range(num_episodes):
        state = test_env.reset()
        episode_reward = 0
        done = False
        trajectory = []

        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, info = test_env.step(action)

            if return_trajectory:
                ship_state = test_env.ship.state.copy()
                trajectory_point = {
                    'x': ship_state['x'],
                    'y': ship_state['y'],
                    'psi': ship_state['psi'],
                    'u': ship_state['u'],
                    'r': ship_state['r']
                }
                if 'vm' in ship_state:
                    trajectory_point['vm'] = ship_state['vm']
                    xG = test_env.ship.xG if hasattr(test_env.ship, 'xG') else 0.2
                    trajectory_point['v'] = ship_state['vm'] + xG * ship_state['r']
                else:
                    trajectory_point['v'] = ship_state['v']
                trajectory.append(trajectory_point)

            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)
        if info.get('success', False):
            success_count += 1

        if return_trajectory and episode_reward > best_reward:
            best_reward = episode_reward
            best_trajectory = trajectory

        logger.info(f"Eval Episode {i + 1}: Reward = {episode_reward:.2f}, "
                    f"Success = {info.get('success', False)}, "
                    f"Final Velocity = {info.get('resultant_velocity', 0):.3f} m/s, "
                    f"CTE = {info.get('cross_track_error', 0):.2f}m")

    avg_reward = np.mean(total_rewards)
    success_rate = success_count / num_episodes

    if return_trajectory:
        return avg_reward, success_rate, best_trajectory, test_env
    else:
        return avg_reward, success_rate


# ============== 主训练函数 ==============
def main():
    parser = argparse.ArgumentParser(description='Enhanced PER-SAC Otter USV Berthing Training')
    parser.add_argument('--episodes', type=int, default=2000, help='训练回合数')
    parser.add_argument('--test_interval', type=int, default=50, help='测试间隔')
    parser.add_argument('--save_interval', type=int, default=100, help='保存间隔')
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    default_output_dir = os.path.join(root_dir, 'assets', 'results', 'results_enhanced')
    default_checkpoint_dir = os.path.join(root_dir, 'assets', 'checkpoints', 'checkpoints_enhanced')

    parser.add_argument('--output_dir', type=str, default=default_output_dir, help='输出目录')
    parser.add_argument('--checkpoint_dir', type=str, default=default_checkpoint_dir, help='检查点目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--render', action='store_true', help='是否渲染')
    parser.add_argument('--no_trajectory_plot', action='store_true', help='禁用轨迹对比图')
    parser.add_argument('--warmup_episodes', type=int, default=200, help='预热回合数')
    parser.add_argument('--log_interval', type=int, default=100, help='训练日志输出间隔')
    parser.add_argument('--no_verbose', action='store_true', help='禁用详细日志输出')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='加载检查点路径')
    # 速度规划参数
    parser.add_argument('--initial_velocity', type=float, default=1.5, help='初始速度 (m/s)')
    parser.add_argument('--use_paper_velocity', action='store_true', default=True, help='使用论文的速度规划方法')
    parser.add_argument('--no_paper_velocity', action='store_false', dest='use_paper_velocity',
                        help='使用自定义分段速度规划')
    # 新增：课程学习参数
    parser.add_argument('--initial_curriculum_level', type=int, default=1, help='初始课程等级 (1-4)')
    parser.add_argument('--curriculum_patience', type=int, default=200, help='课程升级前需要稳定的episode数')
    
    # 新增：路径参数 (berthing_sim_smoothest.py)
    parser.add_argument('--kc', type=float, default=0.45, help='直线段斜率')
    parser.add_argument('--DockingLine_x', type=float, default=70.0, help='泊位x坐标')
    parser.add_argument('--DockingLine_y', type=float, default=95.0, help='泊位y坐标')
    parser.add_argument('--turnpoint', type=float, default=35.0, help='转向点位置')
    parser.add_argument('--safe_distance', type=float, default=60.0, help='Tanh安全距离')
    parser.add_argument('--target_ratio', type=float, default=0.95, help='Tanh目标比例')

    # 新增：环境干扰参数
    parser.add_argument('--enable_current', action='store_true', default=True, help='启用海流干扰')
    parser.add_argument('--no_current', action='store_false', dest='enable_current', help='禁用海流干扰')
    parser.add_argument('--V_c_max', type=float, default=0.3, help='海流速度上限 (m/s)')
    parser.add_argument('--current_mode', type=str, default='curriculum', choices=['random', 'fixed', 'curriculum'],
                        help='海流干扰模式')
    parser.add_argument('--enable_wind', action='store_true', default=True, help='启用风干扰')
    parser.add_argument('--no_wind', action='store_false', dest='enable_wind', help='禁用风干扰')
    parser.add_argument('--V_w_max', type=float, default=5.0, help='风速上限 (m/s)')
    parser.add_argument('--wind_mode', type=str, default='curriculum', choices=['random', 'fixed', 'curriculum'],
                        help='风干扰模式')

    args = parser.parse_args()

    np.random.seed(args.seed)
    T.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 创建环境（使用课程学习）- 添加路径参数和干扰参数
    env = OtterBerthingTrackingEnv(
        render_mode='human' if args.render else None,
        log_interval=args.log_interval,
        verbose=not args.no_verbose,
        curriculum_level=args.initial_curriculum_level,
        initial_velocity=args.initial_velocity,
        use_paper_velocity=args.use_paper_velocity,
        # 路径参数 (berthing_sim_smoothest.py)
        kc=args.kc,
        DockingLine_x=args.DockingLine_x,
        DockingLine_y=args.DockingLine_y,
        turnpoint=args.turnpoint,
        safe_distance=args.safe_distance,
        target_ratio=args.target_ratio,
        # 干扰参数
        enable_current=args.enable_current,
        V_c_max=args.V_c_max,
        current_mode=args.current_mode,
        enable_wind=args.enable_wind,
        V_w_max=args.V_w_max,
        wind_mode=args.wind_mode,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Enhanced PER-SAC Otter USV Training Configuration")
    logger.info(f"{'=' * 70}")
    logger.info(f"Ship: Otter USV (L={env.ship_length:.2f}m, B={env.ship_beam:.2f}m)")
    logger.info(f"Control: Dual Propeller (No Rudder)")
    logger.info(f"Initial Velocity: {args.initial_velocity} m/s")
    logger.info(f"Velocity Planning: Tanh Continuous (safe_dist={args.safe_distance}m, ratio={args.target_ratio})")
    logger.info(f"Trajectory Type: Exponential Convergence (berthing_sim_smoothest.py)")
    logger.info(f"Path Parameters: kc={args.kc}, turnpoint={args.turnpoint}")
    logger.info(f"Docking Target: ({args.DockingLine_x}, {args.DockingLine_y}) m")
    logger.info(f"Trajectory Arc Length: {env.trajectory.arc_length:.1f}m")
    logger.info(f"State dimension: {state_dim}")
    logger.info(f"Action dimension: {action_dim}")
    logger.info(f"Initial Curriculum Level: {args.initial_curriculum_level}")
    logger.info(f"Max episodes: {args.episodes}")
    logger.info(f"{'=' * 70}")
    logger.info(f"Disturbance Configuration:")
    logger.info(f"  Ocean Current: {'Enabled' if args.enable_current else 'Disabled'} "
                f"(V_c_max={args.V_c_max}m/s, mode={args.current_mode})")
    logger.info(f"  Wind: {'Enabled' if args.enable_wind else 'Disabled'} "
                f"(V_w_max={args.V_w_max}m/s, mode={args.wind_mode})")
    logger.info(f"  Device: {device}")
    logger.info(f"{'=' * 70}\n")

    # 创建增强版Agent
    agent = EnhancedPerSacAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=1e-5,
        critic_lr=1e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        fc1_dim=256,
        fc2_dim=256,
        ckpt_dir=args.checkpoint_dir,
        total_episodes=args.episodes,
        warmup_episodes=args.warmup_episodes
    )

    if args.load_checkpoint:
        try:
            agent.load_models(args.load_checkpoint)
            logger.info(f"Loaded checkpoint: {args.load_checkpoint}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

    # 课程学习管理器（使用初始课程等级）
    curriculum = CurriculumManager(
        level_thresholds=[0.5, 0.6, 0.7, 0.8],
        window_size=100,
        patience=args.curriculum_patience,
        initial_level=args.initial_curriculum_level  # 同步初始级别
    )

    monitor = TrainingMonitor(window_size=100)

    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    logger.info("Starting Enhanced PER-SAC Otter USV training with Curriculum Learning...")
    start_time = time.time()
    best_success_rate = 0
    beta = 0.4
    consecutive_decline = 0
    last_success_rate = 0

    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            if episode < args.warmup_episodes:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, evaluate=False)

            next_state, reward, done, info = env.step(action)

            # 标记是否来自成功episode（当前还不知道，先设为False）
            agent.store_transition(state, action, reward / 10.0, next_state, done, is_success=False)

            if episode >= args.warmup_episodes:
                # 动态调整成功采样比例
                success_sample_ratio = 0.2 + 0.1 * consecutive_decline
                success_sample_ratio = min(success_sample_ratio, 0.4)

                agent.update(beta, success_sample_ratio)
                beta = min(1.0, beta + 0.0001)

            episode_reward += reward
            episode_length += 1
            state = next_state

            if args.render and episode % 50 == 0:
                env.render()

        # Episode结束
        success = info.get('success', False)
        agent.end_episode(episode_reward, success)

        # 更新课程学习
        new_level = curriculum.update(success)
        if new_level is not None:
            env.set_curriculum_level(new_level)
            logger.info(f"[Curriculum] Environment updated to level {new_level}")

        # 更新学习率调度
        if episode >= args.warmup_episodes:
            current_lr = agent.step_schedulers(episode)

        # 获取当前成功率
        current_success_rate = curriculum.get_success_rate()

        # 检查成功率下降
        if current_success_rate < last_success_rate - 0.1 and episode >= args.warmup_episodes:
            consecutive_decline += 1
            if consecutive_decline >= 3 and len(agent.success_trajectory_buffer) > 0:
                agent.replay_success_trajectories(n_trajectories=2)
                consecutive_decline = 0
        else:
            consecutive_decline = max(0, consecutive_decline - 1)

        # 更新目标熵
        if episode >= args.warmup_episodes:
            agent.update_target_entropy(current_success_rate)

        last_success_rate = current_success_rate

        # 定期衰减优先级
        if episode % 100 == 0 and episode > 0:
            agent.memory.decay_priorities(decay_factor=0.999)

        # 记录到监控器
        reward_history = env.reward_history[-1] if len(env.reward_history) > 0 else None
        monitor.update_episode(episode_reward, episode_length, success, info,
                               reward_history, curriculum.get_current_level())

        # 日志输出
        if (episode + 1) % 10 == 0:
            stats = monitor.get_recent_stats()
            training_stats = agent.training_stats
            buffer_stats = agent.memory.get_stats()

            logger.info(
                f"Episode {episode + 1}: "
                f"Reward = {episode_reward:.2f}, "
                f"Avg Reward = {stats.get('avg_reward', 0):.2f}, "
                f"Success Rate = {current_success_rate:.2%}, "
                f"Curriculum = {curriculum.get_current_level()}, "
                f"Alpha = {np.mean(training_stats['alpha']) if len(training_stats['alpha']) > 0 else 0:.3f}, "
                f"Success Buffer = {buffer_stats['success_buffer_size']}"
            )

        # 测试评估
        if (episode + 1) % args.test_interval == 0:
            logger.info(f"Testing at episode {episode + 1}...")
            test_reward, test_success_rate = evaluate_agent(agent, env, num_episodes=10)
            monitor.update_test(episode + 1, test_reward, test_success_rate)

            logger.info(f"Test Results: Avg Reward = {test_reward:.2f}, "
                        f"Success Rate = {test_success_rate:.2%}")

            if test_success_rate > best_success_rate:
                best_success_rate = test_success_rate
                agent.save_models('best')
                logger.info(f"New best model saved with success rate: {best_success_rate:.2%}")

        # 保存检查点和绘图
        if (episode + 1) % args.save_interval == 0:
            agent.save_models(episode + 1)
            monitor.save_data(os.path.join(args.output_dir, f'training_data.json'))

            progress_plots_dir = os.path.join(plots_dir, 'progress')
            monitor.plot_training_progress_separate(progress_plots_dir, episode + 1)

        # 轨迹可视化
        if (episode + 1) % 100 == 0 and not args.no_trajectory_plot:
            logger.info(f"Plotting trajectory at episode {episode + 1}...")

            result = evaluate_agent(agent, env, num_episodes=1, return_trajectory=True)
            _, _, best_trajectory, test_env = result

            if best_trajectory and len(best_trajectory) > 0:
                reference_trajectory = test_env.trajectory.points
                berth_info = get_berth_info(test_env)

                trajectory_plots_dir = os.path.join(plots_dir, f'trajectory_ep{episode + 1}')
                stats = plot_trajectory_separate(
                    best_trajectory,
                    reference_trajectory,
                    berth_info,
                    episode + 1,
                    trajectory_plots_dir,
                    test_env
                )
                logger.info(f"Trajectory stats: {stats}")

    # 训练结束
    training_time = time.time() - start_time
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Training completed in {training_time:.2f} seconds ({training_time / 3600:.2f} hours)")
    logger.info(f"Best success rate: {best_success_rate:.2%}")
    logger.info(f"Final curriculum level: {curriculum.get_current_level()}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"{'=' * 70}")

    # 最终评估
    logger.info("\nFinal evaluation...")
    final_reward, final_success = evaluate_agent(agent, env, num_episodes=20)
    logger.info(f"Final performance: Avg Reward = {final_reward:.2f}, Success Rate = {final_success:.2%}")


if __name__ == '__main__':
    main()
