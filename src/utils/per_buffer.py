"""
优先经验回放缓冲区 - 增强版

修复内容：
1. 成功经验优先级提升 - 防止遗忘好策略
2. 成功轨迹单独存储区 - 保证成功样本比例
3. 优先级衰减机制 - 防止某些样本永远不被采样
4. 动态alpha调整 - 根据训练阶段调整优先级敏感度
"""

import numpy as np
from collections import deque
import torch


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区 - 增强版"""

    def __init__(self, max_size, state_dim, action_dim, batch_size,
                 alpha=0.6,
                 success_priority_boost=2.0,  # 成功经验优先级提升倍数
                 success_buffer_ratio=0.1):  # 成功经验缓冲区占比

        self.mem_size = max_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.initial_alpha = alpha
        self.mem_cnt = 0

        # ===== 新增：成功经验相关参数 =====
        self.success_priority_boost = success_priority_boost
        self.success_buffer_ratio = success_buffer_ratio
        self.success_buffer_size = int(max_size * success_buffer_ratio)

        # 主缓冲区
        self.state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.next_state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=bool)
        self.priorities = np.zeros((max_size,), dtype=np.float32)
        self.priorities[0] = 1.0

        # ===== 新增：成功经验单独存储区 =====
        self.success_state_memory = np.zeros((self.success_buffer_size, state_dim), dtype=np.float32)
        self.success_action_memory = np.zeros((self.success_buffer_size, action_dim), dtype=np.float32)
        self.success_reward_memory = np.zeros(self.success_buffer_size, dtype=np.float32)
        self.success_next_state_memory = np.zeros((self.success_buffer_size, state_dim), dtype=np.float32)
        self.success_terminal_memory = np.zeros(self.success_buffer_size, dtype=bool)
        self.success_priorities = np.zeros((self.success_buffer_size,), dtype=np.float32)
        self.success_mem_cnt = 0

        # ===== 新增：标记是否为成功样本 =====
        self.is_success_sample = np.zeros(max_size, dtype=bool)

        # 统计信息
        self.total_success_stored = 0
        self.total_stored = 0

    def store_transition(self, state, action, reward, state_, done, is_success=False):
        """
        存储经验

        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            state_: 下一状态
            done: 是否结束
            is_success: 是否是成功轨迹的一部分
        """
        idx = self.mem_cnt % self.mem_size

        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.next_state_memory[idx] = state_
        self.terminal_memory[idx] = done
        self.is_success_sample[idx] = is_success

        # 设置优先级
        if self.mem_cnt == 0:
            base_priority = 1.0
        else:
            base_priority = self.priorities[:min(self.mem_cnt, self.mem_size)].max()

        # ===== 关键：成功经验优先级提升 =====
        if is_success:
            self.priorities[idx] = base_priority * self.success_priority_boost
        else:
            self.priorities[idx] = base_priority

        self.mem_cnt += 1
        self.total_stored += 1

        # ===== 同时存储到成功经验缓冲区 =====
        if is_success:
            self._store_success_transition(state, action, reward, state_, done)

    def _store_success_transition(self, state, action, reward, state_, done):
        """存储成功经验到单独缓冲区"""
        idx = self.success_mem_cnt % self.success_buffer_size

        self.success_state_memory[idx] = state
        self.success_action_memory[idx] = action
        self.success_reward_memory[idx] = reward
        self.success_next_state_memory[idx] = state_
        self.success_terminal_memory[idx] = done

        # 成功经验始终保持较高优先级
        if self.success_mem_cnt == 0:
            self.success_priorities[idx] = 1.0
        else:
            max_mem = min(self.success_mem_cnt, self.success_buffer_size)
            self.success_priorities[idx] = self.success_priorities[:max_mem].max()

        self.success_mem_cnt += 1
        self.total_success_stored += 1

    def sample_buffer(self, beta=0.4, success_sample_ratio=0.2):
        """
        采样经验

        Args:
            beta: 重要性采样指数
            success_sample_ratio: 从成功缓冲区采样的比例

        Returns:
            states, actions, rewards, next_states, terminals, indices, weights
        """
        max_mem = min(self.mem_cnt, self.mem_size)
        max_success_mem = min(self.success_mem_cnt, self.success_buffer_size)

        # 计算从成功缓冲区采样的数量
        if max_success_mem > 0:
            n_success_samples = int(self.batch_size * success_sample_ratio)
            n_success_samples = min(n_success_samples, max_success_mem)
            n_main_samples = self.batch_size - n_success_samples
        else:
            n_success_samples = 0
            n_main_samples = self.batch_size

        # 从主缓冲区采样
        priorities = self.priorities[:max_mem]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        main_indices = np.random.choice(max_mem, n_main_samples, p=probs, replace=False)

        main_states = self.state_memory[main_indices]
        main_actions = self.action_memory[main_indices]
        main_rewards = self.reward_memory[main_indices]
        main_next_states = self.next_state_memory[main_indices]
        main_terminals = self.terminal_memory[main_indices]

        # 主缓冲区的重要性采样权重
        main_weights = (max_mem * probs[main_indices]) ** (-beta)
        main_weights /= main_weights.max()

        # 从成功缓冲区采样
        if n_success_samples > 0:
            success_priorities = self.success_priorities[:max_success_mem]
            success_probs = success_priorities ** self.alpha
            success_probs /= success_probs.sum()

            success_indices = np.random.choice(max_success_mem, n_success_samples,
                                               p=success_probs, replace=False)

            success_states = self.success_state_memory[success_indices]
            success_actions = self.success_action_memory[success_indices]
            success_rewards = self.success_reward_memory[success_indices]
            success_next_states = self.success_next_state_memory[success_indices]
            success_terminals = self.success_terminal_memory[success_indices]

            # 成功缓冲区的权重（稍微提高）
            success_weights = (max_success_mem * success_probs[success_indices]) ** (-beta)
            success_weights /= success_weights.max()

            # 合并
            states = np.concatenate([main_states, success_states], axis=0)
            actions = np.concatenate([main_actions, success_actions], axis=0)
            rewards = np.concatenate([main_rewards, success_rewards], axis=0)
            next_states = np.concatenate([main_next_states, success_next_states], axis=0)
            terminals = np.concatenate([main_terminals, success_terminals], axis=0)
            weights = np.concatenate([main_weights, success_weights], axis=0)

            # 标记索引来源（主缓冲区用正数，成功缓冲区用负数-1开始）
            indices = np.concatenate([main_indices, -(success_indices + 1)], axis=0)
        else:
            states = main_states
            actions = main_actions
            rewards = main_rewards
            next_states = main_next_states
            terminals = main_terminals
            weights = main_weights
            indices = main_indices

        return states, actions, rewards, next_states, terminals, indices, weights

    def update_priorities(self, indices, td_errors):
        """更新优先级"""
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        if torch.is_tensor(td_errors):
            td_errors = td_errors.detach().cpu().numpy()
        priorities = np.abs(td_errors) + 1e-5

        for i, idx in enumerate(indices):
            if idx >= 0:
                # 主缓冲区
                self.priorities[idx] = priorities[i]
            else:
                # 成功缓冲区（索引为负数-1）
                success_idx = -idx - 1
                if success_idx < self.success_buffer_size:
                    # 成功经验保持更高的基础优先级
                    self.success_priorities[success_idx] = max(priorities[i], 0.5)

    def decay_priorities(self, decay_factor=0.999):
        """
        优先级衰减 - 防止某些样本永远不被采样

        定期调用此函数，让旧的高优先级样本逐渐降低优先级
        """
        max_mem = min(self.mem_cnt, self.mem_size)
        self.priorities[:max_mem] *= decay_factor

        # 设置最小优先级
        min_priority = 0.01
        self.priorities[:max_mem] = np.maximum(self.priorities[:max_mem], min_priority)

    def adjust_alpha(self, new_alpha):
        """动态调整alpha值"""
        self.alpha = np.clip(new_alpha, 0.0, 1.0)

    def get_success_ratio(self):
        """获取成功经验占比"""
        if self.total_stored == 0:
            return 0.0
        return self.total_success_stored / self.total_stored

    def get_stats(self):
        """获取缓冲区统计信息"""
        return {
            'total_stored': self.total_stored,
            'total_success_stored': self.total_success_stored,
            'success_ratio': self.get_success_ratio(),
            'main_buffer_size': min(self.mem_cnt, self.mem_size),
            'success_buffer_size': min(self.success_mem_cnt, self.success_buffer_size),
            'current_alpha': self.alpha,
            'avg_priority': self.priorities[:min(self.mem_cnt, self.mem_size)].mean() if self.mem_cnt > 0 else 0,
            'max_priority': self.priorities[:min(self.mem_cnt, self.mem_size)].max() if self.mem_cnt > 0 else 0
        }

    def ready(self):
        """检查是否可以开始采样"""
        return self.mem_cnt >= self.batch_size


class GpuPrioritizedReplayBuffer:
    """GPU 优先经验回放缓冲区"""

    def __init__(self, max_size, state_dim, action_dim, batch_size,
                 alpha=0.6,
                 success_priority_boost=2.0,
                 success_buffer_ratio=0.1,
                 device=None):
        self.mem_size = max_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.initial_alpha = alpha
        self.mem_cnt = 0
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 成功经验相关
        self.success_priority_boost = success_priority_boost
        self.success_buffer_ratio = success_buffer_ratio
        self.success_buffer_size = int(max_size * success_buffer_ratio)
        if self.success_buffer_size <= 0:
            self.success_buffer_size = 1

        # 主缓冲区
        self.state_memory = torch.zeros((max_size, state_dim), device=self.device, dtype=torch.float32)
        self.action_memory = torch.zeros((max_size, action_dim), device=self.device, dtype=torch.float32)
        self.reward_memory = torch.zeros(max_size, device=self.device, dtype=torch.float32)
        self.next_state_memory = torch.zeros((max_size, state_dim), device=self.device, dtype=torch.float32)
        self.terminal_memory = torch.zeros(max_size, device=self.device, dtype=torch.bool)
        self.priorities = torch.zeros((max_size,), device=self.device, dtype=torch.float32)
        self.priorities[0] = 1.0

        # 成功经验缓冲区
        self.success_state_memory = torch.zeros((self.success_buffer_size, state_dim), device=self.device, dtype=torch.float32)
        self.success_action_memory = torch.zeros((self.success_buffer_size, action_dim), device=self.device, dtype=torch.float32)
        self.success_reward_memory = torch.zeros(self.success_buffer_size, device=self.device, dtype=torch.float32)
        self.success_next_state_memory = torch.zeros((self.success_buffer_size, state_dim), device=self.device, dtype=torch.float32)
        self.success_terminal_memory = torch.zeros(self.success_buffer_size, device=self.device, dtype=torch.bool)
        self.success_priorities = torch.zeros((self.success_buffer_size,), device=self.device, dtype=torch.float32)
        self.success_mem_cnt = 0

        # 标记成功样本
        self.is_success_sample = torch.zeros(max_size, device=self.device, dtype=torch.bool)

        # 统计信息
        self.total_success_stored = 0
        self.total_stored = 0

    def _to_tensor(self, array, dtype=torch.float32):
        if torch.is_tensor(array):
            tensor = array.to(self.device)
        else:
            tensor = torch.as_tensor(array, device=self.device)
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor

    def store_transition(self, state, action, reward, state_, done, is_success=False):
        """存储经验"""
        state_t = self._to_tensor(state, torch.float32)
        action_t = self._to_tensor(action, torch.float32)
        reward_t = self._to_tensor(reward, torch.float32)
        next_state_t = self._to_tensor(state_, torch.float32)
        done_t = self._to_tensor(done, torch.bool)

        idx = self.mem_cnt % self.mem_size
        self.state_memory[idx] = state_t
        self.action_memory[idx] = action_t
        self.reward_memory[idx] = reward_t
        self.next_state_memory[idx] = next_state_t
        self.terminal_memory[idx] = done_t
        self.is_success_sample[idx] = bool(is_success)

        if self.mem_cnt == 0:
            base_priority = 1.0
        else:
            max_mem = min(self.mem_cnt, self.mem_size)
            base_priority = float(self.priorities[:max_mem].max().item())

        if is_success:
            self.priorities[idx] = base_priority * self.success_priority_boost
        else:
            self.priorities[idx] = base_priority

        self.mem_cnt += 1
        self.total_stored += 1

        if is_success:
            self._store_success_transition(state_t, action_t, reward_t, next_state_t, done_t)

    def store_transitions_batch(self, states, actions, rewards, next_states, dones, is_success=False):
        """批量存储经验"""
        states_t = self._to_tensor(states, torch.float32)
        actions_t = self._to_tensor(actions, torch.float32)
        rewards_t = self._to_tensor(rewards, torch.float32)
        next_states_t = self._to_tensor(next_states, torch.float32)
        dones_t = self._to_tensor(dones, torch.bool)

        batch = states_t.shape[0]
        indices = (self.mem_cnt + torch.arange(batch, device=self.device)) % self.mem_size

        self.state_memory[indices] = states_t
        self.action_memory[indices] = actions_t
        self.reward_memory[indices] = rewards_t
        self.next_state_memory[indices] = next_states_t
        self.terminal_memory[indices] = dones_t
        self.is_success_sample[indices] = bool(is_success)

        if self.mem_cnt == 0:
            base_priority = 1.0
        else:
            max_mem = min(self.mem_cnt, self.mem_size)
            base_priority = float(self.priorities[:max_mem].max().item())

        if is_success:
            self.priorities[indices] = base_priority * self.success_priority_boost
            for i in range(batch):
                self._store_success_transition(
                    states_t[i], actions_t[i], rewards_t[i], next_states_t[i], dones_t[i]
                )
        else:
            self.priorities[indices] = base_priority

        self.mem_cnt += batch
        self.total_stored += batch

    def _store_success_transition(self, state, action, reward, state_, done):
        idx = self.success_mem_cnt % self.success_buffer_size
        self.success_state_memory[idx] = state
        self.success_action_memory[idx] = action
        self.success_reward_memory[idx] = reward
        self.success_next_state_memory[idx] = state_
        self.success_terminal_memory[idx] = done

        if self.success_mem_cnt == 0:
            self.success_priorities[idx] = 1.0
        else:
            max_mem = min(self.success_mem_cnt, self.success_buffer_size)
            self.success_priorities[idx] = self.success_priorities[:max_mem].max()

        self.success_mem_cnt += 1
        self.total_success_stored += 1

    def sample_buffer(self, beta=0.4, success_sample_ratio=0.2):
        max_mem = min(self.mem_cnt, self.mem_size)
        max_success_mem = min(self.success_mem_cnt, self.success_buffer_size)

        if max_success_mem > 0:
            n_success_samples = int(self.batch_size * success_sample_ratio)
            n_success_samples = min(n_success_samples, max_success_mem)
            n_main_samples = self.batch_size - n_success_samples
        else:
            n_success_samples = 0
            n_main_samples = self.batch_size

        priorities = self.priorities[:max_mem]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        main_indices = torch.multinomial(probs, n_main_samples, replacement=False)
        main_states = self.state_memory[main_indices]
        main_actions = self.action_memory[main_indices]
        main_rewards = self.reward_memory[main_indices]
        main_next_states = self.next_state_memory[main_indices]
        main_terminals = self.terminal_memory[main_indices]

        main_weights = (max_mem * probs[main_indices]) ** (-beta)
        main_weights = main_weights / main_weights.max()

        if n_success_samples > 0:
            success_priorities = self.success_priorities[:max_success_mem]
            success_probs = success_priorities ** self.alpha
            success_probs = success_probs / success_probs.sum()

            success_indices = torch.multinomial(success_probs, n_success_samples, replacement=False)
            success_states = self.success_state_memory[success_indices]
            success_actions = self.success_action_memory[success_indices]
            success_rewards = self.success_reward_memory[success_indices]
            success_next_states = self.success_next_state_memory[success_indices]
            success_terminals = self.success_terminal_memory[success_indices]

            success_weights = (max_success_mem * success_probs[success_indices]) ** (-beta)
            success_weights = success_weights / success_weights.max()

            states = torch.cat([main_states, success_states], dim=0)
            actions = torch.cat([main_actions, success_actions], dim=0)
            rewards = torch.cat([main_rewards, success_rewards], dim=0)
            next_states = torch.cat([main_next_states, success_next_states], dim=0)
            terminals = torch.cat([main_terminals, success_terminals], dim=0)
            weights = torch.cat([main_weights, success_weights], dim=0)
            indices = torch.cat([main_indices, -(success_indices + 1)], dim=0)
        else:
            states = main_states
            actions = main_actions
            rewards = main_rewards
            next_states = main_next_states
            terminals = main_terminals
            weights = main_weights
            indices = main_indices

        return states, actions, rewards, next_states, terminals, indices, weights

    def update_priorities(self, indices, td_errors):
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, device=self.device)
        if not torch.is_tensor(td_errors):
            td_errors = torch.as_tensor(td_errors, device=self.device)

        priorities = td_errors.abs() + 1e-5
        main_mask = indices >= 0
        if main_mask.any():
            main_idx = indices[main_mask].long()
            self.priorities[main_idx] = priorities[main_mask]

        success_mask = ~main_mask
        if success_mask.any():
            success_idx = (-indices[success_mask] - 1).long()
            if success_idx.numel() > 0:
                min_priority = torch.tensor(0.5, device=self.device)
                self.success_priorities[success_idx] = torch.maximum(priorities[success_mask], min_priority)

    def decay_priorities(self, decay_factor=0.999):
        max_mem = min(self.mem_cnt, self.mem_size)
        if max_mem <= 0:
            return
        self.priorities[:max_mem] *= decay_factor
        min_priority = 0.01
        self.priorities[:max_mem] = torch.maximum(self.priorities[:max_mem], torch.tensor(min_priority, device=self.device))

    def adjust_alpha(self, new_alpha):
        self.alpha = float(np.clip(new_alpha, 0.0, 1.0))

    def get_success_ratio(self):
        if self.total_stored == 0:
            return 0.0
        return self.total_success_stored / self.total_stored

    def get_stats(self):
        max_mem = min(self.mem_cnt, self.mem_size)
        avg_priority = float(self.priorities[:max_mem].mean().item()) if self.mem_cnt > 0 else 0.0
        max_priority = float(self.priorities[:max_mem].max().item()) if self.mem_cnt > 0 else 0.0
        return {
            'total_stored': self.total_stored,
            'total_success_stored': self.total_success_stored,
            'success_ratio': self.get_success_ratio(),
            'main_buffer_size': max_mem,
            'success_buffer_size': min(self.success_mem_cnt, self.success_buffer_size),
            'current_alpha': self.alpha,
            'avg_priority': avg_priority,
            'max_priority': max_priority
        }

    def ready(self):
        return self.mem_cnt >= self.batch_size


class SuccessTrajectoryBuffer:
    """
    成功轨迹整体存储缓冲区

    用于存储完整的成功episode，而非单个transition
    这有助于在训练不稳定时回放完整的成功策略
    """

    def __init__(self, max_trajectories=100):
        self.max_trajectories = max_trajectories
        self.trajectories = deque(maxlen=max_trajectories)
        self.trajectory_rewards = deque(maxlen=max_trajectories)

    def store_trajectory(self, trajectory, total_reward):
        """
        存储一条完整轨迹

        Args:
            trajectory: list of (state, action, reward, next_state, done)
            total_reward: episode总奖励
        """
        self.trajectories.append(trajectory)
        self.trajectory_rewards.append(total_reward)

    def sample_trajectory(self, n=1, prioritize_by_reward=True):
        """
        采样轨迹

        Args:
            n: 采样数量
            prioritize_by_reward: 是否按奖励优先采样

        Returns:
            list of trajectories
        """
        if len(self.trajectories) == 0:
            return []

        n = min(n, len(self.trajectories))

        if prioritize_by_reward:
            # 按奖励进行加权采样
            rewards = np.array(list(self.trajectory_rewards))
            # 归一化为概率
            probs = rewards - rewards.min() + 1e-6
            probs = probs / probs.sum()

            indices = np.random.choice(len(self.trajectories), n, p=probs, replace=False)
        else:
            indices = np.random.choice(len(self.trajectories), n, replace=False)

        return [list(self.trajectories)[i] for i in indices]

    def get_best_trajectory(self):
        """获取奖励最高的轨迹"""
        if len(self.trajectories) == 0:
            return None

        best_idx = np.argmax(list(self.trajectory_rewards))
        return list(self.trajectories)[best_idx]

    def __len__(self):
        return len(self.trajectories)

    def get_stats(self):
        """获取统计信息"""
        if len(self.trajectory_rewards) == 0:
            return {'count': 0, 'avg_reward': 0, 'max_reward': 0, 'min_reward': 0}

        rewards = list(self.trajectory_rewards)
        return {
            'count': len(rewards),
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards)
        }
