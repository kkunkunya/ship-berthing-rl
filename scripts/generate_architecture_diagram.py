"""Generate system architecture diagram for SAC ship berthing system"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
color_env = '#E8F4F8'
color_agent = '#FFF4E6'
color_training = '#F0F8E8'
color_arrow = '#4A90E2'

# Title
ax.text(7, 9.5, 'SAC-PER Ship Berthing System Architecture',
        ha='center', va='top', fontsize=18, fontweight='bold')

# 1. Environment Block
env_box = FancyBboxPatch((0.5, 6), 5, 2.5, boxstyle="round,pad=0.1",
                         edgecolor='#2C5F7C', facecolor=color_env, linewidth=2)
ax.add_patch(env_box)
ax.text(3, 8.2, 'Ship Berthing Environment', ha='center', fontsize=12, fontweight='bold')

# Environment components
ax.text(1, 7.6, '• Otter USV Dynamics', ha='left', fontsize=9)
ax.text(1, 7.3, '• Reference Trajectory', ha='left', fontsize=9)
ax.text(1, 7.0, '• Berth Position (70, 95)', ha='left', fontsize=9)
ax.text(1, 6.6, '• Observation: 30D State', ha='left', fontsize=9)
ax.text(1, 6.3, '  (pos, vel, heading, disturbances)', ha='left', fontsize=8, style='italic')

# Disturbance box
dist_box = Rectangle((3.8, 6.3), 1.5, 1.8, edgecolor='#E74C3C',
                     facecolor='#FADBD8', linewidth=1.5, linestyle='--')
ax.add_patch(dist_box)
ax.text(4.55, 7.9, 'Disturbances', ha='center', fontsize=9, fontweight='bold')
ax.text(4.55, 7.5, 'Current:', ha='center', fontsize=8)
ax.text(4.55, 7.2, '0.075 m/s', ha='center', fontsize=8)
ax.text(4.55, 6.9, 'Wind:', ha='center', fontsize=8)
ax.text(4.55, 6.6, '1.2 m/s', ha='center', fontsize=8)

# 2. SAC Agent Block
agent_box = FancyBboxPatch((6.5, 6), 6.5, 2.5, boxstyle="round,pad=0.1",
                          edgecolor='#E67E22', facecolor=color_agent, linewidth=2)
ax.add_patch(agent_box)
ax.text(9.75, 8.2, 'SAC Agent with PER', ha='center', fontsize=12, fontweight='bold')

# Actor Network
actor_box = Rectangle((7, 6.8), 2.2, 1.2, edgecolor='#27AE60',
                      facecolor='#D5F4E6', linewidth=1.5)
ax.add_patch(actor_box)
ax.text(8.1, 7.7, 'Actor Network', ha='center', fontsize=9, fontweight='bold')
ax.text(8.1, 7.4, '30D → 256 → 256', ha='center', fontsize=8)
ax.text(8.1, 7.1, '→ 2D Action', ha='center', fontsize=8)
ax.text(8.1, 6.9, '(n, δ)', ha='center', fontsize=7, style='italic')

# Critic Network
critic_box = Rectangle((9.5, 6.8), 2.2, 1.2, edgecolor='#8E44AD',
                       facecolor='#EBDEF0', linewidth=1.5)
ax.add_patch(critic_box)
ax.text(10.6, 7.7, 'Critic Network', ha='center', fontsize=9, fontweight='bold')
ax.text(10.6, 7.4, '32D → 256 → 256', ha='center', fontsize=8)
ax.text(10.6, 7.1, '→ Q-value', ha='center', fontsize=8)
ax.text(10.6, 6.9, '(Twin Q)', ha='center', fontsize=7, style='italic')

# Entropy
entropy_box = Rectangle((12, 6.8), 0.8, 1.2, edgecolor='#F39C12',
                        facecolor='#FCF3CF', linewidth=1.5)
ax.add_patch(entropy_box)
ax.text(12.4, 7.5, 'α', ha='center', fontsize=11, fontweight='bold')
ax.text(12.4, 7.1, 'Adaptive', ha='center', fontsize=7)
ax.text(12.4, 6.9, 'Entropy', ha='center', fontsize=7)

# PER Buffer
per_box = Rectangle((7, 6.2), 5.8, 0.45, edgecolor='#3498DB',
                    facecolor='#D6EAF8', linewidth=1.5)
ax.add_patch(per_box)
ax.text(10, 6.4, 'Prioritized Experience Replay (1M transitions)',
        ha='center', fontsize=8, fontweight='bold')

# 3. Curriculum Learning Block
curr_box = FancyBboxPatch((0.5, 3.5), 12.5, 1.8, boxstyle="round,pad=0.1",
                         edgecolor='#16A085', facecolor=color_training, linewidth=2)
ax.add_patch(curr_box)
ax.text(6.75, 5.1, 'Curriculum Learning (4 Levels)', ha='center',
        fontsize=12, fontweight='bold')

# Curriculum levels
levels = ['Level 1\n(Easy)', 'Level 2\n(Medium)', 'Level 3\n(Hard)', 'Level 4\n(Expert)']
level_colors = ['#A9DFBF', '#7DCEA0', '#52BE80', '#27AE60']
for i, (level, color) in enumerate(zip(levels, level_colors)):
    x = 1.5 + i * 2.8
    level_box = Rectangle((x, 3.7), 2.3, 1.3, edgecolor='#1E8449',
                          facecolor=color, linewidth=1.2)
    ax.add_patch(level_box)
    ax.text(x + 1.15, 4.5, level, ha='center', fontsize=8, fontweight='bold')
    ax.text(x + 1.15, 3.95, f'Tolerance: {["0.5m", "0.3m", "0.2m", "0.1m"][i]}',
            ha='center', fontsize=7)

    # Arrow between levels
    if i < 3:
        arrow = FancyArrowPatch((x + 2.4, 4.3), (x + 2.7, 4.3),
                               arrowstyle='->', mutation_scale=15,
                               color='#1E8449', linewidth=1.5)
        ax.add_patch(arrow)

# 4. Training Loop Block
train_box = FancyBboxPatch((0.5, 1), 12.5, 1.8, boxstyle="round,pad=0.1",
                          edgecolor='#C0392B', facecolor='#FADBD8', linewidth=2)
ax.add_patch(train_box)
ax.text(6.75, 2.6, 'Training Configuration', ha='center',
        fontsize=12, fontweight='bold')

# Training details in two columns
train_details_left = [
    '• Algorithm: SAC (Soft Actor-Critic)',
    '• Update-to-Data Ratio: 0.5',
    '• Batch Size: 256',
    '• Learning Rate: 5e-6',
]
train_details_right = [
    '• Replay Buffer: 1M (PER)',
    '• Training Episodes: 400',
    '• Network: 256-256 (2 layers)',
    '• Target Update: τ=0.005',
]

for i, detail in enumerate(train_details_left):
    ax.text(1, 2.3 - i*0.25, detail, ha='left', fontsize=8)

for i, detail in enumerate(train_details_right):
    ax.text(7, 2.3 - i*0.25, detail, ha='left', fontsize=8)

# 5. Main Flow Arrows
# Environment -> Agent (State)
arrow1 = FancyArrowPatch((5.5, 7.2), (6.5, 7.2),
                        arrowstyle='->', mutation_scale=20,
                        color=color_arrow, linewidth=2.5)
ax.add_patch(arrow1)
ax.text(6, 7.5, 'State\n(30D)', ha='center', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Agent -> Environment (Action)
arrow2 = FancyArrowPatch((6.5, 6.8), (5.5, 6.8),
                        arrowstyle='->', mutation_scale=20,
                        color=color_arrow, linewidth=2.5)
ax.add_patch(arrow2)
ax.text(6, 6.5, 'Action\n(2D)', ha='center', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Environment -> Curriculum
arrow3 = FancyArrowPatch((3, 6), (3, 5.3),
                        arrowstyle='->', mutation_scale=20,
                        color='#16A085', linewidth=2)
ax.add_patch(arrow3)
ax.text(3.5, 5.7, 'Performance\nFeedback', ha='center', fontsize=7)

# Curriculum -> Agent
arrow4 = FancyArrowPatch((6.75, 3.5), (9.75, 6),
                        arrowstyle='->', mutation_scale=20,
                        color='#16A085', linewidth=2, linestyle='--')
ax.add_patch(arrow4)
ax.text(8, 4.5, 'Difficulty\nAdjustment', ha='center', fontsize=7)

# Agent -> Training
arrow5 = FancyArrowPatch((9.75, 6), (6.75, 2.8),
                        arrowstyle='->', mutation_scale=20,
                        color='#C0392B', linewidth=2, linestyle='--')
ax.add_patch(arrow5)
ax.text(8.5, 4.2, 'Experience\nSamples', ha='center', fontsize=7)

# 6. Legend
legend_elements = [
    mpatches.Patch(facecolor=color_env, edgecolor='#2C5F7C', label='Environment'),
    mpatches.Patch(facecolor=color_agent, edgecolor='#E67E22', label='Agent'),
    mpatches.Patch(facecolor=color_training, edgecolor='#16A085', label='Training'),
    mpatches.FancyArrow(0, 0, 1, 0, width=0.3, color=color_arrow, label='Data Flow'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9)

# Footer
ax.text(7, 0.3, 'SAC-PER: Soft Actor-Critic with Prioritized Experience Replay',
        ha='center', fontsize=9, style='italic', color='#555')

plt.tight_layout()
plt.savefig('assets/results/demo_delivery/architecture/system_architecture.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ System architecture diagram saved!")
plt.close()
