"""Generate neural network structure diagram"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(7, 7.5, 'SAC Neural Network Architecture', ha='center', fontsize=16, fontweight='bold')

# Colors
color_input = '#E8F4F8'
color_hidden = '#FFF4E6'
color_output = '#F0F8E8'
color_special = '#FADBD8'

# ===== ACTOR NETWORK =====
ax.text(3.5, 6.8, 'Actor Network (Policy)', ha='center', fontsize=12, fontweight='bold')

# Input layer
input_box = Rectangle((2.2, 5.5), 2.6, 0.8, edgecolor='#2C5F7C', facecolor=color_input, linewidth=2)
ax.add_patch(input_box)
ax.text(3.5, 6.1, 'Input Layer', ha='center', fontsize=9, fontweight='bold')
ax.text(3.5, 5.8, '30D State', ha='center', fontsize=8)

# Hidden layers
hidden1_box = Rectangle((2.2, 4.2), 2.6, 0.8, edgecolor='#E67E22', facecolor=color_hidden, linewidth=2)
ax.add_patch(hidden1_box)
ax.text(3.5, 4.8, 'Hidden Layer 1', ha='center', fontsize=9, fontweight='bold')
ax.text(3.5, 4.5, '256 units (ReLU)', ha='center', fontsize=8)

hidden2_box = Rectangle((2.2, 2.9), 2.6, 0.8, edgecolor='#E67E22', facecolor=color_hidden, linewidth=2)
ax.add_patch(hidden2_box)
ax.text(3.5, 3.5, 'Hidden Layer 2', ha='center', fontsize=9, fontweight='bold')
ax.text(3.5, 3.2, '256 units (ReLU)', ha='center', fontsize=8)

# Output layer
output_box = Rectangle((2.2, 1.6), 2.6, 0.8, edgecolor='#27AE60', facecolor=color_output, linewidth=2)
ax.add_patch(output_box)
ax.text(3.5, 2.2, 'Output Layer', ha='center', fontsize=9, fontweight='bold')
ax.text(3.5, 1.9, '2D Action (μ, σ)', ha='center', fontsize=8)

# Action details
ax.text(3.5, 1.3, 'n: Propeller RPM', ha='center', fontsize=7, style='italic')
ax.text(3.5, 1.1, 'δ: Rudder Angle', ha='center', fontsize=7, style='italic')

# Arrows for Actor
for y1, y2 in [(5.5, 5.0), (4.2, 3.7), (2.9, 2.4)]:
    arrow = FancyArrowPatch((3.5, y1), (3.5, y2), arrowstyle='->', mutation_scale=15, color='#4A90E2', linewidth=2)
    ax.add_patch(arrow)

# ===== CRITIC NETWORK =====
ax.text(10.5, 6.8, 'Critic Network (Q-function)', ha='center', fontsize=12, fontweight='bold')

# Input layer (State + Action)
input_box2 = Rectangle((9.2, 5.5), 2.6, 0.8, edgecolor='#2C5F7C', facecolor=color_input, linewidth=2)
ax.add_patch(input_box2)
ax.text(10.5, 6.1, 'Input Layer', ha='center', fontsize=9, fontweight='bold')
ax.text(10.5, 5.8, '32D (30D State + 2D Action)', ha='center', fontsize=7)

# Hidden layers
hidden1_box2 = Rectangle((9.2, 4.2), 2.6, 0.8, edgecolor='#8E44AD', facecolor=color_hidden, linewidth=2)
ax.add_patch(hidden1_box2)
ax.text(10.5, 4.8, 'Hidden Layer 1', ha='center', fontsize=9, fontweight='bold')
ax.text(10.5, 4.5, '256 units (ReLU)', ha='center', fontsize=8)

hidden2_box2 = Rectangle((9.2, 2.9), 2.6, 0.8, edgecolor='#8E44AD', facecolor=color_hidden, linewidth=2)
ax.add_patch(hidden2_box2)
ax.text(10.5, 3.5, 'Hidden Layer 2', ha='center', fontsize=9, fontweight='bold')
ax.text(10.5, 3.2, '256 units (ReLU)', ha='center', fontsize=8)

# Output layer
output_box2 = Rectangle((9.2, 1.6), 2.6, 0.8, edgecolor='#3498DB', facecolor=color_output, linewidth=2)
ax.add_patch(output_box2)
ax.text(10.5, 2.2, 'Output Layer', ha='center', fontsize=9, fontweight='bold')
ax.text(10.5, 1.9, '1D Q-value', ha='center', fontsize=8)

# Twin Q note
ax.text(10.5, 1.3, 'Twin Q-networks (Q1, Q2)', ha='center', fontsize=7, style='italic')
ax.text(10.5, 1.1, 'min(Q1, Q2) for target', ha='center', fontsize=7, style='italic')

# Arrows for Critic
for y1, y2 in [(5.5, 5.0), (4.2, 3.7), (2.9, 2.4)]:
    arrow = FancyArrowPatch((10.5, y1), (10.5, y2), arrowstyle='->', mutation_scale=15, color='#4A90E2', linewidth=2)
    ax.add_patch(arrow)

# ===== CONNECTIONS =====
# State to both networks
state_circle = Circle((1, 6), 0.3, edgecolor='#2C5F7C', facecolor=color_input, linewidth=2)
ax.add_patch(state_circle)
ax.text(1, 6, 'S', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(1, 5.4, 'State\n(30D)', ha='center', fontsize=7)

arrow_s_actor = FancyArrowPatch((1.3, 6), (2.2, 5.9), arrowstyle='->', mutation_scale=12, color='#2C5F7C', linewidth=1.5)
ax.add_patch(arrow_s_actor)

arrow_s_critic = FancyArrowPatch((1.3, 6), (9.2, 5.9), arrowstyle='->', mutation_scale=12, color='#2C5F7C', linewidth=1.5, linestyle='--')
ax.add_patch(arrow_s_critic)

# Action from Actor to Critic
arrow_a_critic = FancyArrowPatch((4.8, 2), (9.2, 5.7), arrowstyle='->', mutation_scale=12, color='#27AE60', linewidth=1.5, linestyle='--')
ax.add_patch(arrow_a_critic)
ax.text(7, 4.2, 'Action\n(2D)', ha='center', fontsize=7, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ===== SPECIAL MODULES =====
# Entropy module
entropy_box = Rectangle((5.8, 5.5), 1.8, 1.3, edgecolor='#F39C12', facecolor=color_special, linewidth=2, linestyle='--')
ax.add_patch(entropy_box)
ax.text(6.7, 6.5, 'Adaptive Entropy', ha='center', fontsize=9, fontweight='bold')
ax.text(6.7, 6.2, 'α (Temperature)', ha='center', fontsize=8)
ax.text(6.7, 5.9, 'Auto-tuned', ha='center', fontsize=7, style='italic')

# Target networks
target_box = Rectangle((5.8, 3.5), 1.8, 1.3, edgecolor='#E74C3C', facecolor=color_special, linewidth=2, linestyle='--')
ax.add_patch(target_box)
ax.text(6.7, 4.5, 'Target Networks', ha='center', fontsize=9, fontweight='bold')
ax.text(6.7, 4.2, 'Soft Update', ha='center', fontsize=8)
ax.text(6.7, 3.9, 'τ = 0.005', ha='center', fontsize=7, style='italic')

# PER module
per_box = Rectangle((5.8, 1.6), 1.8, 1.3, edgecolor='#3498DB', facecolor=color_special, linewidth=2, linestyle='--')
ax.add_patch(per_box)
ax.text(6.7, 2.6, 'PER Sampling', ha='center', fontsize=9, fontweight='bold')
ax.text(6.7, 2.3, 'Priority: TD-error', ha='center', fontsize=8)
ax.text(6.7, 2.0, 'α=0.6, β=0.4', ha='center', fontsize=7, style='italic')

# ===== INPUT DETAILS =====
ax.text(7, 0.5, 'Input State (30D): Position (x,y), Velocity (u,v,r), Heading (ψ), Reference trajectory, Disturbances (Vc, Vw, βc, βw), ...',
        ha='center', fontsize=7, style='italic', bbox=dict(boxstyle='round', facecolor='#E8F4F8', alpha=0.5))

# ===== LEGEND =====
legend_elements = [
    mpatches.Patch(facecolor=color_input, edgecolor='#2C5F7C', label='Input'),
    mpatches.Patch(facecolor=color_hidden, edgecolor='#E67E22', label='Hidden'),
    mpatches.Patch(facecolor=color_output, edgecolor='#27AE60', label='Output'),
    mpatches.Patch(facecolor=color_special, edgecolor='black', linestyle='--', label='Special Module'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig('assets/results/demo_delivery/architecture/network_structure.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Network structure diagram saved!")
plt.close()
