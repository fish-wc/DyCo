#!/usr/bin/env python3
"""
Generate Pareto Frontier plot for MBTI-DMAS paper
Combines real data from compare_appendx.xlsx with RACE scores from paper
""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 300

# Real data from compare_appendx.xlsx (tokens in thousands)
data = {
    # Method: (Total Tokens in K, RACE Overall Score)
    'SingleAgent': (76.5, 37.80),
    'CoTAgent': (126.9, 35.27),
    'ToTAgent': (18.0, 40.62),
    'SCAgent': (264.5, 38.25),
    'ReActAgent': (204.1, 38.22),
    'DebateAgent': (261.7, 38.66),
    'MetaGPT': (184.6, 39.72),
    'AutoGen': (198.4, 40.43),
    'CrewAI': (215.3, 40.88),
    'Func-DMAS': (56.9, 39.77),
    '4×ENFP': (66.9, 33.30),
    '4×INFJ': (89.3, 42.53),
    'NF Group': (288.4, 41.24),  # infj_infp_enfj_enfp
    'NT Group': (276.0, 42.04),  # intj_intp_entj_entp
    'ST Group': (290.2, 42.43),  # istj_estj_istp_estp
    'SF Group': (327.0, 41.85),  # isfj_esfj_isfp_esfp
    'NT-ST Mixed': (298.7, 41.81),  # intp_entp_entj_istj
    'Diverse Group': (234.6, 42.75),  # entj_intj_estp_isfj
}

# Separate by category for coloring
single_agent = ['SingleAgent', 'CoTAgent', 'ToTAgent', 'SCAgent', 'ReActAgent']
multi_agent_baseline = ['DebateAgent', 'MetaGPT', 'AutoGen', 'CrewAI']
mbti_dmas = ['Func-DMAS', '4×ENFP', '4×INFJ', 'NF Group', 'NT Group', 
        'ST Group', 'SF Group', 'NT-ST Mixed', 'Diverse Group']

# Create figure
fig, ax = plt.subplots(figsize=(7, 5))

# Plot each category with different colors and markers
for method, (tokens, race) in data.items():
    if method in single_agent:
        color = '#1f77b4'  # Blue
        marker = 'o'
        label = 'Single-Agent' if method == single_agent[0] else ''
    elif method in multi_agent_baseline:
        color = '#ff7f0e'  # Orange
        marker = 's'
        label = 'Multi-Agent Baseline' if method == multi_agent_baseline[0] else ''
    elif method in mbti_dmas:
        color = '#2ca02c'  # Green
        marker = '^'
      label = 'MBTI-DMAS' if method == mbti_dmas[0] else ''
    
    ax.scatter(tokens, race, c=color, marker=marker, s=80, 
        alpha=0.7, edgecolors='black', linewidth=0.5, label=label)
    
    # Add error bars (std from paper)
    if method in ['SingleAgent', 'ToTAgent', 'CrewAI', 'Diverse Group']:
        std = 0.38 if method == 'SingleAgent' else (0.42 if method == 'ToTAgent' else (0.45 if method == 'CrewAI' else 0.31))
      ax.errorbar(tokens, race, yerr=std, fmt='none', ecolor='gray', 
                   alpha=0.3, capsize=3, linewidth=1)

# Identify Pareto frontier
# A point is Pareto-optimal if no other point has both higher RACE and lower tokens
pareto_points = []
for method, (tokens, race) in data.items():
    is_pareto = True
    for other_method, (other_tokens, other_race) in data.items():
        if other_tokens < tokens and other_race > race:
        is_pareto = False
            break
    if is_pareto:
        pareto_points.append((tokens, race, method))

# Sort Pareto points by tokens
pareto_points.sort(key=lambda x: x[0])

# Draw Pareto frontier curve
if len(pareto_points) > 1:
    pareto_tokens = [p[0] for p in pareto_points]
    pareto_race = [p[1] for p in pareto_points]
    ax.plot(pareto_tokens, pareto_race, 'k--', alpha=0.3, linewidth=1.5, 
            label='Pareto Frontier', zorder=1)

# Annotate key points
key_points = {
    'ToTAgent': (0, 8),
    'Func-DMAS': (0, -8),
    'Diverse Group': (10, 5),
    'CrewAI': (-15, 5),
    '4×INFJ': (10, -8),
}

for method, (tokens, race) in data.items():
    if method in key_points:
      offset_x, offset_y = key_points[method]
        ax.annotate(method, (tokens, race), 
           xytext=(offset_x, offset_y), textcoords='offset points',
               fontsize=8, ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2',
                    color='gray', lw=0.8))

# Set labels and title
ax.set_xlabel('Average Total Tokens per Task (×1000)', fontsize=11)
ax.set_ylabel('RACE Overall Score', fontsize=11)
ax.set_title('Efficiency-Quality Pareto Frontier', fontsize=12, fontweight='bold')

# Set axis limits with some padding
ax.set_xlim(0, 350)
ax.set_ylim(32, 44)

# Add grid
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

# Legend
handles, labels = ax.get_legend_handles_labels()
# Remove duplicates
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='lower right', 
         framealpha=0.9, edgecolor='gray')

# Tight layout
plt.tight_layout()

# Save figure
output_path = 'pareto_frontier.pdf'
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"Pareto frontier plot saved to: {output_path}")

# Also save as PNG for preview
plt.savefig('pareto_frontier.png', format='png', bbox_inches='tight', dpi=300)
print(f"PNG preview saved to: pareto_frontier.png")

plt.close()

# Print Pareto-optimal points
print("\nPareto-optimal configurations:")
for tokens, race, method in pareto_points:
    print(f"  {method:20s}: {tokens:6.1f}K tokens, RACE {race:.2f}")
