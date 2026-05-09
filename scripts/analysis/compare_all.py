#!/usr/bin/env python3
"""
Compare Greedy, AnA, and Random Play distributions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

# Load datasets
import re

try:
    greedy = np.loadtxt('data/raw/greedy_100k.txt', dtype=int)
    random = np.loadtxt('data/raw/random_100k.txt', dtype=int)
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

# Parse AnA from data/raw/games_1000.txt (it has different format)
try:
    with open('data/raw/games_1000.txt', 'r') as f:
        content = f.read()
    ana = np.array([int(m) for m in re.findall(r'game \d+ score=(\d+)', content)], dtype=int)
except FileNotFoundError:
    print("Error: data/raw/games_1000.txt not found")
    sys.exit(1)

print("Data loaded:")
print(f"  Greedy : {len(greedy):,} scores")
print(f"  AnA    : {len(ana):,} scores")
print(f"  Random : {len(random):,} scores\n")

# Create comparison figure
fig = plt.figure(figsize=(18, 10))

# Color scheme
colors = {
    'greedy': 'steelblue',
    'ana': 'coral',
    'random': 'lightgreen'
}

# 1. Overlaid histograms
ax1 = plt.subplot(2, 3, 1)
ax1.hist(greedy, bins=40, density=True, alpha=0.5, label='Greedy', color=colors['greedy'], edgecolor='black')
ax1.hist(ana, bins=30, density=True, alpha=0.5, label='AnA', color=colors['ana'], edgecolor='black')
ax1.hist(random, bins=40, density=True, alpha=0.5, label='Random', color=colors['random'], edgecolor='black')
ax1.set_xlabel('Score')
ax1.set_ylabel('Density')
ax1.set_title('Distribution Comparison (Overlaid)')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Box plots
ax2 = plt.subplot(2, 3, 2)
bp = ax2.boxplot([greedy, ana, random], labels=['Greedy', 'AnA', 'Random'], patch_artist=True)
colors_list = [colors['greedy'], colors['ana'], colors['random']]
for patch, color in zip(bp['boxes'], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel('Score')
ax2.set_title('Box Plot Comparison')
ax2.grid(alpha=0.3, axis='y')

# 3. Cumulative distributions
ax3 = plt.subplot(2, 3, 3)
for scores, label, color in [(greedy, 'Greedy', colors['greedy']),
                              (ana, 'AnA', colors['ana']),
                              (random, 'Random', colors['random'])]:
    sorted_scores = np.sort(scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax3.plot(sorted_scores, cumulative, linewidth=2.5, label=label, color=color)
ax3.set_xlabel('Score')
ax3.set_ylabel('Cumulative Probability')
ax3.set_title('Cumulative Distribution Function')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Violin plots
ax4 = plt.subplot(2, 3, 4)
positions = [1, 2, 3]
parts = ax4.violinplot([greedy, ana, random], positions=positions, widths=0.7,
                        showmeans=True, showmedians=True)
ax4.set_xticks(positions)
ax4.set_xticklabels(['Greedy', 'AnA', 'Random'])
ax4.set_ylabel('Score')
ax4.set_title('Violin Plot Comparison')
ax4.grid(alpha=0.3, axis='y')

# 5. Statistics table
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')

strategies = ['Greedy', 'AnA', 'Random']
datasets = [greedy, ana, random]

stats_lines = ["Algorithm    Mean      Median    SD       Min   Max    P25   P75\n"]
stats_lines.append("="*75 + "\n")

for strategy, data in zip(strategies, datasets):
    mean = data.mean()
    median = np.median(data)
    sd = data.std()
    minv = data.min()
    maxv = data.max()
    p25 = np.percentile(data, 25)
    p75 = np.percentile(data, 75)

    line = f"{strategy:<12} {mean:7.2f}   {median:7.1f}    {sd:6.2f}   {minv:4d}  {maxv:4d}  {p25:4.0f}  {p75:4.0f}\n"
    stats_lines.append(line)

stats_text = "".join(stats_lines)
ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 6. Summary statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_lines = []
for strategy, data in zip(strategies, datasets):
    mad = np.mean(np.abs(data - data.mean()))
    skew = stats.skew(data)
    sample = np.random.choice(data, min(5000, len(data)), replace=False)
    _, p = stats.shapiro(sample)

    summary = f"{strategy}:\n"
    summary += f"  MAD: {mad:.2f}\n"
    summary += f"  Skewness: {skew:.4f}\n"
    summary += f"  Shapiro-Wilk p: {p:.2e}\n"
    summary += f"  Normal: {'Yes' if p > 0.05 else 'No'}\n\n"
    summary_lines.append(summary)

summary_text = "".join(summary_lines)
summary_text += "="*40 + "\n"
summary_text += f"Greedy vs AnA: +{ana.mean() - greedy.mean():.2f}\n"
summary_text += f"AnA vs Random: +{ana.mean() - random.mean():.2f}\n"
summary_text += f"Greedy vs Random: +{greedy.mean() - random.mean():.2f}\n"

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.suptitle('Apple Game: Strategy Comparison (Greedy vs AnA vs Random)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/comparison_all_strategies.png', dpi=150, bbox_inches='tight')
print("Graph saved: reports/figures/comparison_all_strategies.png")

# Print to console
print("\n" + "="*75)
print("COMPARISON TABLE")
print("="*75)
for strategy, data in zip(strategies, datasets):
    mean = data.mean()
    median = np.median(data)
    sd = data.std()
    minv = data.min()
    maxv = data.max()
    p25 = np.percentile(data, 25)
    p75 = np.percentile(data, 75)
    mad = np.mean(np.abs(data - data.mean()))

    print(f"\n{strategy} (n={len(data):,}):")
    print(f"  Mean: {mean:.4f}, Median: {median:.4f}")
    print(f"  SD: {sd:.4f}, MAD: {mad:.4f}")
    print(f"  Range: [{minv}, {maxv}]")
    print(f"  IQR: [{p25:.0f}, {p75:.0f}]")

print("\n" + "="*75)
print("PERFORMANCE RANKING")
print("="*75)
ranking = sorted([(s, d.mean()) for s, d in zip(strategies, datasets)], key=lambda x: -x[1])
for i, (name, mean) in enumerate(ranking, 1):
    print(f"{i}. {name:<10} : {mean:.4f}")

print("\n" + "="*75)
print("ADVANTAGE OVER BASELINE")
print("="*75)
greedy_mean = greedy.mean()
print(f"AnA advantage over Greedy  : +{ana.mean() - greedy_mean:.4f} (+{(ana.mean() / greedy_mean - 1) * 100:.2f}%)")
print(f"Greedy advantage over Random: +{greedy_mean - random.mean():.4f} (+{(greedy_mean / random.mean() - 1) * 100:.2f}%)")
print(f"AnA advantage over Random   : +{ana.mean() - random.mean():.4f} (+{(ana.mean() / random.mean() - 1) * 100:.2f}%)")
