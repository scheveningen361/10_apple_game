#!/usr/bin/env python3
"""
Apple Game Greedy Distribution Visualization
Reads greedy_100k.txt and generates distribution plots
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

# Read scores
try:
    scores = np.loadtxt('greedy_100k.txt', dtype=int)
except FileNotFoundError:
    print("Error: greedy_100k.txt not found")
    sys.exit(1)

print(f"Loaded {len(scores)} scores")
print(f"Min: {scores.min()}, Max: {scores.max()}")
print(f"Mean: {scores.mean():.4f}, Median: {np.median(scores):.4f}")
print(f"SD: {scores.std():.4f}\n")

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Histogram with KDE
ax1 = plt.subplot(2, 3, 1)
ax1.hist(scores, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
kde = stats.gaussian_kde(scores)
xs = np.linspace(scores.min(), scores.max(), 200)
ax1.plot(xs, kde(xs), 'r-', linewidth=2, label='KDE')
ax1.axvline(scores.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.2f}')
ax1.axvline(np.median(scores), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.2f}')
ax1.set_xlabel('Score')
ax1.set_ylabel('Density')
ax1.set_title('Histogram with KDE')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Box plot
ax2 = plt.subplot(2, 3, 2)
bp = ax2.boxplot(scores, vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
ax2.set_ylabel('Score')
ax2.set_title('Box Plot')
ax2.grid(alpha=0.3, axis='y')
q1 = np.percentile(scores, 25)
q3 = np.percentile(scores, 75)
ax2.text(1.15, q1, f'Q1: {q1:.0f}', fontsize=10)
ax2.text(1.15, np.median(scores), f'Median: {np.median(scores):.0f}', fontsize=10)
ax2.text(1.15, q3, f'Q3: {q3:.0f}', fontsize=10)

# 3. Q-Q plot (normality check)
ax3 = plt.subplot(2, 3, 3)
stats.probplot(scores, dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot (Normal Distribution Test)')
ax3.grid(alpha=0.3)

# 4. Cumulative distribution
ax4 = plt.subplot(2, 3, 4)
sorted_scores = np.sort(scores)
cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
ax4.plot(sorted_scores, cumulative, linewidth=2, color='steelblue')
ax4.axhline(0.25, color='red', linestyle='--', alpha=0.5, label=f'P25: {np.percentile(scores, 25):.0f}')
ax4.axhline(0.50, color='green', linestyle='--', alpha=0.5, label=f'P50: {np.percentile(scores, 50):.0f}')
ax4.axhline(0.75, color='orange', linestyle='--', alpha=0.5, label=f'P75: {np.percentile(scores, 75):.0f}')
ax4.set_xlabel('Score')
ax4.set_ylabel('Cumulative Probability')
ax4.set_title('Cumulative Distribution')
ax4.legend()
ax4.grid(alpha=0.3)

# 5. Violin plot
ax5 = plt.subplot(2, 3, 5)
parts = ax5.violinplot([scores], positions=[1], widths=0.7, showmeans=True, showmedians=True)
ax5.set_ylabel('Score')
ax5.set_title('Violin Plot')
ax5.set_xticks([1])
ax5.set_xticklabels(['Greedy'])
ax5.grid(alpha=0.3, axis='y')

# 6. Statistics text
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

stats_text = f"""
STATISTICS SUMMARY
{'='*40}

Sample Size    : {len(scores):,}
Mean           : {scores.mean():.4f}
Median         : {np.median(scores):.4f}
Std Dev        : {scores.std():.4f}
MAD            : {np.mean(np.abs(scores - np.mean(scores))):.4f}
Variance       : {scores.var():.4f}

Min            : {scores.min()}
Max            : {scores.max()}
Range          : {scores.max() - scores.min()}

Q1 (P25)       : {np.percentile(scores, 25):.4f}
Q3 (P75)       : {np.percentile(scores, 75):.4f}
IQR            : {np.percentile(scores, 75) - np.percentile(scores, 25):.4f}

Skewness       : {stats.skew(scores):.4f}
Kurtosis       : {stats.kurtosis(scores):.4f}

Normality Test (Shapiro-Wilk):
{'='*40}
(Using sample of 5000 for speed)
"""

sample = np.random.choice(scores, 5000, replace=False)
stat, p_value = stats.shapiro(sample)
stats_text += f"Statistic      : {stat:.6f}\np-value        : {p_value:.6e}\n"

if p_value > 0.05:
    stats_text += "Result         : Normal ✓ (p > 0.05)"
else:
    stats_text += "Result         : Not Normal (p < 0.05)"

ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('greedy_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Graph saved: greedy_distribution.png")
plt.show()
