#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import re

# Parse scores from games_1000.txt
with open('games_1000.txt', 'r') as f:
    content = f.read()
scores = np.array([int(m) for m in re.findall(r'game \d+ score=(\d+)', content)])
print(f"Total scores: {len(scores)}, range: [{scores.min()}, {scores.max()}]")

N_BUCKETS = 20

# Equal-width buckets
ew_edges = np.linspace(scores.min(), scores.max() + 1, N_BUCKETS + 1)
ew_labels = np.digitize(scores, ew_edges[1:])  # 0~19
ew_counts = np.bincount(ew_labels, minlength=N_BUCKETS)

# Quantile-based buckets
quantiles = np.percentile(scores, np.linspace(0, 100, N_BUCKETS + 1))
quantiles = np.unique(quantiles)  # remove duplicates
q_labels = np.digitize(scores, quantiles[1:])
q_labels = np.clip(q_labels, 0, N_BUCKETS - 1)
q_counts = np.bincount(q_labels, minlength=N_BUCKETS)

# Print table
print("\n" + "="*70)
print(f"{'Bucket':>6}  {'EqualWidth Range':>20}  {'Count':>6}  {'Quantile Range':>20}  {'Count':>6}")
print("="*70)
for i in range(N_BUCKETS):
    ew_lo = int(ew_edges[i])
    ew_hi = int(ew_edges[i+1]) - 1
    if i < len(quantiles) - 1:
        q_lo = int(quantiles[i])
        q_hi = int(quantiles[i+1]) - 1
    else:
        q_lo = q_hi = "?"
    print(f"{i:>6}  {f'{ew_lo}-{ew_hi}':>20}  {ew_counts[i]:>6}  {f'{q_lo}-{q_hi}':>20}  {q_counts[i]:>6}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(N_BUCKETS), ew_counts, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].set_title('Equal-Width Buckets (20)')
axes[0].set_xlabel('Bucket index')
axes[0].set_ylabel('Count')
axes[0].axhline(len(scores)/N_BUCKETS, color='red', linestyle='--', label=f'Ideal: {len(scores)//N_BUCKETS}')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].bar(range(len(q_counts)), q_counts, color='coral', edgecolor='black', alpha=0.8)
axes[1].set_title('Quantile-Based Buckets (20)')
axes[1].set_xlabel('Bucket index')
axes[1].set_ylabel('Count')
axes[1].axhline(len(scores)/N_BUCKETS, color='red', linestyle='--', label=f'Ideal: {len(scores)//N_BUCKETS}')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('bucket_distribution.png', dpi=150, bbox_inches='tight')
print("\nSaved: bucket_distribution.png")

# Summary
print(f"\nEqual-Width   - std of counts: {ew_counts.std():.1f}, max: {ew_counts.max()}, min: {ew_counts.min()}")
print(f"Quantile-Based- std of counts: {q_counts.std():.1f}, max: {q_counts.max()}, min: {q_counts.min()}")
print(f"\nQuantile bucket edges: {[int(q) for q in quantiles]}")
