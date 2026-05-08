#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import re

# Parse scores from games_1000.txt
with open('games_1000.txt', 'r') as f:
    content = f.read()
scores = np.array([int(m) for m in re.findall(r'game \d+ score=(\d+)', content)])
print(f"Total scores: {len(scores)}, range: [{scores.min()}, {scores.max()}]")

N_BUCKETS = 10

# Equal-width buckets
bucket_width = (scores.max() - scores.min() + 1) / N_BUCKETS
edges = [scores.min() + i * bucket_width for i in range(N_BUCKETS + 1)]
print(f"Bucket width: {bucket_width:.2f}")
print(f"Edges: {[int(e) for e in edges]}")

# Assign to buckets
bucket_indices = np.digitize(scores, edges[1:]) - 1
bucket_indices = np.clip(bucket_indices, 0, N_BUCKETS - 1)

# Count before balancing
counts_before = np.bincount(bucket_indices, minlength=N_BUCKETS)
max_count = len(scores) // N_BUCKETS
print(f"\nBefore balancing:")
print(f"  Max count: {counts_before.max()}, Min count: {counts_before.min()}")
print(f"  Target per bucket: {max_count}")

# Balance by removing excess
kept_indices = []
bucket_counts = np.zeros(N_BUCKETS, dtype=int)

for i, bucket_id in enumerate(bucket_indices):
    if bucket_counts[bucket_id] < max_count:
        kept_indices.append(i)
        bucket_counts[bucket_id] += 1

kept_indices = np.array(kept_indices)
balanced_scores = scores[kept_indices]
balanced_bucket_indices = bucket_indices[kept_indices]

print(f"\nAfter balancing (keeping {len(balanced_scores)} / {len(scores)}):")
print(f"  Counts per bucket: {bucket_counts}")
print(f"  std: {bucket_counts.std():.1f}")

# Print table
print("\n" + "="*65)
print(f"{'Bucket':>6}  {'Range':>15}  {'Before':>8}  {'After':>8}  {'Removed':>8}")
print("="*65)
for i in range(N_BUCKETS):
    lo = int(edges[i])
    hi = int(edges[i+1]) - 1
    before = counts_before[i]
    after = bucket_counts[i]
    removed = before - after
    print(f"{i:>6}  {f'{lo}-{hi}':>15}  {before:>8}  {after:>8}  {removed:>8}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before
axes[0].bar(range(N_BUCKETS), counts_before, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].axhline(max_count, color='red', linestyle='--', linewidth=2, label=f'Target: {max_count}')
axes[0].set_title('Before Balancing')
axes[0].set_xlabel('Bucket index')
axes[0].set_ylabel('Count')
axes[0].set_ylim(0, counts_before.max() * 1.1)
axes[0].legend()
axes[0].grid(alpha=0.3, axis='y')
for i, c in enumerate(counts_before):
    axes[0].text(i, c + 2, str(c), ha='center', fontsize=9)

# After
axes[1].bar(range(N_BUCKETS), bucket_counts, color='coral', edgecolor='black', alpha=0.8)
axes[1].axhline(max_count, color='red', linestyle='--', linewidth=2, label=f'Target: {max_count}')
axes[1].set_title('After Balancing')
axes[1].set_xlabel('Bucket index')
axes[1].set_ylabel('Count')
axes[1].set_ylim(0, counts_before.max() * 1.1)
axes[1].legend()
axes[1].grid(alpha=0.3, axis='y')
for i, c in enumerate(bucket_counts):
    axes[1].text(i, c + 2, str(c), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('balanced_bucket_distribution.png', dpi=150, bbox_inches='tight')
print("\nSaved: balanced_bucket_distribution.png")

# Save balanced data
print(f"\nBucket edges: {[int(e) for e in edges]}")
print(f"Balanced dataset size: {len(balanced_scores)}")
print(f"Removed: {len(scores) - len(balanced_scores)} samples")
