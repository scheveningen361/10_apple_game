#!/usr/bin/env python3
"""
Extract individual records from data/raw/games_1000.txt
Each record = (board_state, label_remaining_apples)
"""

import numpy as np
import matplotlib.pyplot as plt
import re

# Parse data/raw/games_1000.txt
with open('data/raw/games_1000.txt', 'r') as f:
    content = f.read()

# Split by game blocks
games = content.strip().split('\n\n')
print(f"Total games: {len([g for g in games if g.strip()])}")

records = []
NROWS, NCOLS = 10, 17

for game_block in games:
    if not game_block.strip():
        continue

    lines = game_block.strip().split('\n')

    # Parse game header
    header = lines[0]
    match = re.search(r'game \d+ score=(\d+)', header)
    if not match:
        continue
    total_score = int(match.group(1))

    # Parse board
    board_line = lines[1]
    if not board_line.startswith('board:'):
        continue
    board_vals = list(map(int, board_line.split()[1:]))
    if len(board_vals) != NROWS * NCOLS:
        continue

    # Parse moves
    moves_line = lines[2]
    if not moves_line.startswith('moves'):
        continue

    # Extract moves: "moves N: r1 c1 r2 c2 | r1 c1 r2 c2 | ..."
    moves_part = moves_line.split(': ', 1)[1]
    move_strs = moves_part.split(' | ')
    moves = []
    for m_str in move_strs:
        parts = m_str.split()
        if len(parts) == 4:
            moves.append(tuple(map(int, parts)))

    if not moves:
        continue

    # Reconstruct game step by step
    board = np.array(board_vals, dtype=np.uint8).reshape(NROWS, NCOLS)
    remaining = total_score

    for move in moves:
        # Record current board state with label = remaining apples
        records.append({
            'board': board.copy(),
            'label': remaining
        })

        # Apply move: set all non-zero cells in rectangle to 0
        r1, c1, r2, c2 = move
        removed_in_move = 0
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if board[r, c] > 0:
                    removed_in_move += board[r, c]
                    board[r, c] = 0

        remaining -= removed_in_move

    if remaining != 0:
        print(f"Warning: game ended with remaining={remaining}, expected 0")

print(f"Total records generated: {len(records)}")

# Extract labels
labels = np.array([r['label'] for r in records], dtype=np.uint8)
print(f"Label range: [{labels.min()}, {labels.max()}]")
print(f"Label statistics: mean={labels.mean():.2f}, std={labels.std():.2f}")

# Create 10 equal-count buckets based on quantiles
N_BUCKETS = 10
target_per_bucket = len(labels) // N_BUCKETS

# Compute quantile boundaries for equal-count buckets
quantiles = np.percentile(labels, np.linspace(0, 100, N_BUCKETS + 1))
print(f"\nQuantile boundaries: {[int(q) for q in quantiles]}")

# Assign to buckets
bucket_ids = np.digitize(labels, quantiles[1:-1])  # 0 to N_BUCKETS-1
bucket_counts_before = np.bincount(bucket_ids, minlength=N_BUCKETS)

# Balance: keep exactly target_per_bucket from each bucket
kept_indices = []
bucket_counts = np.zeros(N_BUCKETS, dtype=int)

for i, bucket_id in enumerate(bucket_ids):
    if bucket_counts[bucket_id] < target_per_bucket:
        kept_indices.append(i)
        bucket_counts[bucket_id] += 1

kept_indices = np.array(kept_indices)
balanced_labels = labels[kept_indices]
balanced_records = [records[i] for i in kept_indices]

print(f"\nAfter balancing:")
print(f"  Kept {len(kept_indices)} / {len(labels)} records")
print(f"  Removed: {len(labels) - len(kept_indices)}")
print(f"  Each bucket: exactly {target_per_bucket} records")
print(f"  Bucket counts: {bucket_counts}")

# Print detailed table
print("\n" + "="*70)
print(f"{'Bucket':>6}  {'Score Range':>20}  {'Before':>8}  {'After':>8}  {'Removed':>8}")
print("="*70)
for i in range(N_BUCKETS):
    lo = int(quantiles[i])
    hi = int(quantiles[i+1])
    before = bucket_counts_before[i]
    after = bucket_counts[i]
    removed = before - after
    print(f"{i:>6}  {f'{lo}-{hi}':>20}  {before:>8}  {after:>8}  {removed:>8}")

# Verify all buckets have same count
assert len(set(bucket_counts)) == 1, "Not all buckets have same count!"
print(f"\n✓ All {N_BUCKETS} buckets have exactly {bucket_counts[0]} records")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(N_BUCKETS), bucket_counts_before, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].axhline(target_per_bucket, color='red', linestyle='--', linewidth=2, label=f'Target: {target_per_bucket}')
axes[0].set_title(f'Before Balancing (Total: {len(labels)} records)')
axes[0].set_xlabel('Bucket index')
axes[0].set_ylabel('Count')
axes[0].legend()
axes[0].grid(alpha=0.3, axis='y')
for i, c in enumerate(bucket_counts_before):
    axes[0].text(i, c + 20, str(c), ha='center', fontsize=9)

axes[1].bar(range(N_BUCKETS), bucket_counts, color='coral', edgecolor='black', alpha=0.8)
axes[1].axhline(target_per_bucket, color='red', linestyle='--', linewidth=2, label=f'Target: {target_per_bucket}')
axes[1].set_title(f'After Balancing (Total: {len(kept_indices)} records)')
axes[1].set_xlabel('Bucket index')
axes[1].set_ylabel('Count')
axes[1].set_ylim(0, bucket_counts_before.max() * 1.1)
axes[1].legend()
axes[1].grid(alpha=0.3, axis='y')
for i, c in enumerate(bucket_counts):
    axes[1].text(i, c + 20, str(c), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('reports/figures/record_bucket_distribution.png', dpi=150, bbox_inches='tight')
print("\nSaved: reports/figures/record_bucket_distribution.png")

# Distribution histogram
fig2, ax = plt.subplots(figsize=(12, 5))
ax.hist(balanced_labels, bins=50, color='coral', edgecolor='black', alpha=0.7)
ax.axvline(balanced_labels.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {balanced_labels.mean():.1f}')
for i in range(N_BUCKETS + 1):
    ax.axvline(quantiles[i], color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('Label (Remaining Apples)')
ax.set_ylabel('Count')
ax.set_title(f'Balanced Records Label Distribution ({len(balanced_labels)} records, 10 buckets)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/record_label_histogram.png', dpi=150, bbox_inches='tight')
print("Saved: reports/figures/record_label_histogram.png")

print(f"\nQuantile bucket edges: {[int(q) for q in quantiles]}")
