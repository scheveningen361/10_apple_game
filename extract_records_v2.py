#!/usr/bin/env python3
"""
Extract individual records from games_1000.txt
Each record = (board_state, label_remaining_apples)
"""

import numpy as np
import matplotlib.pyplot as plt
import re

# Parse games_1000.txt
with open('games_1000.txt', 'r') as f:
    lines = f.readlines()

print("Parsing games_1000.txt...")
NROWS, NCOLS = 10, 17
records = []
games_parsed = 0

i = 0
while i < len(lines):
    line = lines[i].strip()
    if not line:
        i += 1
        continue

    # Parse game header: "game N score=S"
    match = re.search(r'game \d+ score=(\d+)', line)
    if not match:
        i += 1
        continue

    total_score = int(match.group(1))

    # Parse board line: "board: v1 v2 v3 ..."
    i += 1
    if i >= len(lines):
        break
    board_line = lines[i].strip()
    if not board_line.startswith('board:'):
        continue

    board_vals = list(map(int, board_line.split()[1:]))
    if len(board_vals) != NROWS * NCOLS:
        print(f"Warning: game {games_parsed} has {len(board_vals)} board values, expected {NROWS * NCOLS}")
        i += 1
        continue

    # Parse moves line: "moves N: r1 c1 r2 c2 | r1 c1 r2 c2 | ..."
    i += 1
    if i >= len(lines):
        break
    moves_line = lines[i].strip()
    if not moves_line.startswith('moves'):
        continue

    # Extract moves from the pipe-separated format: "moves N: r1 c1 r2 c2 | r1 c1 r2 c2 | ..."
    moves_part = moves_line.split(': ', 1)
    if len(moves_part) < 2:
        i += 1
        continue

    # Split by pipe, then parse each move
    move_blocks = moves_part[1].split(' | ')
    moves = []
    for block in move_blocks:
        tokens = block.strip().split()
        if len(tokens) >= 4:
            try:
                r1, c1, r2, c2 = int(tokens[0]), int(tokens[1]), int(tokens[2]), int(tokens[3])
                moves.append((r1, c1, r2, c2))
            except ValueError:
                pass

    if not moves:
        i += 1
        continue

    # Reconstruct game step by step
    board = np.array(board_vals, dtype=np.uint8).reshape(NROWS, NCOLS)
    removed_so_far = 0

    for move_idx, (r1, c1, r2, c2) in enumerate(moves):
        # Record (board_before, remaining_apples_from_now)
        remaining = total_score - removed_so_far
        records.append({
            'board': board.copy(),
            'label': remaining
        })

        # Apply move: sum values in rectangle and set to 0
        removed_in_move = 0
        for r in range(r1, r2 + 1):
            if 0 <= r < NROWS:
                for c in range(c1, c2 + 1):
                    if 0 <= c < NCOLS and board[r, c] > 0:
                        removed_in_move += board[r, c]
                        board[r, c] = 0

        removed_so_far += removed_in_move

    games_parsed += 1
    i += 1

print(f"Total games parsed: {games_parsed}")
print(f"Total records generated: {len(records)}")

# Extract labels (use int for now to detect overflow)
labels_raw = np.array([r['label'] for r in records], dtype=np.int32)

# Filter to valid range [0, 170] - discard invalid records
valid_mask = (labels_raw >= 0) & (labels_raw <= 170)
valid_indices = np.where(valid_mask)[0]
invalid_count = len(labels_raw) - len(valid_indices)

if invalid_count > 0:
    print(f"Warning: {invalid_count} invalid records removed (labels outside [0,170])")
    print(f"  Raw range: [{labels_raw.min()}, {labels_raw.max()}]")
    records = [records[i] for i in valid_indices]

labels = np.array([r['label'] for r in records], dtype=np.uint8)
print(f"Label range (after filtering): [{labels.min()}, {labels.max()}]")
print(f"Valid records kept: {len(records):,}")
print(f"Label statistics: mean={labels.mean():.2f}, std={labels.std():.2f}, median={np.median(labels):.2f}")

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

print(f"\nAfter balancing:")
print(f"  Kept {len(kept_indices)} / {len(labels)} records")
print(f"  Removed: {len(labels) - len(kept_indices)}")
print(f"  Target per bucket: {target_per_bucket}")
print(f"  Bucket counts: {bucket_counts}")

# Print detailed table
print("\n" + "="*75)
print(f"{'Bucket':>6}  {'Score Range':>20}  {'Before':>8}  {'After':>8}  {'Removed':>8}  {'%Keep':>8}")
print("="*75)
for i in range(N_BUCKETS):
    lo = int(quantiles[i])
    hi = int(quantiles[i+1])
    before = bucket_counts_before[i]
    after = bucket_counts[i]
    removed = before - after
    pct_keep = (after / before * 100) if before > 0 else 0
    print(f"{i:>6}  {f'{lo}-{hi}':>20}  {before:>8}  {after:>8}  {removed:>8}  {pct_keep:>7.1f}%")

# Verify all buckets have same count
if len(set(bucket_counts)) == 1:
    print(f"\n✓ All {N_BUCKETS} buckets have exactly {bucket_counts[0]} records")
else:
    print(f"\n✗ Bucket counts are not equal: {bucket_counts}")
    print(f"  std: {bucket_counts.std():.2f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(N_BUCKETS), bucket_counts_before, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].axhline(target_per_bucket, color='red', linestyle='--', linewidth=2, label=f'Target: {target_per_bucket}')
axes[0].set_title(f'Before Balancing (Total: {len(labels):,} records)')
axes[0].set_xlabel('Bucket index')
axes[0].set_ylabel('Count')
axes[0].legend()
axes[0].grid(alpha=0.3, axis='y')
for i, c in enumerate(bucket_counts_before):
    axes[0].text(i, c + max(bucket_counts_before)*0.02, str(c), ha='center', fontsize=9)

axes[1].bar(range(N_BUCKETS), bucket_counts, color='coral', edgecolor='black', alpha=0.8)
axes[1].axhline(target_per_bucket, color='red', linestyle='--', linewidth=2, label=f'Target: {target_per_bucket}')
axes[1].set_title(f'After Balancing (Total: {len(kept_indices):,} records)')
axes[1].set_xlabel('Bucket index')
axes[1].set_ylabel('Count')
axes[1].set_ylim(0, bucket_counts_before.max() * 1.1)
axes[1].legend()
axes[1].grid(alpha=0.3, axis='y')
for i, c in enumerate(bucket_counts):
    axes[1].text(i, c + max(bucket_counts_before)*0.02, str(c), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('record_bucket_distribution_v2.png', dpi=150, bbox_inches='tight')
print("\nSaved: record_bucket_distribution_v2.png")

# Distribution histogram
fig2, ax = plt.subplots(figsize=(12, 5))
ax.hist(balanced_labels, bins=50, color='coral', edgecolor='black', alpha=0.7)
ax.axvline(balanced_labels.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {balanced_labels.mean():.1f}')
for i in range(N_BUCKETS + 1):
    ax.axvline(quantiles[i], color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('Label (Remaining Apples)')
ax.set_ylabel('Count')
ax.set_title(f'Balanced Records Label Distribution ({len(balanced_labels):,} records, 10 buckets)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('record_label_histogram_v2.png', dpi=150, bbox_inches='tight')
print("Saved: record_label_histogram_v2.png")

print(f"\nQuantile bucket edges: {[int(q) for q in quantiles]}")
