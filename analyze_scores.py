#!/usr/bin/env python3
"""
Parse games_1000.txt and generate score distribution summary.
"""

from collections import Counter
import re

# Read games_1000.txt
with open('games_1000.txt', 'r') as f:
    content = f.read()

# Extract all scores using regex
pattern = r'game \d+ score=(\d+)'
scores = [int(m) for m in re.findall(pattern, content)]

print(f"Total games: {len(scores)}")
print(f"Unique scores: {len(set(scores))}\n")

# Count occurrences
score_counts = Counter(scores)

# Write to file
with open('score_distribution.txt', 'w') as f:
    f.write("Score Distribution Summary\n")
    f.write("==========================\n\n")
    f.write(f"Total games: {len(scores)}\n")
    f.write(f"Unique scores: {len(set(scores))}\n")
    f.write(f"Min: {min(scores)}\n")
    f.write(f"Max: {max(scores)}\n\n")

    f.write("Score Count Frequency Percentage\n")
    f.write("───── ───── ────────── ──────────\n")

    for score in sorted(score_counts.keys()):
        count = score_counts[score]
        freq = count / len(scores) * 100
        f.write(f"{score:5d} {count:5d} {count:10d} {freq:6.2f}%\n")

print("✓ 점수별 분포 저장: score_distribution.txt")
with open('score_distribution.txt', 'r') as f:
    print("\n" + f.read())
