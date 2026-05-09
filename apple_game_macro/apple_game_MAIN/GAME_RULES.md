# Apple Game - Detailed Rules

## Table of Contents
1. [Game Overview](#game-overview)
2. [Board Structure](#board-structure)
3. [Basic Rules](#basic-rules)
4. [Valid Moves](#valid-moves)
5. [Scoring System](#scoring-system)
6. [Game Flow](#game-flow)
7. [Game End Conditions](#game-end-conditions)
8. [Examples](#examples)
9. [Strategy Considerations](#strategy-considerations)

---

## Game Overview

The Apple Game is a puzzle game where players select rectangular regions on a grid to remove apples. The goal is to maximize the number of apples removed by making strategic moves that satisfy specific mathematical constraints.

---

## Board Structure

### Grid Dimensions
- **Rows**: 10
- **Columns**: 17
- **Total Cells**: 170

### Cell Contents
- Each cell contains either:
  - **An apple with a number** (1-9): Represents the value of that apple
  - **Empty cell (0)**: No apple present (after removal)

### Initial State
- The board starts with apples numbered 1 through 9 randomly distributed across the grid
- All cells initially contain apples (no empty cells at the start)

---

## Basic Rules

### Rule 1: Rectangular Selection
- Players must select a **rectangular region** on the board
- The rectangle is defined by:
  - **Top-left corner**: (r1, c1) - row and column indices
  - **Bottom-right corner**: (r2, c2) - row and column indices
- The rectangle can be:
  - A single cell (1×1)
  - A row segment (1×n)
  - A column segment (n×1)
  - Any rectangular area (m×n)

### Rule 2: Sum Constraint
- The **sum of all numbers** within the selected rectangle must equal **exactly 10**
- This is the primary constraint for a valid move
- Examples:
  - Single cell with value 10: ❌ Invalid (numbers are 1-9 only)
  - Two cells: 3 + 7 = 10: ✅ Valid
  - Three cells: 2 + 3 + 5 = 10: ✅ Valid
  - Four cells: 1 + 2 + 3 + 4 = 10: ✅ Valid
  - Five cells: 1 + 1 + 2 + 3 + 3 = 10: ✅ Valid
  - Any combination summing to 10: ✅ Valid

### Rule 3: Minimum Apple Count
- The selected rectangle must contain **at least 2 apples**
- This prevents selecting single cells (even if they had value 10, which is impossible)
- Empty cells (0) do not count as apples
- Examples:
  - Rectangle with 1 apple: ❌ Invalid
  - Rectangle with 2 apples: ✅ Valid
  - Rectangle with 3+ apples: ✅ Valid

### Rule 4: Apple Removal
- When a valid move is executed:
  - All apples within the selected rectangle are **removed**
  - The cells become **empty (0)**
  - Empty cells remain empty for the rest of the game
  - They do not refill or regenerate

---

## Valid Moves

### Definition
A move is **valid** if and only if:
1. ✅ The selected region is a rectangle
2. ✅ The sum of all numbers in the rectangle equals **10**
3. ✅ The rectangle contains **at least 2 apples** (non-zero cells)

### Move Representation
Moves are represented as a tuple: `(r1, c1, r2, c2)`
- `r1, c1`: Top-left corner coordinates (row, column)
- `r2, c2`: Bottom-right corner coordinates (row, column)
- Indices are 0-based (0 to 9 for rows, 0 to 16 for columns)

### Finding Valid Moves
The game algorithm searches all possible rectangles:
- **Total possible rectangles**: (10 × 11 / 2) × (17 × 18 / 2) = 55 × 153 = **8,415 rectangles**
- For each rectangle, it checks:
  1. Sum of values = 10?
  2. Number of apples ≥ 2?

---

## Scoring System

### Score Calculation
- **Score = Number of apples removed**
- Each apple removed contributes **1 point** to the total score
- The value of the apple (1-9) does **not** affect the score
- Only the **count** of removed apples matters

### Example Scoring
```
Move 1: Removes 3 apples → +3 points
Move 2: Removes 5 apples → +5 points
Move 3: Removes 2 apples → +2 points
Total Score: 10 points
```

### Score Optimization
- To maximize score, players should:
  - Make as many moves as possible
  - Remove as many apples as possible per move
  - However, there's a trade-off:
    - Moves with fewer apples may allow more total moves
    - Moves with more apples give immediate higher scores

---

## Game Flow

### Turn Sequence
1. **Board State**: Current state of the 10×17 grid
2. **Move Selection**: Player (or algorithm) selects a valid rectangle
3. **Validation**: Check if the move satisfies all rules
4. **Execution**: Remove apples in the selected rectangle (set to 0)
5. **Score Update**: Add the number of removed apples to total score
6. **Repeat**: Continue until no valid moves remain

### Move Execution Process
```
Before Move:
[3] [5] [2] [1] [4]
[7] [2] [3] [8] [1]

Selected Rectangle: (0,0) to (0,2)  // First row, columns 0-2
Sum: 3 + 5 + 2 = 10 ✅
Apples: 3 ≥ 2 ✅

After Move:
[0] [0] [0] [1] [4]  // Removed apples become 0
[7] [2] [3] [8] [1]
```

---

## Game End Conditions

### Condition 1: No Valid Moves
- The game ends when **no valid rectangles** can be found
- This occurs when:
  - No remaining rectangles sum to exactly 10, OR
  - All remaining rectangles with sum=10 contain fewer than 2 apples

### Condition 2: Empty Board
- The game can end if all apples are removed
- However, this is rare and typically the game ends due to Condition 1

### Game Over
- When the game ends, the **final score** is the total number of apples removed
- The goal is to maximize this score

---

## Examples

### Example 1: Simple Two-Apple Move
```
Board State:
[3] [7] [2] [5]
[1] [4] [6] [8]

Valid Move: Select rectangle (0,0) to (0,1)
- Cells: [3] [7]
- Sum: 3 + 7 = 10 ✅
- Apples: 2 ≥ 2 ✅

After Move:
[0] [0] [2] [5]
[1] [4] [6] [8]

Score: +2 points
```

### Example 2: Multi-Apple Move
```
Board State:
[1] [2] [3] [4]
[5] [1] [2] [3]

Valid Move: Select rectangle (0,0) to (0,3)
- Cells: [1] [2] [3] [4]
- Sum: 1 + 2 + 3 + 4 = 10 ✅
- Apples: 4 ≥ 2 ✅

After Move:
[0] [0] [0] [0]
[5] [1] [2] [3]

Score: +4 points
```

### Example 3: Rectangular Area
```
Board State:
[2] [3] [1] [4]
[1] [2] [3] [4]

Valid Move: Select rectangle (0,0) to (1,1)
- Cells: [2] [3]
         [1] [2]
- Sum: 2 + 3 + 1 + 2 = 8 ❌ (Not valid)

Alternative Valid Move: Select rectangle (0,1) to (1,2)
- Cells: [3] [1]
         [2] [3]
- Sum: 3 + 1 + 2 + 3 = 9 ❌ (Not valid)

Another Valid Move: Select rectangle (0,0) to (1,0)
- Cells: [2]
         [1]
- Sum: 2 + 1 = 3 ❌ (Not valid)

Valid Move Found: Select rectangle (0,2) to (1,3)
- Cells: [1] [4]
         [3] [4]
- Sum: 1 + 4 + 3 + 4 = 12 ❌ (Not valid)

Actually Valid: Select rectangle (0,0) to (0,3) + (1,3)
Wait, let's find a real example:

Board:
[1] [2] [3] [4]
[2] [3] [1] [4]

Valid: (0,0) to (1,2)
- [1] [2] [3]
  [2] [3] [1]
- Sum: 1+2+3+2+3+1 = 12 ❌

Better example:
[1] [1] [2] [3] [3]
[2] [2] [2] [2] [2]

Valid: (0,0) to (0,4)
- [1] [1] [2] [3] [3]
- Sum: 1+1+2+3+3 = 10 ✅
- Apples: 5 ≥ 2 ✅
```

### Example 4: Invalid Moves
```
Board State:
[5] [5] [3] [7]

Invalid Move 1: Select (0,0) to (0,0)
- Single cell: [5]
- Sum: 5 ≠ 10 ❌
- Apples: 1 < 2 ❌

Invalid Move 2: Select (0,0) to (0,1)
- Cells: [5] [5]
- Sum: 5 + 5 = 10 ✅
- Apples: 2 ≥ 2 ✅
Wait, this is actually VALID!

Invalid Move 3: Select (0,2) to (0,3)
- Cells: [3] [7]
- Sum: 3 + 7 = 10 ✅
- Apples: 2 ≥ 2 ✅
This is also VALID!

Actually Invalid Example:
[9] [1] [2] [8]

Invalid: Select (0,0) to (0,0)
- Single cell: [9]
- Sum: 9 ≠ 10 ❌
- Apples: 1 < 2 ❌
```

---

## Strategy Considerations

### Key Strategic Principles

#### 1. Minimize Apples Per Move
- **Principle**: Use moves that remove fewer apples
- **Rationale**: Leaves more apples on the board for future moves
- **Trade-off**: Lower immediate score, but potentially more total moves

#### 2. Maximize High-Value Apples
- **Principle**: Prioritize moves containing higher numbers (8, 9)
- **Rationale**: High-value apples are harder to combine into sum=10
- **Benefit**: Removes difficult-to-use apples early

#### 3. Board State Management
- **Empty Space Distribution**: Consider how moves affect empty space
- **Connectivity**: Maintain connected regions of apples
- **Future Moves**: Consider what moves will be available after current move

#### 4. Move Order Optimization
- **Greedy Approach**: Choose the best immediate move
- **Look-Ahead**: Simulate future moves to choose better current move
- **Heuristic Balance**: Balance between immediate score and future opportunities

### Algorithm Strategies Used

#### Strategy 1: Min Apples, Max Number Bias
- **Primary**: Minimize apples removed per move
- **Tie-breaker**: Maximize the highest number in the move
- **Rationale**: Preserves more apples while removing difficult high numbers

#### Strategy 2: Min Apples, Center Close
- **Primary**: Minimize apples removed per move
- **Tie-breaker**: Choose moves closer to board center
- **Rationale**: Maintains board structure

#### Strategy 3: Min Apples, Empty Bias
- **Primary**: Minimize apples removed per move
- **Tie-breaker**: Choose moves near empty spaces
- **Rationale**: Creates strategic empty regions

#### Strategy 4: Max Number, Min Apples
- **Primary**: Maximize highest number in move
- **Tie-breaker**: Minimize apples removed
- **Rationale**: Removes high numbers first, then optimizes

### Computational Complexity

#### Finding Valid Moves
- **Time Complexity**: O(ROWS² × COLS²)
- **Space Complexity**: O(ROWS × COLS) for Summed Area Table
- **Optimization**: Uses Summed Area Table (SAT) for O(1) rectangle sum calculation

#### Total Possible Moves
- **Maximum rectangles**: ~8,415 per board state
- **Typical valid moves**: Varies greatly (0 to hundreds)
- **Game length**: Typically 10-50 moves depending on strategy

---

## Technical Implementation Details

### Move Validation Algorithm
```python
def is_valid_move(board, r1, c1, r2, c2):
    # Calculate sum of rectangle
    rectangle_sum = sum_rectangle(board, r1, c1, r2, c2)
    
    # Count apples in rectangle
    apple_count = count_apples(board, r1, c1, r2, c2)
    
    # Check constraints
    if rectangle_sum == 10 and apple_count >= 2:
        return True
    return False
```

### Score Calculation
```python
def calculate_score(move, board):
    r1, c1, r2, c2 = move
    apple_count = 0
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            if board[r][c] > 0:  # Has an apple
                apple_count += 1
    return apple_count
```

### Board Update
```python
def execute_move(board, move):
    r1, c1, r2, c2 = move
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            board[r][c] = 0  # Remove apple
```

---

## Summary

### Core Rules
1. **Board**: 10 rows × 17 columns grid
2. **Selection**: Must be a rectangular region
3. **Sum Constraint**: Selected numbers must sum to exactly 10
4. **Minimum Apples**: Must contain at least 2 apples
5. **Removal**: Selected apples are removed (become 0)
6. **Scoring**: 1 point per apple removed
7. **Game End**: No valid moves remain

### Winning Strategy
- Maximize total apples removed
- Balance between immediate score and future opportunities
- Consider board state and move order
- Use heuristics to guide move selection

---

## References

This document is based on analysis of the game implementation in:
- `apple_solver.py`: Core game logic and algorithms
- `main.py`: Game execution and automation
- Constants: `ROWS=10`, `COLS=17`, `TARGET_SUM=10`

---

*Last Updated: Based on code analysis of the Apple Game Macro project*

