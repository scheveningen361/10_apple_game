package main

// playGreedy plays the greedy strategy: fewest non-zero apples first,
// tie-break by highest max value. Deterministic.
func playGreedy(b board) int {
	var val, cnt ps2d
	removed := 0
	for {
		buildPS(&b, &val, &cnt)
		var (
			minCnt int16 = nTotal + 1
			bestMV int8
			best   rect
			found  bool
		)
		for r1 := 0; r1 < nRows; r1++ {
			for r2 := r1; r2 < nRows; r2++ {
				for c1 := 0; c1 < nCols; c1++ {
					for c2 := c1; c2 < nCols; c2++ {
						if qps(&val, r1, c1, r2, c2) != 10 {
							continue
						}
						c := qps(&cnt, r1, c1, r2, c2)
						if c > minCnt {
							continue
						}
						mv := maxInRect(&b, r1, c1, r2, c2)
						if c < minCnt || mv > bestMV {
							minCnt, bestMV = c, mv
							best = rect{int8(r1), int8(c1), int8(r2), int8(c2)}
							found = true
						}
					}
				}
			}
		}
		if !found {
			break
		}
		removed += applyRect(&b, best)
	}
	return removed
}
