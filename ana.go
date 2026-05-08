package main

// playMCGreedyAllCands (AnA) evaluates ALL valid rectangles (no min-count
// filter) using Greedy lookahead, picks the best.
func playMCGreedyAllCands(b board) int {
	var val, cnt ps2d
	var cb candBuf
	removed := 0

	for {
		buildPS(&b, &val, &cnt)
		nCands := 0
		for r1 := 0; r1 < nRows; r1++ {
			for r2 := r1; r2 < nRows; r2++ {
				for c1 := 0; c1 < nCols; c1++ {
					for c2 := c1; c2 < nCols; c2++ {
						if qps(&val, r1, c1, r2, c2) == 10 {
							c := qps(&cnt, r1, c1, r2, c2)
							cb.rects[nCands] = rect{int8(r1), int8(c1), int8(r2), int8(c2)}
							cb.cnts[nCands] = c
							nCands++
						}
					}
				}
			}
		}
		if nCands == 0 {
			break
		}
		bestScore := -1
		var bestR rect
		for i := 0; i < nCands; i++ {
			b2 := b
			applyRect(&b2, cb.rects[i])
			if s := playGreedy(b2); s > bestScore {
				bestScore = s
				bestR = cb.rects[i]
			}
		}
		removed += applyRect(&b, bestR)
	}
	return removed
}

// playMCGreedyAllCandsTrace is AnA with move recording.
func playMCGreedyAllCandsTrace(b board) (score int, moves []rect) {
	var val, cnt ps2d
	var cb candBuf

	for {
		buildPS(&b, &val, &cnt)
		nCands := 0
		for r1 := 0; r1 < nRows; r1++ {
			for r2 := r1; r2 < nRows; r2++ {
				for c1 := 0; c1 < nCols; c1++ {
					for c2 := c1; c2 < nCols; c2++ {
						if qps(&val, r1, c1, r2, c2) == 10 {
							cb.rects[nCands] = rect{int8(r1), int8(c1), int8(r2), int8(c2)}
							nCands++
						}
					}
				}
			}
		}
		if nCands == 0 {
			break
		}
		bestScore := -1
		var bestR rect
		for i := 0; i < nCands; i++ {
			b2 := b
			applyRect(&b2, cb.rects[i])
			if s := playGreedy(b2); s > bestScore {
				bestScore = s
				bestR = cb.rects[i]
			}
		}
		score += applyRect(&b, bestR)
		moves = append(moves, bestR)
	}
	return
}
