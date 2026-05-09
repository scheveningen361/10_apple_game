package main

import "math/rand"

// rolloutBuf holds pre-allocated scratch for random rollouts.
type rolloutBuf struct {
	rects [maxRects]rect
}

// candBuf holds pre-allocated scratch for MC candidate evaluation.
type candBuf struct {
	rects [maxRects]rect
	cnts  [maxRects]int16
}

// playRandom plays randomly (used as MC rollout). Returns total apples removed.
func playRandom(b board, rng *rand.Rand, buf *rolloutBuf) int {
	var val, cnt ps2d
	removed := 0
	for {
		buildPS(&b, &val, &cnt)
		n := 0
		for r1 := 0; r1 < nRows; r1++ {
			for r2 := r1; r2 < nRows; r2++ {
				for c1 := 0; c1 < nCols; c1++ {
					for c2 := c1; c2 < nCols; c2++ {
						if qps(&val, r1, c1, r2, c2) == 10 {
							buf.rects[n] = rect{int8(r1), int8(c1), int8(r2), int8(c2)}
							n++
						}
					}
				}
			}
		}
		if n == 0 {
			break
		}
		removed += applyRect(&b, buf.rects[rng.Intn(n)])
	}
	return removed
}

// playMCGreedyRollout evaluates only min-count candidates using Greedy rollout.
func playMCGreedyRollout(b board) int {
	var val, cnt ps2d
	var cb candBuf
	removed := 0

	for {
		buildPS(&b, &val, &cnt)
		nCands := 0
		var minCnt int16 = nTotal + 1
		for r1 := 0; r1 < nRows; r1++ {
			for r2 := r1; r2 < nRows; r2++ {
				for c1 := 0; c1 < nCols; c1++ {
					for c2 := c1; c2 < nCols; c2++ {
						if qps(&val, r1, c1, r2, c2) != 10 {
							continue
						}
						c := qps(&cnt, r1, c1, r2, c2)
						if c < minCnt {
							minCnt = c
						}
						cb.rects[nCands] = rect{int8(r1), int8(c1), int8(r2), int8(c2)}
						cb.cnts[nCands] = c
						nCands++
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
			if cb.cnts[i] != minCnt {
				continue
			}
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

// playMC plays Monte Carlo with K random rollouts per min-count candidate.
func playMC(b board, rng *rand.Rand, K int) int {
	var val, cnt ps2d
	var cb candBuf
	var rb rolloutBuf
	removed := 0

	for {
		buildPS(&b, &val, &cnt)
		nCands := 0
		var minCnt int16 = nTotal + 1
		for r1 := 0; r1 < nRows; r1++ {
			for r2 := r1; r2 < nRows; r2++ {
				for c1 := 0; c1 < nCols; c1++ {
					for c2 := c1; c2 < nCols; c2++ {
						if qps(&val, r1, c1, r2, c2) != 10 {
							continue
						}
						c := qps(&cnt, r1, c1, r2, c2)
						if c < minCnt {
							minCnt = c
						}
						cb.rects[nCands] = rect{int8(r1), int8(c1), int8(r2), int8(c2)}
						cb.cnts[nCands] = c
						nCands++
					}
				}
			}
		}
		if nCands == 0 {
			break
		}
		bestTotal := -1
		var bestR rect
		for i := 0; i < nCands; i++ {
			if cb.cnts[i] != minCnt {
				continue
			}
			b2 := b
			applyRect(&b2, cb.rects[i])
			total := 0
			for k := 0; k < K; k++ {
				b3 := b2
				total += playRandom(b3, rng, &rb)
			}
			if total > bestTotal {
				bestTotal = total
				bestR = cb.rects[i]
			}
		}
		removed += applyRect(&b, bestR)
	}
	return removed
}

// playMCGreedyAllCands2Ply evaluates all candidates with 2-ply greedy lookahead.
func playMCGreedyAllCands2Ply(b board) int {
	var val1, cnt1 ps2d
	var val2, cnt2 ps2d
	var l1 [maxRects]rect
	var l2 [maxRects]rect
	removed := 0

	for {
		buildPS(&b, &val1, &cnt1)
		n1 := 0
		for r1 := 0; r1 < nRows; r1++ {
			for r2 := r1; r2 < nRows; r2++ {
				for c1 := 0; c1 < nCols; c1++ {
					for c2 := c1; c2 < nCols; c2++ {
						if qps(&val1, r1, c1, r2, c2) == 10 {
							l1[n1] = rect{int8(r1), int8(c1), int8(r2), int8(c2)}
							n1++
						}
					}
				}
			}
		}
		if n1 == 0 {
			break
		}

		bestScore := -1
		var bestR rect
		for i := 0; i < n1; i++ {
			b2 := b
			applyRect(&b2, l1[i])
			buildPS(&b2, &val2, &cnt2)
			n2 := 0
			for r1 := 0; r1 < nRows; r1++ {
				for r2 := r1; r2 < nRows; r2++ {
					for c1 := 0; c1 < nCols; c1++ {
						for c2 := c1; c2 < nCols; c2++ {
							if qps(&val2, r1, c1, r2, c2) == 10 {
								l2[n2] = rect{int8(r1), int8(c1), int8(r2), int8(c2)}
								n2++
							}
						}
					}
				}
			}
			var score int
			if n2 == 0 {
				score = playGreedy(b2)
			} else {
				score = -1
				for j := 0; j < n2; j++ {
					b3 := b2
					applyRect(&b3, l2[j])
					if s := playGreedy(b3); s > score {
						score = s
					}
				}
			}
			if score > bestScore {
				bestScore = score
				bestR = l1[i]
			}
		}
		removed += applyRect(&b, bestR)
	}
	return removed
}
