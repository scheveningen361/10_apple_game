package main

import "math/rand"

// ── RAnA: Randomized AnA, Best-of-K ──────────────────────────────────────────
//
// AnA의 한계: 내부 Greedy 평가가 결정론적이라 항상 동일한 경로 탐색.
// 개선: Greedy 타이 브레이킹을 무작위화하고 K번 실행해 최고 점수 반환.
//
// 무작위화 지점 두 곳:
//   1. playGreedyRand  – 동점 (minCnt, maxVal) 후보 중 reservoir sampling
//   2. playAnARand     – Greedy 평가 점수가 같은 후보 중 reservoir sampling

// playGreedyRand is Greedy with random tie-breaking via reservoir sampling.
// Among all candidates sharing the same (minCnt, maxVal), each has equal
// probability of being selected.
func playGreedyRand(b board, rng *rand.Rand) int {
	var val, cnt ps2d
	removed := 0
	for {
		buildPS(&b, &val, &cnt)
		var (
			minCnt   int16 = nTotal + 1
			bestMV   int8
			best     rect
			found    bool
			tieCount int
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
						r := rect{int8(r1), int8(c1), int8(r2), int8(c2)}
						if c < minCnt || mv > bestMV {
							// strictly better: reset reservoir
							minCnt, bestMV = c, mv
							best = r
							found = true
							tieCount = 1
						} else if c == minCnt && mv == bestMV {
							// tie: reservoir sampling (uniform over all tied)
							tieCount++
							if rng.Intn(tieCount) == 0 {
								best = r
							}
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

// playAnARand is AnA with randomized Greedy evaluation.
// Uses playGreedyRand as evaluator; also randomly breaks ties among
// candidates that share the top Greedy-rand score.
func playAnARand(b board, rng *rand.Rand) int {
	var val, cnt ps2d
	var cands [maxRects]rect
	removed := 0

	for {
		buildPS(&b, &val, &cnt)
		nCands := 0
		for r1 := 0; r1 < nRows; r1++ {
			for r2 := r1; r2 < nRows; r2++ {
				for c1 := 0; c1 < nCols; c1++ {
					for c2 := c1; c2 < nCols; c2++ {
						if qps(&val, r1, c1, r2, c2) == 10 {
							cands[nCands] = rect{int8(r1), int8(c1), int8(r2), int8(c2)}
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
		tieCount := 0
		for i := 0; i < nCands; i++ {
			b2 := b
			applyRect(&b2, cands[i])
			s := playGreedyRand(b2, rng)
			if s > bestScore {
				bestScore = s
				bestR = cands[i]
				tieCount = 1
			} else if s == bestScore {
				tieCount++
				if rng.Intn(tieCount) == 0 {
					bestR = cands[i]
				}
			}
		}
		removed += applyRect(&b, bestR)
	}
	return removed
}

// playAnARandBestOf runs playAnARand K times and returns the maximum score.
// Time cost: K × AnA (approximately).
func playAnARandBestOf(b board, K int, rng *rand.Rand) int {
	best := 0
	for k := 0; k < K; k++ {
		if s := playAnARand(b, rng); s > best {
			best = s
		}
	}
	return best
}
