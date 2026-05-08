package main

import "sort"

// ── 비간섭 최댓값 (Non-Interference Maximum, NIM) ────────────────────────────
//
// 정의: 현재 보드에서 서로 비간섭(non-zero 셀이 겹치지 않는) 직사각형들을
//       선택했을 때 제거할 수 있는 사과(non-zero 셀) 수의 최댓값.
//
// 근사 알고리즘:
//   1. 합=10인 모든 유효 직사각형을 열거
//   2. non-zero 셀 수(nz) 오름차순 정렬
//   3. nz가 적은 것부터 탐욕적으로 선택 (이미 선택된 rect와 non-zero 셀 겹침 없을 때)
//   4. 선택된 rect들의 nz 합 반환

type nimRect struct {
	r  rect
	nz int16
}

func computeNIM(b *board) int {
	var val, cnt ps2d
	buildPS(b, &val, &cnt)

	var rects [maxRects]nimRect
	n := 0
	for r1 := 0; r1 < nRows; r1++ {
		for r2 := r1; r2 < nRows; r2++ {
			for c1 := 0; c1 < nCols; c1++ {
				for c2 := c1; c2 < nCols; c2++ {
					if qps(&val, r1, c1, r2, c2) == 10 {
						rects[n] = nimRect{
							r:  rect{int8(r1), int8(c1), int8(r2), int8(c2)},
							nz: qps(&cnt, r1, c1, r2, c2),
						}
						n++
					}
				}
			}
		}
	}
	if n == 0 {
		return 0
	}

	// nz 오름차순 정렬 (nz 적은 것 = 값이 큰 사과들만 포함 = 셀 수 적음)
	sort.Slice(rects[:n], func(i, j int) bool {
		return rects[i].nz < rects[j].nz
	})

	// 탐욕적 선택: non-zero 셀이 겹치는 rect 제외
	var used [nRows][nCols]bool
	total := 0

	for i := 0; i < n; i++ {
		r1, c1 := int(rects[i].r.r1), int(rects[i].r.c1)
		r2, c2 := int(rects[i].r.r2), int(rects[i].r.c2)

		overlap := false
		for r := r1; r <= r2 && !overlap; r++ {
			for c := c1; c <= c2 && !overlap; c++ {
				if b[r][c] != 0 && used[r][c] {
					overlap = true
				}
			}
		}
		if overlap {
			continue
		}

		total += int(rects[i].nz)
		for r := r1; r <= r2; r++ {
			for c := c1; c <= c2; c++ {
				if b[r][c] != 0 {
					used[r][c] = true
				}
			}
		}
	}
	return total
}

// ── NIA (Non-Interference Algorithm) ─────────────────────────────────────────
//
// 매 스텝: 유효한 직사각형 각각을 지운 후의 NIM을 계산하고,
// NIM이 가장 높은 직사각형을 선택.

func playNIA(b board) int {
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

		bestNIM := -1
		var bestR rect
		for i := 0; i < nCands; i++ {
			b2 := b
			applyRect(&b2, cands[i])
			if nim := computeNIM(&b2); nim > bestNIM {
				bestNIM = nim
				bestR = cands[i]
			}
		}
		removed += applyRect(&b, bestR)
	}
	return removed
}
