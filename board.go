package main

import "math/rand"

const (
	nRows    = 10
	nCols    = 17
	nTotal   = nRows * nCols // 170
	maxRects = (nRows*(nRows+1)/2)*(nCols*(nCols+1)/2) // 8415
)

// board stores apple values; 0 means removed.
type board [nRows][nCols]int8

// ps2d is a 2-D prefix-sum table (row-0 and col-0 are always 0).
type ps2d [nRows + 1][nCols + 1]int16

// rect is a rectangle selection on the board.
type rect struct{ r1, c1, r2, c2 int8 }

// ── Prefix-sum helpers ────────────────────────────────────────────────────────

func buildPS(b *board, val, cnt *ps2d) {
	for i := 0; i < nRows; i++ {
		for j := 0; j < nCols; j++ {
			v := int16(b[i][j])
			val[i+1][j+1] = v + val[i][j+1] + val[i+1][j] - val[i][j]
			var nz int16
			if v > 0 {
				nz = 1
			}
			cnt[i+1][j+1] = nz + cnt[i][j+1] + cnt[i+1][j] - cnt[i][j]
		}
	}
}

func qps(ps *ps2d, r1, c1, r2, c2 int) int16 {
	return ps[r2+1][c2+1] - ps[r1][c2+1] - ps[r2+1][c1] + ps[r1][c1]
}

func maxInRect(b *board, r1, c1, r2, c2 int) int8 {
	var mv int8
	for i := r1; i <= r2; i++ {
		for j := c1; j <= c2; j++ {
			if b[i][j] > mv {
				mv = b[i][j]
			}
		}
	}
	return mv
}

func applyRect(b *board, r rect) int {
	removed := 0
	for i := r.r1; i <= r.r2; i++ {
		for j := r.c1; j <= r.c2; j++ {
			if b[i][j] > 0 {
				b[i][j] = 0
				removed++
			}
		}
	}
	return removed
}

// ── Board generation ──────────────────────────────────────────────────────────

func genBoard(rng *rand.Rand) board {
	for {
		var b board
		var sum int
		for i := 0; i < nRows; i++ {
			for j := 0; j < nCols; j++ {
				if i == nRows-1 && j == nCols-1 {
					continue
				}
				v := rng.Intn(9) + 1
				b[i][j] = int8(v)
				sum += v
			}
		}
		if r := (10 - sum%10) % 10; r >= 1 {
			b[nRows-1][nCols-1] = int8(r)
			return b
		}
	}
}
