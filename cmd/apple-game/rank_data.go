package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"time"
)

type rankCand struct {
	r       rect
	board   [nTotal]byte
	removed uint8
	target  uint8
	future  int
}

func collectRankCandidates(b board) []rankCand {
	var val, cnt ps2d
	buildPS(&b, &val, &cnt)

	cands := make([]rankCand, 0, 128)
	for r1 := 0; r1 < nRows; r1++ {
		for r2 := r1; r2 < nRows; r2++ {
			for c1 := 0; c1 < nCols; c1++ {
				for c2 := c1; c2 < nCols; c2++ {
					if qps(&val, r1, c1, r2, c2) != 10 || qps(&cnt, r1, c1, r2, c2) == 0 {
						continue
					}

					r := rect{int8(r1), int8(c1), int8(r2), int8(c2)}
					b2 := b
					removed := applyRect(&b2, r)
					future := playGreedy(b2)
					target := removed + future
					if target > nTotal {
						target = nTotal
					}

					var flat [nTotal]byte
					for i := 0; i < nRows; i++ {
						for j := 0; j < nCols; j++ {
							flat[i*nCols+j] = byte(b2[i][j])
						}
					}

					cands = append(cands, rankCand{
						r:       r,
						board:   flat,
						removed: uint8(removed),
						target:  uint8(target),
						future:  future,
					})
				}
			}
		}
	}
	return cands
}

func betterRankCand(a, b rankCand) bool {
	if a.target != b.target {
		return a.target > b.target
	}
	if a.future != b.future {
		return a.future > b.future
	}
	if a.removed != b.removed {
		return a.removed > b.removed
	}
	return false
}

func runGenRank(numGames int, filename string) {
	fmt.Printf("Generating rank data: %d games -> %s\n", numGames, filename)
	fmt.Printf("Format: repeated groups [uint16 n][candidate x n], candidate=[removed_1 target_1 board_170]\n\n")

	f, err := os.Create(filename)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot create %s: %v\n", filename, err)
		os.Exit(1)
	}
	defer f.Close()

	bw := bufio.NewWriterSize(f, 16<<20)
	rng := rand.New(rand.NewSource(20260511))
	start := time.Now()

	var (
		groups       int
		records      int
		totalScore   int
		maxGroupSize int
		hdr          [2]byte
		row          [nTotal + 2]byte
	)

	for g := 0; g < numGames; g++ {
		b := genBoard(rng)
		gameScore := 0

		for {
			cands := collectRankCandidates(b)
			if len(cands) == 0 {
				break
			}
			if len(cands) > maxGroupSize {
				maxGroupSize = len(cands)
			}

			binary.LittleEndian.PutUint16(hdr[:], uint16(len(cands)))
			if _, err := bw.Write(hdr[:]); err != nil {
				fmt.Fprintf(os.Stderr, "write header error: %v\n", err)
				os.Exit(1)
			}

			best := cands[0]
			for _, c := range cands {
				row[0] = c.removed
				row[1] = c.target
				copy(row[2:], c.board[:])
				if _, err := bw.Write(row[:]); err != nil {
					fmt.Fprintf(os.Stderr, "write record error: %v\n", err)
					os.Exit(1)
				}
				if betterRankCand(c, best) {
					best = c
				}
			}

			groups++
			records += len(cands)
			gameScore += applyRect(&b, best.r)
		}
		totalScore += gameScore

		if (g+1)%100 == 0 || g+1 == numGames {
			elapsed := time.Since(start)
			eta := time.Duration(0)
			if g+1 < numGames {
				eta = elapsed * time.Duration(numGames-g-1) / time.Duration(g+1)
			}
			fmt.Printf("  [%5d/%d] groups=%8d records=%10d avg_score=%.2f elapsed=%v ETA=%v\n",
				g+1, numGames, groups, records, float64(totalScore)/float64(g+1),
				elapsed.Round(time.Millisecond), eta.Round(time.Second))
		}
	}

	if err := bw.Flush(); err != nil {
		fmt.Fprintf(os.Stderr, "flush error: %v\n", err)
		os.Exit(1)
	}

	info, _ := f.Stat()
	fmt.Printf("\n=== Rank Data Generation Complete ===\n")
	fmt.Printf("  Games        : %d\n", numGames)
	fmt.Printf("  Groups       : %d\n", groups)
	fmt.Printf("  Records      : %d\n", records)
	fmt.Printf("  Avg groups   : %.1f/game\n", float64(groups)/float64(numGames))
	fmt.Printf("  Avg records  : %.1f/group\n", float64(records)/float64(groups))
	fmt.Printf("  Max group    : %d candidates\n", maxGroupSize)
	fmt.Printf("  Teacher avg  : %.2f / %d\n", float64(totalScore)/float64(numGames), nTotal)
	fmt.Printf("  File         : %s (%.1f MB)\n", filename, float64(info.Size())/1e6)
	fmt.Printf("  Time         : %v\n", time.Since(start).Round(time.Millisecond))
}
