package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"time"
)

const improvedAnATopK = 4

type improvedCand struct {
	r       rect
	removed int
	future  int
	score   int
	maxVal  int8
}

func enumerateImprovedCandidates(b board) []improvedCand {
	var val, cnt ps2d
	buildPS(&b, &val, &cnt)

	cands := make([]improvedCand, 0, 128)
	for r1 := 0; r1 < nRows; r1++ {
		for r2 := r1; r2 < nRows; r2++ {
			for c1 := 0; c1 < nCols; c1++ {
				for c2 := c1; c2 < nCols; c2++ {
					if qps(&val, r1, c1, r2, c2) != 10 {
						continue
					}
					r := rect{int8(r1), int8(c1), int8(r2), int8(c2)}
					b2 := b
					removed := applyRect(&b2, r)
					future := playGreedy(b2)
					cands = append(cands, improvedCand{
						r:       r,
						removed: removed,
						future:  future,
						score:   removed + future,
						maxVal:  maxInRect(&b, r1, c1, r2, c2),
					})
				}
			}
		}
	}
	return cands
}

func betterImprovedCand(a, b improvedCand) bool {
	if a.score != b.score {
		return a.score > b.score
	}
	if a.future != b.future {
		return a.future > b.future
	}
	if a.removed != b.removed {
		return a.removed > b.removed
	}
	return a.maxVal > b.maxVal
}

func playImprovedAnA(b board) int {
	removed := 0

	for {
		cands := enumerateImprovedCandidates(b)
		if len(cands) == 0 {
			break
		}
		sort.Slice(cands, func(i, j int) bool {
			return betterImprovedCand(cands[i], cands[j])
		})

		top := improvedAnATopK
		if len(cands) < top {
			top = len(cands)
		}

		best := cands[0]
		bestRefined := -1
		for i := 0; i < top; i++ {
			root := cands[i]
			b2 := b
			applyRect(&b2, root.r)

			next := enumerateImprovedCandidates(b2)
			refined := root.removed
			if len(next) == 0 {
				refined = root.score
			} else {
				for _, n := range next {
					score := root.removed + n.score
					if score > refined {
						refined = score
					}
				}
			}

			if refined > bestRefined || (refined == bestRefined && betterImprovedCand(root, best)) {
				bestRefined = refined
				best = root
			}
		}

		removed += applyRect(&b, best.r)
	}

	return removed
}

func runImprovedAnAvsAnA(n int) {
	master := rand.New(rand.NewSource(time.Now().UnixNano()))
	boards := make([]board, n)
	for i := range boards {
		boards[i] = genBoard(master)
	}

	type result struct{ ana, improved int }
	results := make([]result, n)

	var (
		mu        sync.Mutex
		completed int
		totalDur  time.Duration
	)
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())

	fmt.Printf("ImprovedAnA(topK=%d)  vs  AnA  --  %d boards  --  %d CPUs\n\n", improvedAnATopK, n, runtime.NumCPU())
	overallStart := time.Now()

	for i := 0; i < n; i++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int) {
			defer func() { <-sem; wg.Done() }()
			t0 := time.Now()
			b := boards[idx]
			ana := playMCGreedyAllCands(b)
			improved := playImprovedAnA(b)
			dur := time.Since(t0)

			mu.Lock()
			completed++
			totalDur += dur
			eta := (totalDur / time.Duration(completed)) * time.Duration(n-completed)
			best := ana
			if improved > best {
				best = improved
			}
			as, is := " ", " "
			if ana == best {
				as = "*"
			}
			if improved == best {
				is = "*"
			}
			diff := improved - ana
			sign := "+"
			if diff < 0 {
				sign = ""
			}
			fmt.Printf("[%2d/%d]  AnA:%3d%s  Improved:%3d%s  (%s%d)   %6.1fs  ETA %v\n",
				completed, n, ana, as, improved, is, sign, diff,
				dur.Seconds(), eta.Round(time.Second))
			mu.Unlock()
			results[idx] = result{ana, improved}
		}(i)
	}
	wg.Wait()

	type stat struct{ sum, min, max int }
	st := [2]stat{{0, nTotal + 1, 0}, {0, nTotal + 1, 0}}
	wins := [2]int{}
	for _, r := range results {
		vals := [2]int{r.ana, r.improved}
		best := 0
		for _, v := range vals {
			if v > best {
				best = v
			}
		}
		for i, v := range vals {
			st[i].sum += v
			if v < st[i].min {
				st[i].min = v
			}
			if v > st[i].max {
				st[i].max = v
			}
			if v == best {
				wins[i]++
			}
		}
	}
	names := [2]string{"AnA", "ImprovedAnA"}
	printStatTable(names[:], st[0].sum, st[1].sum, st[0].min, st[1].min, st[0].max, st[1].max, wins[0], wins[1], n)
	fmt.Printf("  Improved gain : %+.2f\n", float64(st[1].sum-st[0].sum)/float64(n))
	fmt.Printf("  Total time    : %v\n", time.Since(overallStart).Round(time.Millisecond))
}
