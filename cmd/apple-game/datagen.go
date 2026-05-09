package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ── Training data generation ──────────────────────────────────────────────────

type gameRecord struct {
	flat  [nRows * nCols]uint8
	label uint8
}

// collectGameRecords plays Greedy on b and returns one record per step.
// label[t] = total apples removable from step t onward.
func collectGameRecords(b board) []gameRecord {
	var val, cnt ps2d

	type stepInfo struct {
		snap    board
		removed int
	}
	var steps []stepInfo

	for {
		buildPS(&b, &val, &cnt)
		var minCnt int16 = nTotal + 1
		var bestMV int8
		var best rect
		found := false

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
		snap := b
		removed := applyRect(&b, best)
		steps = append(steps, stepInfo{snap, removed})
	}

	if len(steps) == 0 {
		return nil
	}

	total := 0
	for _, s := range steps {
		total += s.removed
	}

	records := make([]gameRecord, len(steps))
	remaining := total
	for i, s := range steps {
		var rec gameRecord
		for r := 0; r < nRows; r++ {
			for c := 0; c < nCols; c++ {
				rec.flat[r*nCols+c] = uint8(s.snap[r][c])
			}
		}
		rec.label = uint8(remaining)
		records[i] = rec
		remaining -= s.removed
	}
	return records
}

func runGenerate(numGames int, filename string) {
	f, err := os.Create(filename)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot create %s: %v\n", filename, err)
		os.Exit(1)
	}
	defer f.Close()

	bw := bufio.NewWriterSize(f, 8<<20)
	rng := rand.New(rand.NewSource(42))
	start := time.Now()
	totalRecords := 0
	var row [nRows*nCols + 1]byte

	for g := 0; g < numGames; g++ {
		for _, rec := range collectGameRecords(genBoard(rng)) {
			copy(row[:nRows*nCols], rec.flat[:])
			row[nRows*nCols] = rec.label
			bw.Write(row[:])
			totalRecords++
		}
		if (g+1)%1000 == 0 {
			elapsed := time.Since(start)
			eta := elapsed * time.Duration(numGames-(g+1)) / time.Duration(g+1)
			fmt.Printf("  [%5d/%d]  records: %7d  elapsed: %v  ETA: %v\n",
				g+1, numGames, totalRecords,
				elapsed.Round(time.Millisecond), eta.Round(time.Second))
		}
	}
	bw.Flush()

	elapsed := time.Since(start)
	info, _ := f.Stat()
	fmt.Printf("\n%d games -> %d records\n", numGames, totalRecords)
	fmt.Printf("  avg %.1f steps/game\n", float64(totalRecords)/float64(numGames))
	fmt.Printf("  file: %s  (%.1f MB)\n", filename, float64(info.Size())/1e6)
	fmt.Printf("  time: %v\n", elapsed.Round(time.Millisecond))
}

// ── AnA benchmark with game log ───────────────────────────────────────────────

type gameLog struct {
	idx   int
	b     board
	moves []rect
	score int
}

func runAnaOnly(n int, outFile string) {
	master := rand.New(rand.NewSource(time.Now().UnixNano()))
	seeds := make([]int64, n)
	for i := range seeds {
		seeds[i] = master.Int63()
	}
	logs := make([]gameLog, n)

	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())
	start := time.Now()

	for g := 0; g < n; g++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int, seed int64) {
			defer func() { <-sem; wg.Done() }()
			rng := rand.New(rand.NewSource(seed))
			b := genBoard(rng)
			score, moves := playMCGreedyAllCandsTrace(b)
			logs[idx] = gameLog{idx, b, moves, score}
		}(g, seeds[g])
	}
	wg.Wait()
	elapsed := time.Since(start)

	if outFile != "" {
		f, err := os.Create(outFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "file create error: %v\n", err)
		} else {
			bw := bufio.NewWriterSize(f, 4<<20)
			for _, gl := range logs {
				fmt.Fprintf(bw, "game %d score=%d\n", gl.idx+1, gl.score)
				fmt.Fprintf(bw, "board:")
				for i := 0; i < nRows; i++ {
					for j := 0; j < nCols; j++ {
						fmt.Fprintf(bw, " %d", gl.b[i][j])
					}
				}
				fmt.Fprintf(bw, "\nmoves %d:", len(gl.moves))
				for _, m := range gl.moves {
					fmt.Fprintf(bw, " %d %d %d %d |", m.r1, m.c1, m.r2, m.c2)
				}
				fmt.Fprintf(bw, "\n\n")
			}
			bw.Flush()
			f.Close()
			fmt.Printf("Game log saved: %s\n\n", outFile)
		}
	}

	scores := make([]int, n)
	for i, gl := range logs {
		scores[i] = gl.score
	}

	minS, maxS, sum := scores[0], scores[0], 0
	for _, s := range scores {
		sum += s
		if s < minS {
			minS = s
		}
		if s > maxS {
			maxS = s
		}
	}
	mean := float64(sum) / float64(n)
	var ss float64
	for _, s := range scores {
		d := float64(s) - mean
		ss += d * d
	}
	sd := math.Sqrt(ss / float64(n))
	se := sd / math.Sqrt(float64(n))

	fmt.Printf("=== AnA Benchmark (%d games, %d x %d) ===\n", n, nRows, nCols)
	fmt.Printf("Mean    : %.4f\nSD      : %.4f\nSE      : %.4f\nMin     : %d\nMax     : %d\n",
		mean, sd, se, minS, maxS)
	fmt.Printf("Elapsed : %v\n\n", elapsed.Round(time.Millisecond))

	const binSize = 5
	binMin := (minS / binSize) * binSize
	binMax := ((maxS + binSize - 1) / binSize) * binSize
	nBins := (binMax-binMin)/binSize + 1
	bins := make([]int, nBins)
	for _, s := range scores {
		bins[(s-binMin)/binSize]++
	}
	maxBin := 0
	for _, c := range bins {
		if c > maxBin {
			maxBin = c
		}
	}
	const barWidth = 40
	fmt.Printf("Score distribution (bin=%d)\n", binSize)
	fmt.Printf("%-10s |%-*s  count\n", "score", barWidth, "")
	fmt.Printf("----------+%s\n", strings.Repeat("-", barWidth+8))
	for i, c := range bins {
		lo := binMin + i*binSize
		hi := lo + binSize - 1
		bar := int(float64(c) / float64(maxBin) * barWidth)
		pct := float64(c) / float64(n) * 100
		fmt.Printf("%3d-%3d   |%s  %4d (%.1f%%)\n",
			lo, hi,
			strings.Repeat("#", bar)+strings.Repeat(" ", barWidth-bar),
			c, pct)
	}
	fmt.Printf("----------+%s\n", strings.Repeat("-", barWidth+8))
}

// ── Perfect game search ───────────────────────────────────────────────────────

func savePerfectGame(b board, moves []rect, filename string) {
	f, err := os.Create(filename)
	if err != nil {
		fmt.Fprintf(os.Stderr, "file create error: %v\n", err)
		return
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	fmt.Fprintf(w, "# Apple Game Perfect Score (%d/%d)\n", nTotal, nTotal)
	fmt.Fprintf(w, "# Board: %d rows x %d cols\n# Moves: %d\n\n", nRows, nCols, len(moves))
	fmt.Fprintf(w, "[board]\n")
	for i := 0; i < nRows; i++ {
		for j := 0; j < nCols; j++ {
			if j > 0 {
				fmt.Fprintf(w, " ")
			}
			fmt.Fprintf(w, "%d", b[i][j])
		}
		fmt.Fprintf(w, "\n")
	}
	fmt.Fprintf(w, "\n[moves]  # r1 c1 r2 c2  (0-indexed, inclusive)\n")
	for _, m := range moves {
		fmt.Fprintf(w, "%d %d %d %d\n", m.r1, m.c1, m.r2, m.c2)
	}
	w.Flush()
	fmt.Printf("  Saved: %s\n", filename)
}

// ── SL data generation (AnA trajectories + all candidates) ───────────────────
//
// Format: 171 bytes per record
//   [0..169]  board state b2 AFTER applying candidate (uint8, values 0-9)
//   [170]     playGreedy(b2) score (uint8, 0-170)
//
// At every AnA step, ALL valid candidates are enumerated. For each candidate c:
//   b2     = board after applying c
//   label  = playGreedy(b2)   (Greedy-lookahead value of b2)
// The board then advances along the AnA path (best candidate).
//
// This (b2, playGreedy(b2)) pairing lets the model learn to replicate Greedy
// lookahead but from the AnA board distribution — fixing the distribution shift
// that caused high RMSE when training on Greedy-only trajectories.

type slRecord struct {
	board [nRows * nCols]uint8
	score uint8
}

// collectSLRecords plays one AnA game and returns all (b2, greedy) pairs.
func collectSLRecords(b board) []slRecord {
	var val, cnt ps2d
	var cands [maxRects]rect
	var records []slRecord

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

		for i := 0; i < nCands; i++ {
			b2 := b
			applyRect(&b2, cands[i])
			gs := playGreedy(b2)

			// Record (b2_flat, greedy_score)
			var rec slRecord
			for r := 0; r < nRows; r++ {
				for c := 0; c < nCols; c++ {
					rec.board[r*nCols+c] = uint8(b2[r][c])
				}
			}
			rec.score = uint8(gs)
			records = append(records, rec)

			if gs > bestScore {
				bestScore = gs
				bestR = cands[i]
			}
		}

		applyRect(&b, bestR)
	}
	return records
}

// runGenSL generates SL training data using AnA trajectories.
// Each board contributes (steps × candidates) records.
func runGenSL(numGames int, filename string) {
	fmt.Printf("Generating SL data: %d AnA games -> %s\n", numGames, filename)
	fmt.Printf("Format: 171 bytes/record  [board_170 | greedy_score_1]\n\n")

	f, err := os.Create(filename)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot create %s: %v\n", filename, err)
		os.Exit(1)
	}
	defer f.Close()

	bw := bufio.NewWriterSize(f, 16<<20)

	// Pre-generate seeds for reproducibility.
	master := rand.New(rand.NewSource(12345))
	seeds := make([]int64, numGames)
	for i := range seeds {
		seeds[i] = master.Int63()
	}

	// Per-game results arrive out of order; buffer them for writing.
	type result struct {
		idx     int
		records []slRecord
	}

	// resultCh 버퍼를 넉넉하게 잡아도 되지만,
	// 메인이 드레인하는 동안 워커가 막히지 않도록
	// 런치 루프 자체를 별도 고루틴에서 실행한다.
	// (메인이 sem 전송에서 블록된 채로 resultCh를 읽지 못하는 데드락 방지)
	resultCh := make(chan result, runtime.NumCPU()*4)
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())

	// 워커 런치 + 완료 후 채널 닫기를 고루틴에서 실행.
	go func() {
		for g := 0; g < numGames; g++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int, seed int64) {
				defer func() { <-sem; wg.Done() }()
				rng := rand.New(rand.NewSource(seed))
				b := genBoard(rng)
				resultCh <- result{idx, collectSLRecords(b)}
			}(g, seeds[g])
		}
		wg.Wait()
		close(resultCh)
	}()

	// Drain results and write records.
	var (
		totalRecords int64
		doneGames    int64
		row          [nRows*nCols + 1]byte
	)
	start := time.Now()
	nextReport := int64(100)

	for res := range resultCh {
		for _, rec := range res.records {
			copy(row[:nRows*nCols], rec.board[:])
			row[nRows*nCols] = rec.score
			if _, werr := bw.Write(row[:]); werr != nil {
				fmt.Fprintf(os.Stderr, "write error: %v\n", werr)
				os.Exit(1)
			}
		}
		atomic.AddInt64(&totalRecords, int64(len(res.records)))
		done := atomic.AddInt64(&doneGames, 1)

		if done >= nextReport || done == int64(numGames) {
			elapsed := time.Since(start)
			eta := time.Duration(0)
			if done < int64(numGames) {
				eta = elapsed * time.Duration(int64(numGames)-done) / time.Duration(done)
			}
			tr := atomic.LoadInt64(&totalRecords)
			fmt.Printf("  [%5d/%d]  records: %9d  elapsed: %v  ETA: %v\n",
				done, numGames,
				tr,
				elapsed.Round(time.Millisecond),
				eta.Round(time.Second))
			nextReport = done + 100
		}
	}

	if err = bw.Flush(); err != nil {
		fmt.Fprintf(os.Stderr, "flush error: %v\n", err)
		os.Exit(1)
	}

	elapsed := time.Since(start)
	info, _ := f.Stat()
	tr := atomic.LoadInt64(&totalRecords)

	fmt.Printf("\n=== SL Data Generation Complete ===\n")
	fmt.Printf("  Games   : %d\n", numGames)
	fmt.Printf("  Records : %d\n", tr)
	fmt.Printf("  Avg     : %.1f records/game  (steps × candidates)\n",
		float64(tr)/float64(numGames))
	fmt.Printf("  File    : %s  (%.1f MB)\n", filename, float64(info.Size())/1e6)
	fmt.Printf("  Time    : %v\n", elapsed.Round(time.Millisecond))
}

func runFindPerfect(outFile string) {
	fmt.Printf("Searching for perfect score (%d) with AnA... (%d CPUs)\n", nTotal, runtime.NumCPU())
	fmt.Printf("Output: %s\n\n", outFile)

	type found struct {
		b     board
		moves []rect
	}
	ch := make(chan found, 1)

	var (
		mu       sync.Mutex
		attempts int64
	)
	start := time.Now()

	go func() {
		for {
			time.Sleep(5 * time.Second)
			mu.Lock()
			a := attempts
			mu.Unlock()
			if a == 0 {
				continue
			}
			fmt.Printf("  %d attempts, elapsed %v\n", a, time.Since(start).Round(time.Second))
			select {
			case <-ch:
				return
			default:
			}
		}
	}()

	numWorkers := runtime.NumCPU()
	for w := 0; w < numWorkers; w++ {
		go func(seed int64) {
			rng := rand.New(rand.NewSource(seed))
			for {
				select {
				case <-ch:
					return
				default:
				}
				b := genBoard(rng)
				score, moves := playMCGreedyAllCandsTrace(b)
				mu.Lock()
				attempts++
				mu.Unlock()
				if score == nTotal {
					select {
					case ch <- found{b, moves}:
					default:
					}
					return
				}
			}
		}(time.Now().UnixNano() + int64(w)*1000003)
	}

	result := <-ch
	elapsed := time.Since(start)
	mu.Lock()
	total := attempts
	mu.Unlock()

	fmt.Printf("\nPerfect score achieved!\n")
	fmt.Printf("  Attempts : %d\n", total)
	fmt.Printf("  Elapsed  : %v\n", elapsed.Round(time.Millisecond))
	fmt.Printf("  Moves    : %d\n\n", len(result.moves))

	for i := 0; i < nRows; i++ {
		for j := 0; j < nCols; j++ {
			fmt.Printf("%2d", result.b[i][j])
		}
		fmt.Println()
	}
	fmt.Println()
	savePerfectGame(result.b, result.moves, outFile)
}
