package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"sync"
	"time"
)

// ── Greedy-only benchmark ─────────────────────────────────────────────────────

func runGreedyOnly(n int) {
	master := rand.New(rand.NewSource(time.Now().UnixNano()))
	seeds := make([]int64, n)
	for i := range seeds {
		seeds[i] = master.Int63()
	}
	scores := make([]int, n)
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())
	start := time.Now()
	for g := 0; g < n; g++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int, seed int64) {
			defer func() { <-sem; wg.Done() }()
			rng := rand.New(rand.NewSource(seed))
			scores[idx] = playGreedy(genBoard(rng))
		}(g, seeds[g])
	}
	wg.Wait()
	elapsed := time.Since(start)

	sum, minS, maxS := 0, nTotal+1, 0
	for _, s := range scores {
		sum += s
		if s < minS {
			minS = s
		}
		if s > maxS {
			maxS = s
		}
	}
	fmt.Printf("=== Greedy (%d games) ===\n", n)
	fmt.Printf("Average : %.2f / %d\n", float64(sum)/float64(n), nTotal)
	fmt.Printf("Min     : %d\n", minS)
	fmt.Printf("Max     : %d\n", maxS)
	fmt.Printf("Elapsed : %v\n", elapsed)
}

// ── AnA vs AnA+B comparison ───────────────────────────────────────────────────

func runComparison(n, _ int) {
	master := rand.New(rand.NewSource(time.Now().UnixNano()))
	boards := make([]board, n)
	for i := range boards {
		boards[i] = genBoard(master)
	}

	type result struct{ a, ab int }
	results := make([]result, n)

	var (
		mu        sync.Mutex
		completed int
		totalDur  time.Duration
	)
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())

	fmt.Printf("AnA  vs  AnA+B(2-ply)  —  %d boards  —  %d CPUs\n\n", n, runtime.NumCPU())
	overallStart := time.Now()

	for i := 0; i < n; i++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int) {
			defer func() { <-sem; wg.Done() }()
			t0 := time.Now()
			b := boards[idx]
			ma := playMCGreedyAllCands(b)
			mab := playMCGreedyAllCands2Ply(b)
			dur := time.Since(t0)

			mu.Lock()
			completed++
			totalDur += dur
			eta := (totalDur / time.Duration(completed)) * time.Duration(n-completed)
			best := ma
			if mab > best {
				best = mab
			}
			as, abs := " ", " "
			if ma == best {
				as = "*"
			}
			if mab == best {
				abs = "*"
			}
			diff := mab - ma
			sign := "+"
			if diff < 0 {
				sign = ""
			}
			fmt.Printf("[%2d/%d]  %3d%s   %3d%s  (%s%d)   %6.1fs  ETA %v\n",
				completed, n, ma, as, mab, abs, sign, diff,
				dur.Seconds(), eta.Round(time.Second))
			mu.Unlock()
			results[idx] = result{ma, mab}
		}(i)
	}
	wg.Wait()

	type stat struct{ sum, min, max int }
	st := [2]stat{{0, nTotal + 1, 0}, {0, nTotal + 1, 0}}
	wins := [2]int{}
	for _, r := range results {
		vals := [2]int{r.a, r.ab}
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
	names := [2]string{"AnA", "AnA+B(2-ply)"}
	printStatTable(names[:], st[0].sum, st[1].sum, st[0].min, st[1].min, st[0].max, st[1].max, wins[0], wins[1], n)
	fmt.Printf("  AnA+B gain : %+.2f\n", float64(st[1].sum-st[0].sum)/float64(n))
	fmt.Printf("  Total time : %v\n", time.Since(overallStart).Round(time.Millisecond))
	fmt.Printf("══════════════════════════════════════════════════════\n")
}

// ── NIA vs AnA comparison (100 boards) ───────────────────────────────────────

func runNIAvsAnA(n int) {
	master := rand.New(rand.NewSource(time.Now().UnixNano()))
	boards := make([]board, n)
	for i := range boards {
		boards[i] = genBoard(master)
	}

	type result struct{ ana, nia int }
	results := make([]result, n)

	var (
		mu        sync.Mutex
		completed int
		totalDur  time.Duration
	)
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())

	fmt.Printf("NIA  vs  AnA  —  %d boards  —  %d CPUs\n\n", n, runtime.NumCPU())
	overallStart := time.Now()

	for i := 0; i < n; i++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int) {
			defer func() { <-sem; wg.Done() }()
			t0 := time.Now()
			b := boards[idx]
			ana := playMCGreedyAllCands(b)
			nia := playNIA(b)
			dur := time.Since(t0)

			mu.Lock()
			completed++
			totalDur += dur
			eta := (totalDur / time.Duration(completed)) * time.Duration(n-completed)
			best := ana
			if nia > best {
				best = nia
			}
			as, ns := " ", " "
			if ana == best {
				as = "*"
			}
			if nia == best {
				ns = "*"
			}
			diff := nia - ana
			sign := "+"
			if diff < 0 {
				sign = ""
			}
			fmt.Printf("[%2d/%d]  AnA:%3d%s  NIA:%3d%s  (%s%d)   %6.1fs  ETA %v\n",
				completed, n, ana, as, nia, ns, sign, diff,
				dur.Seconds(), eta.Round(time.Second))
			mu.Unlock()
			results[idx] = result{ana, nia}
		}(i)
	}
	wg.Wait()

	type stat struct{ sum, min, max int }
	st := [2]stat{{0, nTotal + 1, 0}, {0, nTotal + 1, 0}}
	wins := [2]int{}
	for _, r := range results {
		vals := [2]int{r.ana, r.nia}
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
	names := [2]string{"AnA", "NIA"}
	printStatTable(names[:], st[0].sum, st[1].sum, st[0].min, st[1].min, st[0].max, st[1].max, wins[0], wins[1], n)
	fmt.Printf("  NIA gain   : %+.2f\n", float64(st[1].sum-st[0].sum)/float64(n))
	fmt.Printf("  Total time : %v\n", time.Since(overallStart).Round(time.Millisecond))
	fmt.Printf("══════════════════════════════════════════════════════\n")
}

// ── RAnA vs AnA comparison ────────────────────────────────────────────────────

func runRAnAvsAnA(n, K int) {
	master := rand.New(rand.NewSource(time.Now().UnixNano()))
	boards := make([]board, n)
	seeds := make([]int64, n)
	for i := range boards {
		boards[i] = genBoard(master)
		seeds[i] = master.Int63()
	}

	type result struct{ ana, rana int }
	results := make([]result, n)

	var (
		mu        sync.Mutex
		completed int
		totalDur  time.Duration
	)
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())

	fmt.Printf("RAnA(K=%d)  vs  AnA  —  %d boards  —  %d CPUs\n\n", K, n, runtime.NumCPU())
	overallStart := time.Now()

	for i := 0; i < n; i++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int, seed int64) {
			defer func() { <-sem; wg.Done() }()
			t0 := time.Now()
			b := boards[idx]
			rng := rand.New(rand.NewSource(seed))
			ana := playMCGreedyAllCands(b)
			rana := playAnARandBestOf(b, K, rng)
			dur := time.Since(t0)

			mu.Lock()
			completed++
			totalDur += dur
			eta := (totalDur / time.Duration(completed)) * time.Duration(n-completed)
			best := ana
			if rana > best {
				best = rana
			}
			as, rs := " ", " "
			if ana == best {
				as = "*"
			}
			if rana == best {
				rs = "*"
			}
			diff := rana - ana
			sign := "+"
			if diff < 0 {
				sign = ""
			}
			fmt.Printf("[%2d/%d]  AnA:%3d%s  RAnA:%3d%s  (%s%d)   %6.1fs  ETA %v\n",
				completed, n, ana, as, rana, rs, sign, diff,
				dur.Seconds(), eta.Round(time.Second))
			mu.Unlock()
			results[idx] = result{ana, rana}
		}(i, seeds[i])
	}
	wg.Wait()

	type stat struct{ sum, min, max int }
	st := [2]stat{{0, nTotal + 1, 0}, {0, nTotal + 1, 0}}
	wins := [2]int{}
	for _, r := range results {
		vals := [2]int{r.ana, r.rana}
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
	names := [2]string{"AnA", fmt.Sprintf("RAnA(K=%d)", K)}
	printStatTable(names[:], st[0].sum, st[1].sum, st[0].min, st[1].min, st[0].max, st[1].max, wins[0], wins[1], n)
	fmt.Printf("  RAnA gain  : %+.2f\n", float64(st[1].sum-st[0].sum)/float64(n))
	fmt.Printf("  Total time : %v\n", time.Since(overallStart).Round(time.Millisecond))
	fmt.Printf("══════════════════════════════════════════════════════\n")
}

// ── Random Play benchmark ─────────────────────────────────────────────────────

func runRandomBench(n int, outFile string) {
	master := rand.New(rand.NewSource(time.Now().UnixNano()))
	seeds := make([]int64, n)
	for i := range seeds {
		seeds[i] = master.Int63()
	}
	scores := make([]int, n)

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
			var rb rolloutBuf
			scores[idx] = playRandom(b, rng, &rb)
		}(g, seeds[g])
	}
	wg.Wait()
	elapsed := time.Since(start)

	if outFile != "" {
		writeScores(outFile, scores)
	}
	printBenchStats("Random Play", n, scores, elapsed)
	if outFile != "" {
		fmt.Printf("Scores saved  : %s (%d scores)\n", outFile, n)
	}
}

// ── Greedy benchmark ──────────────────────────────────────────────────────────

func runGreedyBench(n int, outFile string) {
	master := rand.New(rand.NewSource(time.Now().UnixNano()))
	seeds := make([]int64, n)
	for i := range seeds {
		seeds[i] = master.Int63()
	}
	scores := make([]int, n)

	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())
	start := time.Now()

	for g := 0; g < n; g++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int, seed int64) {
			defer func() { <-sem; wg.Done() }()
			rng := rand.New(rand.NewSource(seed))
			scores[idx] = playGreedy(genBoard(rng))
		}(g, seeds[g])
	}
	wg.Wait()
	elapsed := time.Since(start)

	if outFile != "" {
		writeScores(outFile, scores)
	}
	printBenchStats("Greedy", n, scores, elapsed)
	if outFile != "" {
		fmt.Printf("Scores saved  : %s (%d scores)\n", outFile, n)
	}
}

// ── NN comparison (requires -tags nn build) ───────────────────────────────────

func runNNComparison(n int, modelPath, libPath string) {
	if err := InitNN(modelPath, libPath); err != nil {
		fmt.Fprintf(os.Stderr, "ONNX init error: %v\n", err)
		os.Exit(1)
	}

	master := rand.New(rand.NewSource(time.Now().UnixNano()))
	boards := make([]board, n)
	for i := range boards {
		boards[i] = genBoard(master)
	}

	type result struct{ a, nn int }
	results := make([]result, n)

	var (
		mu        sync.Mutex
		completed int
		totalDur  time.Duration
	)
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())

	fmt.Printf("AnA  vs  AnA-NN  —  %d boards  —  %d CPUs\n  model: %s\n\n", n, runtime.NumCPU(), modelPath)
	overallStart := time.Now()

	for i := 0; i < n; i++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int) {
			defer func() { <-sem; wg.Done() }()
			nc, err := newNNCtx()
			if err != nil {
				fmt.Fprintf(os.Stderr, "session error: %v\n", err)
				os.Exit(1)
			}
			defer nc.destroy()

			t0 := time.Now()
			b := boards[idx]
			ma := playMCGreedyAllCands(b)
			mn := playMCNNAllCands(b, nc)
			dur := time.Since(t0)

			mu.Lock()
			completed++
			totalDur += dur
			eta := (totalDur / time.Duration(completed)) * time.Duration(n-completed)
			best := ma
			if mn > best {
				best = mn
			}
			as, ns := " ", " "
			if ma == best {
				as = "*"
			}
			if mn == best {
				ns = "*"
			}
			diff := mn - ma
			sign := "+"
			if diff < 0 {
				sign = ""
			}
			fmt.Printf("[%2d/%d]  %3d%s   %3d%s  (%s%d)   %6.1fs  ETA %v\n",
				completed, n, ma, as, mn, ns, sign, diff,
				dur.Seconds(), eta.Round(time.Second))
			mu.Unlock()
			results[idx] = result{ma, mn}
		}(i)
	}
	wg.Wait()

	type stat struct{ sum, min, max int }
	st := [2]stat{{0, nTotal + 1, 0}, {0, nTotal + 1, 0}}
	wins := [2]int{}
	for _, r := range results {
		vals := [2]int{r.a, r.nn}
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
	names := [2]string{"AnA", "AnA-NN"}
	printStatTable(names[:], st[0].sum, st[1].sum, st[0].min, st[1].min, st[0].max, st[1].max, wins[0], wins[1], n)
	fmt.Printf("  NN gain    : %+.2f\n", float64(st[1].sum-st[0].sum)/float64(n))
	fmt.Printf("  Total time : %v\n", time.Since(overallStart).Round(time.Millisecond))
	fmt.Printf("══════════════════════════════════════════════════════\n")
}

// ── ModelAnA comparison (requires -tags nn build) ────────────────────────────
//
// Benchmarks the new ResNet+aux model (train_sl.py / train_rl.py output)
// against AnA on n random boards.
// Uses playModelAnA (nnCtxV2) which feeds both board and auxiliary features.

func runModelAnAvsAnA(n int, modelPath, libPath string) {
	if err := InitNN(modelPath, libPath); err != nil {
		fmt.Fprintf(os.Stderr, "ONNX init error: %v\n", err)
		os.Exit(1)
	}

	master := rand.New(rand.NewSource(time.Now().UnixNano()))
	boards := make([]board, n)
	for i := range boards {
		boards[i] = genBoard(master)
	}

	type result struct{ ana, model int }
	results := make([]result, n)

	var (
		mu        sync.Mutex
		completed int
		totalDur  time.Duration
	)
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())

	fmt.Printf("AnA  vs  ModelAnA  —  %d boards  —  %d CPUs\n  model: %s\n\n",
		n, runtime.NumCPU(), modelPath)
	overallStart := time.Now()

	for i := 0; i < n; i++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int) {
			defer func() { <-sem; wg.Done() }()
			nc, err := newNNCtxV2(modelPath)
			if err != nil {
				fmt.Fprintf(os.Stderr, "session error: %v\n", err)
				os.Exit(1)
			}
			defer nc.destroy()

			t0    := time.Now()
			b     := boards[idx]
			ma    := playMCGreedyAllCands(b)
			mm    := playModelAnA(b, nc)
			dur   := time.Since(t0)

			mu.Lock()
			completed++
			totalDur += dur
			eta  := (totalDur / time.Duration(completed)) * time.Duration(n-completed)
			best := ma
			if mm > best {
				best = mm
			}
			as, ms := " ", " "
			if ma == best {
				as = "*"
			}
			if mm == best {
				ms = "*"
			}
			diff := mm - ma
			sign := "+"
			if diff < 0 {
				sign = ""
			}
			fmt.Printf("[%2d/%d]  AnA %3d%s  Model %3d%s  (%s%d)   %6.1fs  ETA %v\n",
				completed, n, ma, as, mm, ms, sign, diff,
				dur.Seconds(), eta.Round(time.Second))
			mu.Unlock()
			results[idx] = result{ma, mm}
		}(i)
	}
	wg.Wait()

	type stat struct{ sum, min, max int }
	st := [2]stat{{0, nTotal + 1, 0}, {0, nTotal + 1, 0}}
	wins := [2]int{}
	for _, r := range results {
		vals := [2]int{r.ana, r.model}
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
	names := [2]string{"AnA", "ModelAnA"}
	printStatTable(names[:],
		st[0].sum, st[1].sum,
		st[0].min, st[1].min,
		st[0].max, st[1].max,
		wins[0], wins[1], n)
	fmt.Printf("  Model gain : %+.2f\n", float64(st[1].sum-st[0].sum)/float64(n))
	fmt.Printf("  Total time : %v\n", time.Since(overallStart).Round(time.Millisecond))
	fmt.Printf("══════════════════════════════════════════════════════\n")
}

// ── Shared helpers ────────────────────────────────────────────────────────────

func printStatTable(names []string, sum0, sum1, min0, min1, max0, max1, wins0, wins1, n int) {
	fmt.Printf("\n══════════════════════════════════════════════════════\n")
	fmt.Printf("  %-14s  Average    Min    Max   Wins\n", "Algorithm")
	fmt.Printf("  %-14s  -------   ----   ----   ----\n", "---------")
	fmt.Printf("  %-14s  %7.2f   %4d   %4d   %4d\n",
		names[0], float64(sum0)/float64(n), min0, max0, wins0)
	fmt.Printf("  %-14s  %7.2f   %4d   %4d   %4d\n",
		names[1], float64(sum1)/float64(n), min1, max1, wins1)
	fmt.Printf("──────────────────────────────────────────────────────\n")
}

func printBenchStats(label string, n int, scores []int, elapsed time.Duration) {
	minS, maxS := scores[0], scores[0]
	sum := 0
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

	var sumSq, sumAbs float64
	for _, s := range scores {
		d := float64(s) - mean
		sumSq += d * d
		sumAbs += math.Abs(d)
	}
	sd := math.Sqrt(sumSq / float64(n))
	mad := sumAbs / float64(n)

	sorted := make([]int, n)
	copy(sorted, scores)
	sort.Ints(sorted)

	var median float64
	if n%2 == 0 {
		median = float64(sorted[n/2-1]+sorted[n/2]) / 2
	} else {
		median = float64(sorted[n/2])
	}
	p25 := float64(sorted[n/4])
	p75 := float64(sorted[3*n/4])

	fmt.Printf("=== %s Benchmark (%d games, %d×%d) ===\n", label, n, nRows, nCols)
	fmt.Printf("Mean   : %.4f\nMedian : %.4f\nSD     : %.4f\nMAD    : %.4f\n",
		mean, median, sd, mad)
	fmt.Printf("Min    : %d\nMax    : %d\nP25    : %.4f\nP75    : %.4f\n",
		minS, maxS, p25, p75)
	fmt.Printf("Time   : %v\n", elapsed.Round(time.Millisecond))
}

func writeScores(outFile string, scores []int) {
	f, err := os.Create(outFile)
	if err != nil {
		return
	}
	defer f.Close()
	for _, s := range scores {
		fmt.Fprintf(f, "%d\n", s)
	}
}
