package main

import (
	"fmt"
	"os"
)

// runSolverMode reads 170 board values from stdin, runs AnA, prints moves to stdout.
//
// Input  (stdin) : v1 v2 ... v170  (space/newline separated)
// Output (stdout):
//   MOVES <n>
//   r1 c1 r2 c2
//   ...
//   SCORE <removed>
func runSolverMode() {
	var b board
	for i := 0; i < nTotal; i++ {
		var v int
		if _, err := fmt.Scan(&v); err != nil {
			fmt.Fprintf(os.Stderr, "stdin read error at cell %d: %v\n", i, err)
			os.Exit(1)
		}
		b[i/nCols][i%nCols] = int8(v)
	}

	score, moves := playMCGreedyAllCandsTrace(b)

	fmt.Printf("MOVES %d\n", len(moves))
	for _, m := range moves {
		fmt.Printf("%d %d %d %d\n", m.r1, m.c1, m.r2, m.c2)
	}
	fmt.Printf("SCORE %d\n", score)
}
