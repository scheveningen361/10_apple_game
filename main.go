// Apple Game Simulator
//
// Board : 10 rows x 17 cols = 170 apples (values 1-9)
// Goal  : remove as many apples as possible by selecting rectangles
//         whose non-zero sum equals exactly 10.
//
// Algorithms:
//   Greedy  : fewest non-zero apples first; tie-break by highest max value.
//   AnA     : all candidates + 1-ply greedy lookahead (no min-count filter).
//   AnA+B   : all candidates + 2-ply greedy lookahead.
//   NIA     : Non-Interference Algorithm — picks move maximising heuristic NIM.
//
// Usage examples:
//   go run .                         → AnA vs AnA+B, 100 boards
//   go run . -nia -n 100             → NIA vs AnA, 100 boards
//   go run . -ana-only -n 1000       → AnA benchmark
//   go run . -greedy-bench -n 100000 → Greedy large-scale benchmark
//   go run . -solver                 → stdin board → AnA moves stdout (macro)

package main

import (
	"flag"
	"fmt"
	"runtime"
)

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	n           := flag.Int("n", 100, "number of games / boards")
	rollouts    := flag.Int("rollouts", 10, "MC random rollouts (legacy comparison mode)")
	greedyOnly  := flag.Bool("greedy-only", false, "Greedy benchmark only")
	genGames    := flag.Int("gen", 0, "generate training data: number of games (0=off)")
	genOut      := flag.String("out", "training_data.bin", "output file for -gen")
	nnMode      := flag.Bool("nn", false, "AnA vs AnA-NN comparison (requires -tags nn build, old single-input model)")
	nnAnaMode   := flag.Bool("nn-ana", false, "AnA vs ModelAnA comparison (requires -tags nn build, ResNet+aux model)")
	modelPath   := flag.String("model", "model.onnx", "ONNX model file path")
	ortLib      := flag.String("ort-lib", "onnxruntime.dll", "ONNX Runtime shared library path")
	perfectMode := flag.Bool("perfect", false, "search for perfect-score board with AnA")
	perfectOut  := flag.String("perfect-out", "perfect_game.txt", "perfect game output file")
	anaOnly     := flag.Bool("ana-only", false, "AnA benchmark")
	anaOut      := flag.String("ana-out", "", "AnA game log output file (empty = no save)")
	greedyBench := flag.Bool("greedy-bench", false, "Greedy large-scale benchmark")
	greedyOut   := flag.String("greedy-out", "", "Greedy scores output file")
	randomBench := flag.Bool("random-bench", false, "Random play large-scale benchmark")
	randomOut   := flag.String("random-out", "", "Random play scores output file")
	solverMode  := flag.Bool("solver", false, "stdin board -> AnA moves stdout (macro integration)")
	niaMode     := flag.Bool("nia", false, "NIA vs AnA comparison")
	ranaMode    := flag.Bool("rana", false, "RAnA (Best-of-K Randomized AnA) vs AnA comparison")
	ranaK       := flag.Int("k", 3, "K for RAnA: number of random restarts")
	genSL       := flag.Int("gen-sl", 0, "generate SL training data: number of AnA games (0=off)")
	slOut       := flag.String("sl-out", "sl_data.bin", "output file for -gen-sl")
	flag.Parse()

	switch {
	case *solverMode:
		runSolverMode()
	case *genSL > 0:
		runGenSL(*genSL, *slOut)
	case *genGames > 0:
		fmt.Printf("Generating training data: %d Greedy games -> %s\n\n", *genGames, *genOut)
		runGenerate(*genGames, *genOut)
	case *greedyOnly:
		runGreedyOnly(*n)
	case *nnMode:
		runNNComparison(*n, *modelPath, *ortLib)
	case *nnAnaMode:
		runModelAnAvsAnA(*n, *modelPath, *ortLib)
	case *perfectMode:
		runFindPerfect(*perfectOut)
	case *greedyBench:
		runGreedyBench(*n, *greedyOut)
	case *randomBench:
		runRandomBench(*n, *randomOut)
	case *anaOnly:
		runAnaOnly(*n, *anaOut)
	case *niaMode:
		runNIAvsAnA(*n)
	case *ranaMode:
		runRAnAvsAnA(*n, *ranaK)
	default:
		runComparison(*n, *rollouts)
	}
}
