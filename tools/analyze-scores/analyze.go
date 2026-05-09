//go:build ignore
// +build ignore

package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
)

func main() {
	// Read games_1000.txt
	f, err := os.Open("data/raw/games_1000.txt")
	if err != nil {
		fmt.Fprintf(os.Stderr, "파일 열기 실패: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	// Extract scores
	scoreMap := make(map[int]int)
	scanner := bufio.NewScanner(f)
	re := regexp.MustCompile(`game \d+ score=(\d+)`)

	for scanner.Scan() {
		line := scanner.Text()
		matches := re.FindStringSubmatch(line)
		if len(matches) > 1 {
			score, _ := strconv.Atoi(matches[1])
			scoreMap[score]++
		}
	}

	// Sort scores
	var scores []int
	for s := range scoreMap {
		scores = append(scores, s)
	}
	sort.Ints(scores)

	// Write output
	out, err := os.Create("reports/score_distribution.txt")
	if err != nil {
		fmt.Fprintf(os.Stderr, "파일 생성 실패: %v\n", err)
		os.Exit(1)
	}
	defer out.Close()

	bw := bufio.NewWriter(out)

	total := 0
	for _, count := range scoreMap {
		total += count
	}

	minS, maxS := scores[0], scores[len(scores)-1]

	fmt.Fprintf(bw, "점수별 분포 (총 %d게임)\n", total)
	fmt.Fprintf(bw, "════════════════════════\n\n")
	fmt.Fprintf(bw, "최소: %d, 최대: %d, 종류: %d\n\n", minS, maxS, len(scores))

	fmt.Fprintf(bw, "점수  횟수  백분율\n")
	fmt.Fprintf(bw, "──── ──── ────────\n")

	for _, s := range scores {
		count := scoreMap[s]
		pct := float64(count) / float64(total) * 100
		fmt.Fprintf(bw, "%3d  %4d  %6.2f%%\n", s, count, pct)
	}

	bw.Flush()

	// Print to stdout
	fmt.Printf("✓ 점수별 분포 저장: reports/score_distribution.txt\n\n")
	fmt.Printf("점수별 분포 (총 %d게임)\n", total)
	fmt.Printf("════════════════════════\n\n")
	fmt.Printf("최소: %d, 최대: %d, 종류: %d\n\n", minS, maxS, len(scores))
	fmt.Printf("점수  횟수  백분율\n")
	fmt.Printf("──── ──── ────────\n")
	for _, s := range scores {
		count := scoreMap[s]
		pct := float64(count) / float64(total) * 100
		fmt.Printf("%3d  %4d  %6.2f%%\n", s, count, pct)
	}
}
