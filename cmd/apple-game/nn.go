//go:build nn

// nn.go – ONNX-based value network integration
//
// NN 평가 흐름:
//   board (10×17 int8) → float32 /9 정규화 → ONNX 추론 → 예측값 × 170 → int
//
// 사용 알고리즘: playMCNNAllCands
//   안A(MC-Greedy-AllCands)와 동일하지만 playGreedy 대신 NN으로 후보 평가.

package main

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// ── 전역 초기화 (한 번만) ─────────────────────────────────────────────────────

var (
	nnOnce      sync.Once
	nnModelPath string
	nnLibPath   string
)

// InitNN 은 ONNX Runtime 환경을 초기화한다 (프로세스당 1회).
// modelPath: model.onnx 경로
// libPath  : onnxruntime.dll (Windows) / libonnxruntime.so (Linux) 경로
func InitNN(modelPath, libPath string) error {
	var initErr error
	nnOnce.Do(func() {
		nnModelPath = modelPath
		nnLibPath = libPath
		ort.SetSharedLibraryPath(libPath)
		initErr = ort.InitializeEnvironment()
	})
	return initErr
}

// ── 고루틴별 세션 ──────────────────────────────────────────────────────────────

// nnCtx 는 ONNX 세션과 입출력 버퍼를 묶은 고루틴-로컬 컨텍스트다.
// ONNX Runtime 세션은 스레드 안전하지 않으므로 고루틴마다 별도 생성.
type nnCtx struct {
	session *ort.AdvancedSession
	in      []float32
	out     []float32
	inT     *ort.Tensor[float32]
	outT    *ort.Tensor[float32]
}

// newNNCtx 는 새 ONNX 세션을 생성한다. InitNN 이후에 호출해야 한다.
func newNNCtx() (*nnCtx, error) {
	in := make([]float32, nRows*nCols)
	out := make([]float32, 1)

	inT, err := ort.NewTensor(ort.NewShape(1, 1, nRows, nCols), in)
	if err != nil {
		return nil, fmt.Errorf("input tensor: %w", err)
	}
	outT, err := ort.NewTensor(ort.NewShape(1), out)
	if err != nil {
		return nil, fmt.Errorf("output tensor: %w", err)
	}

	sess, err := ort.NewAdvancedSession(
		nnModelPath,
		[]string{"board"}, []string{"value"},
		[]ort.ArbitraryTensor{inT}, []ort.ArbitraryTensor{outT},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("new session: %w", err)
	}
	return &nnCtx{session: sess, in: in, out: out, inT: inT, outT: outT}, nil
}

// destroy 는 세션과 텐서 메모리를 해제한다.
func (c *nnCtx) destroy() {
	c.session.Destroy()
	c.inT.Destroy()
	c.outT.Destroy()
}

// eval 은 보드 b 에 대해 NN 추론을 실행하고 예측 제거 사과 수를 반환한다.
func (c *nnCtx) eval(b *board) int {
	for i := 0; i < nRows; i++ {
		for j := 0; j < nCols; j++ {
			c.in[i*nCols+j] = float32(b[i][j]) / 9.0
		}
	}
	if err := c.session.Run(); err != nil {
		panic("nnCtx.eval: " + err.Error())
	}
	v := c.out[0] * float32(nTotal)
	if v < 0 {
		v = 0
	}
	if v > nTotal {
		v = nTotal
	}
	return int(v + 0.5)
}

// ── nnCtxV2: ResNet + aux-feature model (train_sl / train_rl output) ──────────
//
// ONNX 입력:
//   "board"  shape [1, 1, 10, 17] float32  (값 /9 정규화)
//   "aux"    shape [1, 3]          float32  (3개 보조 특징)
// ONNX 출력:
//   "score"  shape [1]             float32  (정규화된 Greedy 점수, ×170 → int)
//
// 보조 특징 (Python train_sl.py 의 compute_aux 와 완전히 일치):
//   aux[0] = nz_count / 170
//   aux[1] = nz_count * (nz_count - 1) / 170²
//   aux[2] = cell_sum / 1530

type nnCtxV2 struct {
	session  *ort.AdvancedSession
	inBoard  []float32 // [1 * 1 * 10 * 17]
	inAux    []float32 // [1 * 3]
	out      []float32 // [1]
	inBoardT *ort.Tensor[float32]
	inAuxT   *ort.Tensor[float32]
	outT     *ort.Tensor[float32]
}

// newNNCtxV2 creates a session for the ResNet+aux model.
// modelPath must point to the ONNX file exported by train_sl.py / train_rl.py.
// InitNN must have been called first.
func newNNCtxV2(modelPath string) (*nnCtxV2, error) {
	inBoard := make([]float32, nRows*nCols)
	inAux := make([]float32, 3)
	out := make([]float32, 1)

	inBoardT, err := ort.NewTensor(ort.NewShape(1, 1, nRows, nCols), inBoard)
	if err != nil {
		return nil, fmt.Errorf("v2 board tensor: %w", err)
	}
	inAuxT, err := ort.NewTensor(ort.NewShape(1, 3), inAux)
	if err != nil {
		return nil, fmt.Errorf("v2 aux tensor: %w", err)
	}
	outT, err := ort.NewTensor(ort.NewShape(1), out)
	if err != nil {
		return nil, fmt.Errorf("v2 output tensor: %w", err)
	}

	sess, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"board", "aux"}, []string{"score"},
		[]ort.ArbitraryTensor{inBoardT, inAuxT},
		[]ort.ArbitraryTensor{outT},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("v2 session: %w", err)
	}
	return &nnCtxV2{
		session: sess, inBoard: inBoard, inAux: inAux, out: out,
		inBoardT: inBoardT, inAuxT: inAuxT, outT: outT,
	}, nil
}

func (c *nnCtxV2) destroy() {
	c.session.Destroy()
	c.inBoardT.Destroy()
	c.inAuxT.Destroy()
	c.outT.Destroy()
}

// eval evaluates the board and returns predicted removable-apple count (0–170).
// Aux features mirror Python's compute_aux exactly.
func (c *nnCtxV2) eval(b *board) int {
	// Fill board input
	var nz, cellSum int
	for i := 0; i < nRows; i++ {
		for j := 0; j < nCols; j++ {
			v := b[i][j]
			c.inBoard[i*nCols+j] = float32(v) / 9.0
			if v > 0 {
				nz++
				cellSum += int(v)
			}
		}
	}
	// Compute aux features (match Python compute_aux)
	nzF := float32(nz) / float32(nTotal) // nz_count / 170
	c.inAux[0] = nzF
	c.inAux[1] = float32(nz) * float32(nz-1) / (float32(nTotal) * float32(nTotal))
	c.inAux[2] = float32(cellSum) / (float32(nTotal) * 9.0) // cell_sum / 1530

	if err := c.session.Run(); err != nil {
		panic("nnCtxV2.eval: " + err.Error())
	}
	v := c.out[0] * float32(nTotal)
	if v < 0 {
		v = 0
	}
	if v > nTotal {
		v = nTotal
	}
	return int(v + 0.5)
}

// playModelAnA is AnA with the NN as evaluator instead of Greedy.
// Drop-in replacement: same structure as playMCGreedyAllCands,
// but nc.eval(&b2) replaces playGreedy(b2).
func playModelAnA(b board, nc *nnCtxV2) int {
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
		for i := 0; i < nCands; i++ {
			b2 := b
			applyRect(&b2, cands[i])
			score := nc.eval(&b2) // ← NN instead of Greedy
			if score > bestScore {
				bestScore = score
				bestR = cands[i]
			}
		}
		removed += applyRect(&b, bestR)
	}
	return removed
}

// ── 안A-NN 알고리즘 (구 단일입력 모델용) ──────────────────────────────────────────
//
// 안A (playMCGreedyAllCands) 와 동일하지만
// 후보 평가를 playGreedy 대신 NN 추론으로 수행한다.
// → 목적: NN이 Greedy보다 더 나은 평가 함수 역할을 하는지 검증.

func playMCNNAllCands(b board, nc *nnCtx) int {
	var val, cnt ps2d
	var cb candBuf
	removed := 0

	for {
		buildPS(&b, &val, &cnt)

		nCands := 0
		for r1 := 0; r1 < nRows; r1++ {
			for r2 := r1; r2 < nRows; r2++ {
				for c1 := 0; c1 < nCols; c1++ {
					for c2 := c1; c2 < nCols; c2++ {
						if qps(&val, r1, c1, r2, c2) == 10 {
							cb.rects[nCands] = rect{int8(r1), int8(c1), int8(r2), int8(c2)}
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
			applyRect(&b2, cb.rects[i])
			score := nc.eval(&b2) // NN 평가
			if score > bestScore {
				bestScore = score
				bestR = cb.rects[i]
			}
		}
		removed += applyRect(&b, bestR)
	}
	return removed
}
