# Apple Game — 진행 현황

## 알고리즘 성능 비교 (100보드 기준)

| 알고리즘 | 평균 점수 | 최소 | 최대 | 비고 |
|--|--|--|--|--|
| Random | ~103 | - | - | 100K 게임 |
| Greedy | ~114 | - | - | 100K 게임 |
| AnA | 131.62 | 98 | 166 | 100보드 |
| RAnA(K=3) | **132.82** | 100 | 166 | 100보드, AnA 3× 시간 |
| NIA | 116.99 | 84 | 144 | 100보드, AnA 대비 -14점 |

- 만점: 170점 (전체 사과 제거)

## 파일 구조 (모듈화 완료)

```
D:\Documents\Github\10_apple_game\
├── main.go          — main() + 플래그 파싱만
├── board.go         — 타입, 상수, buildPS/qps/applyRect/genBoard
├── greedy.go        — playGreedy
├── mc.go            — rolloutBuf/candBuf, playMC계열, playRandom
├── ana.go           — playMCGreedyAllCands, playMCGreedyAllCandsTrace
├── nia.go           — computeNIM, playNIA (비간섭 알고리즘)
├── rana.go          — playGreedyRand, playAnARand, playAnARandBestOf
├── bench.go         — 모든 run*벤치마크 함수
│                      (runNIAvsAnA, runRAnAvsAnA, runModelAnAvsAnA 포함)
├── datagen.go       — 데이터 생성/저장, runAnaOnly, runFindPerfect
│                      + runGenSL (SL 데이터 생성, Phase 1)
├── solver.go        — runSolverMode (매크로 연동용)
├── nn.go            — ONNX 통합 (//go:build nn 태그)
│                      nnCtx (구 단일입력), nnCtxV2 (ResNet+aux)
│                      playMCNNAllCands, playModelAnA
├── nn_stub.go       — NN 없이 빌드할 때 스텁 (//go:build !nn)
├── train_sl.py      — Phase 2: SL 학습 스크립트 (ResNet+aux, MSE)
├── train_rl.py      — Phase 3: PPO RL 학습 스크립트 (자기 플레이)
└── apple_solver.exe — 빌드 결과물 (매크로용)
```

## 실행 명령 목록

```bash
# 기본: AnA vs AnA+B 비교 (100보드)
go run ./cmd/apple-game

# NIA vs AnA (100보드)
go run ./cmd/apple-game -nia -n 100

# RAnA(K=3) vs AnA (100보드)
go run ./cmd/apple-game -rana -k 3 -n 100

# AnA 단독 벤치마크
go run ./cmd/apple-game -ana-only -n 1000 -ana-out data/raw/games_1000.txt

# Greedy 대규모 벤치마크
go run ./cmd/apple-game -greedy-bench -n 100000 -greedy-out data/raw/greedy_scores.txt

# RL Phase 1: SL 데이터 생성 (AnA 경로, 모든 후보 × playGreedy 라벨)
go run ./cmd/apple-game -gen-sl 500 -sl-out data/generated/sl_data.bin
# 결과: ~130만 레코드, ~228 MB, ~85초 (8코어 기준)

# RL Phase 2: SL 학습 (Colab T4 권장)
python train_sl.py --data sl_data.bin --out model_sl.pt --epochs 50
python train_sl.py --data sl_data.bin --out model_sl.pt --export model_sl.onnx

# RL Phase 3: PPO RL 학습 (Colab T4 권장)
python train_rl.py --sl model_sl.pt --out model_rl.pt --iters 200
python train_rl.py --sl model_sl.pt --out model_rl.pt --export model_rl.onnx

# RL Phase 4: ModelAnA vs AnA 비교 (ONNX 빌드 필요)
go run -tags nn ./cmd/apple-game -nn-ana -n 100 -model models/model_rl.onnx -ort-lib runtime/onnxruntime.dll

# 매크로용 빌드 (ONNX 제외)
GOOS=windows GOARCH=amd64 go build -o apple_solver.exe .

# ONNX 포함 빌드
GOOS=windows GOARCH=amd64 go build -tags nn -o apple_nn.exe .

# 170점 퍼펙트 탐색
go run ./cmd/apple-game -perfect -perfect-out data/generated/perfect_game.txt

# 매크로 연동 (stdin → stdout)
echo "1 2 3 ..." | ./apple_solver.exe -solver
```

## 매크로 파일 위치

```
apple_game_macro/apple_game_MAIN/
├── macro1.py         — 1판 플레이 후 종료
├── macro2.py         — 목표점수 달성까지 반복 후 플레이
│                       사용: python macro2.py 150
├── apple_solver.exe  — Go AnA 엔진 (ONNX 없이 빌드)
├── play_button.png
├── reset_button.png
├── game_board_template.png
└── templates/1.png ~ 9.png
```
