# RL 학습 계획: AnA 모방 학습 → PPO 강화학습

## 전체 파이프라인

```
Phase 1: SL 데이터 생성 (Go)  ✅
  → sl_data.bin (AnA 경로의 모든 후보 + Greedy 점수 라벨)
  명령: go run . -gen-sl 500 -sl-out sl_data.bin
  결과: ~133만 레코드, ~228 MB, ~85초

Phase 2: SL 학습 (Python/Colab T4)  ✅ (train_sl.py 완성)
  → model_sl.pt / model_sl.onnx
  명령: python train_sl.py --data sl_data.bin --out model_sl.pt --export model_sl.onnx
  목표: val RMSE < 3

Phase 3: PPO RL 학습 (Python/Colab T4)  ✅ (train_rl.py 완성)
  → model_rl.pt / model_rl.onnx
  명령: python train_rl.py --sl model_sl.pt --out model_rl.pt --export model_rl.onnx
  목표: 평균 점수 > AnA(131.6)

Phase 4: 통합 평가 (Go + ONNX)  ✅ (playModelAnA + runModelAnAvsAnA 완성)
  명령: go run -tags nn . -nn-ana -n 100 -model model_rl.onnx -ort-lib onnxruntime.dll
  목표: ModelAnA(RL) avg > 135
```

---

## Phase 1: SL 데이터 생성 ✅ 완료

### 핵심 아이디어
AnA는 매 스텝에서 모든 후보 c에 대해 `playGreedy(apply(board, c))` 를 계산함.
이 `(b2, playGreedy(b2))` 쌍 전부를 저장 → 모델이 Greedy를 모방 학습.

### 이전 ValueNet과의 차이
| | 이전 | 이번 SL |
|--|--|--|
| 데이터 출처 | Greedy 게임 경로 | **AnA 게임 경로** (분포 일치) |
| 라벨 | 게임 끝까지 남은 사과 | **playGreedy(b2) 실제값** |
| RMSE | 5.2 (너무 높음) | 목표 < 3 |

### 구현 완료 (datagen.go)
```bash
go run . -gen-sl 500 -sl-out sl_data.bin
```

### 실제 데이터 크기 (측정값)
- 500게임 × **~2,650 records/game** ≈ **133만 레코드**
  (실제 후보수: 스텝당 평균 ~66개, 초기 추정 500개와 달리)
- 171 bytes/record → **~228 MB** (초기 추정 1.7GB보다 훨씬 작음)
- 소요시간: **~85초** (8코어 병렬)
- 포맷: `[board_170_bytes][greedy_score_1_byte]` (uint8, 0~170)

---

## Phase 2: SL 모델 구조

### 모델: ResNet + 보조 특징 (~1.8M 파라미터)

```python
class AppleNetSL(nn.Module):
    """
    입력: board (B, 1, 10, 17) float32 / 9.0
    출력: predicted Greedy score (B,) float32
    """
    def __init__(self, channels=128, n_blocks=6):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        # ResBlocks (수용 영역: 13×13 after 6 blocks)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
        # Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 보조 특징 3개:
        #   ① 남은 비-0 셀 수 / 170
        #   ② 현재 유효 직사각형 수 / 8415
        #   ③ 남은 셀 합 / (170×9)
        self.fc = nn.Sequential(
            nn.Linear(channels + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, aux_features):
        # x: (B, 1, 10, 17)
        # aux_features: (B, 3)
        feat = self.gap(self.blocks(self.stem(x))).flatten(1)  # (B, 128)
        combined = torch.cat([feat, aux_features], dim=1)      # (B, 131)
        return self.fc(combined).squeeze(-1)                    # (B,)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.net(x))
```

### 모델 크기 비교
| 크기 | 구조 | 파라미터 | 추론속도(배치2000) | 예상 RMSE |
|--|--|--|--|--|
| Small (이전) | Conv×3, 128ch | ~200K | ~1ms | ~5.2 |
| **Medium (추천)** | **ResBlock×6, 128ch** | **~1.8M** | **~3ms** | **목표 ~2-3** |
| Large | ResBlock×8, 256ch | ~12M | ~15ms | ~2 |

### SL 학습 설정
```python
# Colab T4 기준
BATCH_SIZE = 2048
EPOCHS = 50
OPTIMIZER = AdamW(lr=1e-3, weight_decay=1e-4)
SCHEDULER = CosineAnnealingLR
LOSS = nn.MSELoss()
# 라벨 정규화: score / 170.0 → [0, 1] (학습 안정화)
```

---

## Phase 3: PPO RL 학습

### 왜 PPO인가 (DQN 아닌 이유)
| | DQN | PPO |
|--|--|--|
| 가변 행동 공간 | ❌ 고정 크기 필요 | ✅ 마스킹으로 처리 |
| SL 초기화 | 어색함 | ✅ 자연스럽게 연결 |
| 정책 안정성 | 불안정 | ✅ 클리핑으로 보호 |
| 구현 복잡도 | 중간 | 중간 |

### PPO 구조 (Actor-Critic 공유 네트워크)
```
정책(Actor):  π(a|s) = softmax( model(apply(s,a)) for a in valid_actions )
가치(Critic): V(s)   = E_π[ model(apply(s,a)) ]   ← 같은 네트워크 공유

PPO Loss =
  -min( r_t × A_t,  clip(r_t, 1-ε, 1+ε) × A_t )   ← 정책 손실
  + λ_v × (V(s_t) - G_t)²                            ← 가치 손실
  - λ_e × Entropy(π(·|s_t))                          ← 탐색 보너스

r_t = π_new(a_t|s_t) / π_old(a_t|s_t)
A_t = G_t - V(s_t)   (Advantage)
G_t = Σ_{k≥t} γ^(k-t) × r_k   (감가 수익, γ=1.0 권장)
```

### SL→RL 연결 핵심
```
SL: label(b2) = playGreedy(b2)   → Greedy의 한계에 갇힘
RL: label(b2) = actual_remaining  → 모델이 Greedy보다 잘하면 라벨도 올라감 → 선순환
```

### Python 게임 엔진 (RL 학습용)
```python
import numpy as np

def build_prefix_sum(board):
    """2D prefix sum for value and count."""
    val = np.zeros((11, 18), dtype=np.int32)
    cnt = np.zeros((11, 18), dtype=np.int32)
    val[1:, 1:] = np.cumsum(np.cumsum(board.astype(np.int32), axis=0), axis=1)
    cnt[1:, 1:] = np.cumsum(np.cumsum((board > 0).astype(np.int32), axis=0), axis=1)
    return val, cnt

def query(ps, r1, c1, r2, c2):
    return ps[r2+1,c2+1] - ps[r1,c2+1] - ps[r2+1,c1] + ps[r1,c1]

def get_valid_rects(board):
    """Returns list of (r1,c1,r2,c2) with sum==10."""
    val, _ = build_prefix_sum(board)
    rects = []
    for r1 in range(10):
        for r2 in range(r1, 10):
            for c1 in range(17):
                for c2 in range(c1, 17):
                    if query(val, r1, c1, r2, c2) == 10:
                        rects.append((r1, c1, r2, c2))
    return rects

def apply_rect(board, r1, c1, r2, c2):
    b2 = board.copy()
    removed = 0
    for r in range(r1, r2+1):
        for c in range(c1, c2+1):
            if b2[r, c] > 0:
                b2[r, c] = 0
                removed += 1
    return b2, removed

def play_episode(model, board, epsilon=0.1, device='cuda'):
    """Play one episode. Returns list of (b2, removed) and total score."""
    trajectory = []
    board = board.copy()
    while True:
        rects = get_valid_rects(board)
        if not rects:
            break
        # Batch evaluate all candidates
        next_boards = []
        removals = []
        for r in rects:
            b2, rem = apply_rect(board, *r)
            next_boards.append(b2)
            removals.append(rem)
        # Model inference (GPU batch)
        x = torch.tensor(np.array(next_boards)[:,None,:,:],
                         dtype=torch.float32, device=device) / 9.0
        aux = compute_aux_features(next_boards, device)  # (N, 3)
        with torch.no_grad():
            scores = model(x, aux).cpu().numpy()  # (N,)
        # Action selection
        if np.random.random() < epsilon:
            idx = np.random.randint(len(rects))
        else:
            idx = scores.argmax()
        trajectory.append((next_boards[idx].copy(), removals[idx]))
        board = next_boards[idx]
    total = sum(r for _, r in trajectory)
    return trajectory, total
```

### PPO 학습 루프
```python
def ppo_epoch(model, old_model, optimizer, n_episodes=200,
              epsilon=0.1, clip_eps=0.2, lambda_v=0.5, lambda_e=0.01):
    all_data = []
    scores = []

    # 1. 에피소드 생성
    for _ in range(n_episodes):
        board = random_board()
        trajectory, total = play_episode(model, board, epsilon)
        scores.append(total)
        # 각 스텝의 (b2, actual_remaining) 계산
        remaining = total
        for b2, removed in trajectory:
            remaining -= removed
            all_data.append((b2, remaining))  # V(b2) 타겟

    # 2. 위 중앙값 에피소드만 사용
    baseline = np.median(scores)
    # (위 필터링 로직 생략)

    # 3. PPO 업데이트
    dataset = BoardDataset(all_data)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    for boards, targets in loader:
        pred = model(boards.to(device), aux.to(device))
        # Value loss
        v_loss = F.mse_loss(pred, targets.to(device))
        # Policy loss (PPO clip)
        log_pi_new = compute_log_pi(model, ...)
        log_pi_old = compute_log_pi(old_model, ...)
        ratio = (log_pi_new - log_pi_old).exp()
        adv = (targets - pred.detach())
        p_loss = -torch.min(ratio * adv, ratio.clamp(1-clip_eps, 1+clip_eps) * adv).mean()
        # Entropy bonus
        entropy = compute_entropy(model, ...)
        loss = p_loss + lambda_v * v_loss - lambda_e * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(scores)
```

### 안정화 기법
1. **KL 정규화**: RL 모델이 SL 모델에서 너무 멀어지지 않도록
2. **Above-median 필터**: 평균 이상 에피소드만 학습
3. **Mixed batch**: SL 데이터 20% 혼합 → catastrophic forgetting 방지
4. **Entropy bonus**: 탐색 유지 (lambda_e=0.01)
5. **Gradient clipping**: max_norm=1.0

---

## Phase 4: Go 통합 ✅ 완료

```go
// nn.go에 추가 (//go:build nn) — nnCtxV2 사용 (board + aux 두 입력)
func playModelAnA(b board, nc *nnCtxV2) int {
    // playMCGreedyAllCands와 동일 구조
    // 단, playGreedy(b2) 대신 nc.eval(&b2) 사용
    var val, cnt ps2d
    var cands [maxRects]rect
    removed := 0
    for {
        buildPS(&b, &val, &cnt)
        nCands := 0
        for r1 := 0; r1 < nRows; r1++ {
            // ... 후보 열거
        }
        if nCands == 0 { break }
        bestScore := -1
        var bestR rect
        for i := 0; i < nCands; i++ {
            b2 := b
            applyRect(&b2, cands[i])
            score := nc.eval(&b2)  // ← Greedy 대신 모델
            if score > bestScore {
                bestScore = score
                bestR = cands[i]
            }
        }
        removed += applyRect(&b, bestR)
    }
    return removed
}
```

---

## 목표 성능

```
Greedy       ~ 113
AnA          ~ 131
RAnA(K=3)    ~ 133
ModelAnA(SL) ~ 133~135  (AnA 수준 재현 목표)
ModelAnA(RL) ~ 135~140  (AnA 초과 목표)
```
