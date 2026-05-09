# RL Apple Game Solver - 기술 문서

## 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [핵심 컴포넌트](#핵심-컴포넌트)
4. [강화학습 알고리즘](#강화학습-알고리즘)
5. [상태 표현 및 행동 공간](#상태-표현-및-행동-공간)
6. [보상 함수 설계](#보상-함수-설계)
7. [학습 전략](#학습-전략)
8. [GUI 시스템](#gui-시스템)
9. [CPU 최적화](#cpu-최적화)
10. [사용 방법](#사용-방법)
11. [성능 평가](#성능-평가)

---

## 프로젝트 개요

### 목적
이 프로젝트는 Apple Puzzle 게임을 해결하기 위한 강화학습(Reinforcement Learning) 기반 솔버를 구현합니다. PPO(Proximal Policy Optimization) 알고리즘을 사용하여 기존 휴리스틱 알고리즘(`solve_min_apples_max_number_bias`)보다 우수한 성능을 달성하는 것을 목표로 합니다.

### 주요 특징
- **PPO 알고리즘**: 안정적이고 CPU 환경에서 효율적인 강화학습 알고리즘
- **최종 보상만 사용**: 게임 종료 시 baseline과의 점수 차이만 보상으로 사용
- **동일 보드 반복 학습**: 같은 보드에서 여러 번 학습하여 점수 향상
- **GUI 기반 학습**: 실시간 진행 상황 모니터링 및 제어
- **자동 체크포인트**: 주기적인 모델 저장 및 성능 테스트

---

## 시스템 아키텍처

### 전체 구조
```
┌─────────────────────────────────────────────────────────┐
│                    GUI (gui.py)                          │
│  - 학습 설정 및 제어                                      │
│  - 진행 상황 표시                                         │
│  - 모델 테스트                                            │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│                 Trainer (trainer.py)                    │
│  - PPO 학습 알고리즘                                      │
│  - 동일 보드 반복 학습                                     │
│  - 체크포인트 관리                                        │
└──────────────┬──────────────────────────────────────────┘
               │
               ├─────────────────┐
               │                 │
┌──────────────▼──────────┐  ┌───▼──────────────────────┐
│   Model (model.py)      │  │ Environment (env.py)    │
│   - Actor-Critic        │  │ - 게임 상태 관리         │
│   - 정책 네트워크        │  │ - 보상 계산             │
│   - 가치 네트워크        │  │ - 행동 실행             │
└─────────────────────────┘  └──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│              Utils (utils.py)                           │
│  - 상태 표현 변환                                        │
│  - 특징 추출                                             │
│  - 모델 저장/로드                                        │
└─────────────────────────────────────────────────────────┘
```

### 데이터 흐름
1. **초기화**: GUI에서 학습 시간 설정
2. **보드 생성**: 랜덤 보드 생성 및 환경 초기화
3. **에피소드 실행**: 
   - 상태 관찰 → 행동 선택 → 환경 업데이트
   - 게임 종료까지 반복
4. **보상 계산**: 게임 종료 시 baseline과 점수 차이 계산
5. **정책 업데이트**: PPO 알고리즘으로 네트워크 업데이트
6. **반복**: 동일 보드에서 여러 에피소드 학습
7. **보드 전환**: 개선 없으면 새 보드로 전환

---

## 핵심 컴포넌트

### 1. environment.py - 게임 환경

#### `AppleGameEnv` 클래스

##### 주요 속성
- `initial_board`: 초기 보드 상태
- `board`: 현재 보드 상태
- `done`: 게임 종료 여부
- `total_removed`: 제거된 총 사과 개수
- `baseline_score`: baseline 알고리즘의 점수

##### 주요 메서드

**`__init__(initial_board=None)`**
- 환경 초기화
- 초기 보드 설정 (없으면 랜덤 생성)
- Baseline 점수 계산

**`reset(new_board=None)`**
- 환경 리셋
- 새 보드 설정 가능
- Baseline 재계산
- 초기 상태 반환

**`get_state()`**
- 현재 상태 벡터 반환
- 보드 벡터 + 특징 벡터 결합

**`get_valid_actions()`**
- 현재 보드에서 유효한 모든 이동 반환
- Summed Area Table 사용하여 효율적 계산

**`step(action)`**
- 행동 실행
- 보드 업데이트
- 게임 종료 여부 확인
- 보상 계산 (게임 종료 시만)
- 반환: (next_state, reward, done, info)

**보상 계산 로직**
```python
if 게임 종료:
    reward = total_removed - baseline_score
else:
    reward = 0  # 즉각 보상 없음
```

**중요 설계 결정**
- **즉각 보상 사용 안함**: 한 턴에서 지운 사과 수를 보상으로 주지 않음
- 이유: 즉각 보상을 주면 많은 사과를 포함한 직사각형만 선택하게 되어 최종 점수가 낮아짐
- 최종 보상만 사용하여 장기적 전략 학습

---

### 2. model.py - PPO 네트워크

#### `ActorCritic` 클래스

##### 네트워크 구조
```
입력 (state_dim = 190)
    ↓
Shared Feature Extractor
    ├─ Linear(state_dim → 128)
    ├─ ReLU
    ├─ Linear(128 → 128)
    └─ ReLU
    ↓
    ├─→ Actor Head          └─→ Critic Head
    │   ├─ Linear(128 → 64)     ├─ Linear(128 → 64)
    │   ├─ ReLU                 ├─ ReLU
    │   └─ Linear(64 → 500)     └─ Linear(64 → 1)
    │       (action logits)         (state value)
```

##### 주요 메서드

**`forward(state)`**
- 순전파 계산
- 입력: 상태 텐서 (batch_size, state_dim)
- 출력: 행동 로짓, 상태 가치

**`get_action(state, valid_actions, deterministic=False)`**
- 정책에서 행동 샘플링
- 마스킹으로 유효한 행동만 선택
- 반환: (action, action_idx, log_prob, value)

**`evaluate_actions(states, actions, valid_actions_list)`**
- PPO 업데이트를 위한 행동 평가
- 배치 처리 지원
- 반환: (action_log_probs, values, entropy)

##### 행동 마스킹
- 가변 크기 행동 공간 처리
- 유효하지 않은 행동은 -inf로 마스킹
- Softmax 적용 시 유효한 행동만 선택됨

---

### 3. trainer.py - PPO 트레이너

#### `PPOTrainer` 클래스

##### 초기화 파라미터
- `state_dim=190`: 상태 차원
- `hidden_dim=128`: 은닉층 차원 (CPU 최적화)
- `lr=3e-4`: 학습률
- `gamma=0.99`: 할인 인자
- `eps_clip=0.2`: PPO 클리핑 파라미터
- `k_epochs=4`: 업데이트 에폭 수
- `save_interval=100`: 모델 저장 간격
- `same_board_episodes=20`: 동일 보드 학습 에피소드 수

##### 주요 메서드

**`train_episode(env, callback=None)`**
- 단일 에피소드 학습
- 상태-행동-보상 수집
- PPO 업데이트 수행
- 반환: (total_reward, score, baseline_score)

**`train_on_same_board(env, callback=None)`**
- 동일 보드에서 여러 에피소드 학습
- 개선 여부 추적
- 반환: (avg_score, improved)

**`test_performance(num_tests=10)`**
- 랜덤 보드에서 모델 성능 테스트
- Deterministic 정책 사용
- 반환: (avg_score, avg_baseline, avg_diff)

**`update(states, actions, old_log_probs, returns, advantages, valid_actions_list)`**
- PPO 업데이트 수행
- 클리핑된 목적 함수 사용
- Actor, Critic, Entropy 손실 결합

##### PPO 업데이트 수식
```
ratio = exp(new_log_prob - old_log_prob)
L_CLIP = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
L_ACTOR = -mean(L_CLIP)
L_CRITIC = MSE(values, returns)
L_ENTROPY = -entropy
L_TOTAL = L_ACTOR + 0.5 * L_CRITIC - 0.01 * L_ENTROPY
```

---

### 4. utils.py - 유틸리티 함수

#### 주요 함수

**`generate_random_board()`**
- 랜덤 보드 생성 (1-9 숫자)

**`extract_features(board)`**
- 보드에서 특징 벡터 추출
- 특징:
  - 총 사과 개수
  - 보드 밀도
  - 평균 사과 값
  - 최대/최소 값
  - 숫자 분포 (1-9)
  - 유효한 이동 수

**`board_to_vector(board)`**
- 보드를 1D 벡터로 변환 (170차원)

**`get_state_representation(board)`**
- 최종 상태 표현 생성
- 보드 벡터 + 특징 벡터 결합 (약 190차원)

**`save_model(model, path, episode, score)`**
- 모델 체크포인트 저장

**`load_model(model, path)`**
- 모델 체크포인트 로드

---

### 5. gui.py - GUI 시스템

#### `RLTrainingGUI` 클래스

##### 주요 기능

**학습 설정**
- 학습 시간 설정 (시간 단위)
- 학습 시작/중지 버튼

**진행 상황 표시**
- 진행률 바
- 현재 에피소드 번호
- 현재 점수 및 baseline
- 경과 시간 및 남은 시간

**통계 표시**
- 최근 50 에피소드 평균
- 총 에피소드 수
- 최고 점수

**모델 테스트**
- 현재 모델 성능 테스트
- 모델 로드 기능

##### 학습 루프
```python
while 학습 시간 미만:
    1. 동일 보드에서 여러 에피소드 학습
    2. 5분마다 성능 테스트
    3. 100 에피소드마다 체크포인트 저장
    4. 개선 없으면 새 보드로 전환
```

---

## 강화학습 알고리즘

### PPO (Proximal Policy Optimization)

#### 알고리즘 선택 이유
1. **안정성**: 정책 업데이트가 안정적
2. **CPU 효율성**: GPU 없이도 효율적 학습 가능
3. **샘플 효율성**: 비교적 적은 샘플로 학습 가능
4. **가변 행동 공간**: 마스킹으로 가변 크기 행동 공간 처리 가능

#### PPO 핵심 개념

**클리핑 목적 함수**
- 정책 업데이트를 제한하여 안정성 확보
- 클리핑 범위: [1-ε, 1+ε] (ε=0.2)

**Actor-Critic 구조**
- Actor: 정책 네트워크 (행동 선택)
- Critic: 가치 네트워크 (상태 평가)

**다중 에폭 업데이트**
- 같은 데이터로 여러 번 업데이트 (k_epochs=4)
- 샘플 효율성 향상

---

## 상태 표현 및 행동 공간

### 상태 표현 (State Representation)

#### 하이브리드 접근
```
상태 벡터 = [보드 벡터 | 특징 벡터]
```

**보드 벡터 (170차원)**
- 10×17 보드를 1D로 평탄화
- 각 셀의 값 (0-9)

**특징 벡터 (약 20차원)**
- 총 사과 개수
- 보드 밀도
- 평균 사과 값
- 최대/최소 값
- 숫자 분포 (1-9, 정규화)
- 유효한 이동 수

**최종 차원**: 약 190차원

### 행동 공간 (Action Space)

#### 가변 크기 행동 공간
- 현재 보드의 유효한 이동 목록에서 선택
- 최대 500개 행동 지원 (마스킹으로 처리)
- 행동 표현: (r1, c1, r2, c2) 튜플

#### 마스킹 메커니즘
1. 유효한 행동만 마스크 생성
2. 유효하지 않은 행동은 -inf로 설정
3. Softmax 적용 시 유효한 행동만 선택됨

---

## 보상 함수 설계

### 보상 설계 원칙

#### 최종 보상만 사용
```python
if 게임 종료:
    reward = total_removed - baseline_score
else:
    reward = 0
```

#### 즉각 보상 사용 안함
- **이유**: 한 턴에서 지운 사과 수를 보상으로 주면
  - 많은 사과를 포함한 직사각형만 선택
  - 단기 이익에 집중
  - 장기적 전략 무시
  - 최종 점수 저하

#### Baseline 비교
- `solve_min_apples_max_number_bias` 알고리즘과 비교
- 양수 보상: baseline보다 높은 점수
- 음수 보상: baseline보다 낮은 점수
- 0 보상: baseline과 동일

### 보상 특성
- **희소성**: 게임 종료 시에만 보상 제공
- **비교 기반**: 절대 점수가 아닌 상대적 성능
- **장기적 관점**: 단기 이익보다 최종 결과 중시

---

## 학습 전략

### 동일 보드 반복 학습

#### 전략 개요
1. 동일 보드에서 여러 에피소드 학습 (기본 20회)
2. 점수 개선 추적
3. 개선 없으면 새 보드로 전환
4. 개선 있으면 계속 학습

#### 개선 판단
```python
if recent_avg_score > initial_score + threshold:
    improved = True
else:
    improved = False
```

#### 장점
- 특정 보드에서 최적 전략 학습
- 지역 최적화 가능
- 학습 효율성 향상

### 보드 전환 전략

#### 전환 조건
1. 개선 없음 (improved = False)
2. 랜덤 전환 (10% 확률)
3. 학습 시간 종료

#### 전환 시 동작
- 새 랜덤 보드 생성
- 환경 리셋
- Baseline 재계산

---

## GUI 시스템

### 주요 화면 구성

#### 1. 학습 설정 패널
- 학습 시간 입력 (시간 단위)
- 시작/중지 버튼

#### 2. 진행 상황 패널
- 진행률 바 (0-100%)
- 상태 메시지
- 에피소드 번호
- 현재 점수 및 baseline
- 경과 시간 / 남은 시간

#### 3. 통계 패널
- 최근 50 에피소드 평균
- 총 에피소드 수
- 최고 점수

#### 4. 모델 테스트 패널
- 테스트 버튼
- 모델 로드 버튼
- 테스트 결과 표시

### 실시간 업데이트
- 1초마다 시간 업데이트
- 에피소드마다 통계 업데이트
- 5분마다 성능 테스트
- 100 에피소드마다 체크포인트 저장

---

## CPU 최적화

### 네트워크 크기 최적화
- 작은 은닉층 (128차원)
- 얕은 네트워크 구조
- 효율적인 연산

### 배치 처리
- 여러 상태를 한 번에 처리
- CPU 메모리 효율적 사용

### 연산 최적화
- PyTorch CPU 연산 활용
- 불필요한 GPU 연산 제거
- 메모리 효율적 텐서 연산

### 학습 파라미터 조정
- 적절한 배치 크기
- 효율적인 업데이트 주기
- 체크포인트 저장 최적화

---

## 사용 방법

### 1. 환경 설정
```bash
pip install torch numpy matplotlib tkinter
```

### 2. GUI 실행
```bash
python rl_solver/main.py
```

또는

```bash
python -m rl_solver.main
```

### 3. 학습 시작
1. GUI에서 학습 시간 설정 (예: 1.0 시간)
2. "Start Training" 버튼 클릭
3. 진행 상황 모니터링
4. 필요시 "Stop Training"으로 중지

### 4. 모델 테스트
1. "Test Current Model" 버튼 클릭
2. 20개 랜덤 보드에서 테스트
3. 결과 확인

### 5. 모델 로드
1. "Load Model" 버튼 클릭
2. 저장된 모델 파일 선택
3. 모델 로드 및 사용

---

## 성능 평가

### 평가 지표

#### 1. 점수 차이 (Score Difference)
```
diff = RL_score - baseline_score
```
- 양수: RL이 baseline보다 우수
- 음수: RL이 baseline보다 열등

#### 2. 평균 점수
- 여러 보드에서의 평균 점수
- Baseline과 비교

#### 3. 개선율
```
improvement = (RL_score - baseline_score) / baseline_score * 100
```

### 테스트 프로세스
1. 랜덤 보드 생성 (기본 20개)
2. Deterministic 정책으로 게임 실행
3. 점수 수집 및 통계 계산
4. Baseline과 비교

---

## 파일 구조

```
rl_solver/
├── __init__.py              # 패키지 초기화
├── environment.py           # 게임 환경
├── model.py                 # PPO 모델
├── trainer.py               # 학습 로직
├── gui.py                   # GUI 시스템
├── utils.py                 # 유틸리티 함수
├── main.py                  # 메인 진입점
├── saved_models/            # 저장된 모델
│   ├── checkpoint_ep100.pt
│   ├── checkpoint_ep200.pt
│   └── final_model.pt
└── RL_TECHNICAL_DOCUMENTATION.md  # 이 문서
```

---

## 주요 파라미터

### 학습 파라미터
- **학습률 (lr)**: 3e-4
- **할인 인자 (gamma)**: 0.99
- **클리핑 범위 (eps_clip)**: 0.2
- **업데이트 에폭 (k_epochs)**: 4
- **은닉층 크기 (hidden_dim)**: 128

### 학습 전략 파라미터
- **동일 보드 에피소드 수**: 20
- **개선 임계값**: 0.1
- **저장 간격**: 100 에피소드
- **테스트 간격**: 5분

---

## 주의사항 및 제한사항

### 주의사항
1. **보상 설계**: 즉각 보상을 사용하지 않음 (의도적 설계)
2. **학습 시간**: 충분한 학습 시간 필요 (수 시간)
3. **CPU 환경**: GPU 없이도 작동하지만 학습 속도는 느림
4. **메모리**: 큰 보드와 많은 행동으로 인한 메모리 사용

### 제한사항
1. **가변 행동 공간**: 최대 500개 행동으로 제한
2. **상태 차원**: 고정된 상태 표현 사용
3. **Baseline 의존**: Baseline 알고리즘 성능에 의존

---

## 향후 개선 방향

### 1. 네트워크 구조 개선
- CNN을 사용한 보드 특징 추출
- Attention 메커니즘 도입
- 더 깊은 네트워크 (GPU 환경)

### 2. 학습 알고리즘 개선
- GAE (Generalized Advantage Estimation) 도입
- 더 정교한 보상 설계
- Curriculum Learning

### 3. 상태 표현 개선
- 더 풍부한 특징 추출
- 보드 임베딩
- 시퀀스 정보 활용

### 4. GUI 개선
- 실시간 그래프 표시
- 더 상세한 통계
- 하이퍼파라미터 튜닝 인터페이스

---

## 참고 자료

### 관련 알고리즘
- PPO: Proximal Policy Optimization
- Actor-Critic Methods
- Policy Gradient Methods

### 관련 프로젝트
- `apple_solver.py`: 기존 휴리스틱 알고리즘
- `solve_min_apples_max_number_bias`: Baseline 알고리즘

---

## 버전 정보

- **버전**: 1.0.0
- **최종 업데이트**: 2024
- **주요 알고리즘**: PPO (Proximal Policy Optimization)

---

*이 문서는 RL Apple Game Solver의 기술적 세부사항을 설명합니다.*

