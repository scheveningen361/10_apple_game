# Apple Game Macro - 기술 문서

## 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [핵심 컴포넌트](#핵심-컴포넌트)
4. [알고리즘 상세](#알고리즘-상세)
5. [설치 및 설정](#설치-및-설정)
6. [사용 방법](#사용-방법)
7. [기술 스택](#기술-스택)
8. [성능 최적화](#성능-최적화)
9. [제한사항 및 향후 개선](#제한사항-및-향후-개선)

---

## 프로젝트 개요

### 프로젝트 목적
이 프로젝트는 Apple Puzzle 게임을 자동으로 플레이하는 매크로 프로그램입니다. 컴퓨터 비전과 최적화 알고리즘을 활용하여 게임 보드를 인식하고, 최적의 이동 경로를 계산한 후 자동으로 게임을 실행합니다.

### 게임 규칙
- **보드 크기**: 10행 × 17열 (170개 셀)
- **목표**: 직사각형 영역을 선택하여 합이 정확히 10이 되도록 만들기
- **제약 조건**: 선택한 영역에는 최소 2개 이상의 사과가 포함되어야 함
- **점수 계산**: 제거된 사과의 개수 = 점수
- **게임 목표**: 최대한 많은 사과를 제거하여 높은 점수 달성

### 주요 기능
1. **화면 캡처 및 보드 인식**: 실시간 스크린샷을 통해 게임 보드 상태를 인식
2. **최적 경로 계산**: 여러 휴리스틱 알고리즘을 사용하여 최적의 이동 순서 계산
3. **자동 플레이**: 계산된 경로를 기반으로 마우스 자동 제어
4. **점수 추적**: 최고 점수 기록 및 목표 점수 달성 시 자동 종료
5. **자동 리셋**: 낮은 점수 예상 시 자동으로 게임 리셋

---

## 시스템 아키텍처

### 전체 구조
```
┌─────────────────────────────────────────────────────────┐
│                    main.py (메인 제어)                    │
│  - 화면 캡처 관리                                         │
│  - 보드 인식 및 파싱                                      │
│  - 마우스 자동화 제어                                     │
│  - 게임 플로우 관리                                       │
└──────────────┬──────────────────────────────────────────┘
               │
               ├─────────────────┐
               │                 │
┌──────────────▼──────────┐  ┌───▼──────────────────────┐
│   apple_solver.py       │  │ capture_area_selector.py │
│   (알고리즘 엔진)        │  │ (GUI 도구)               │
│                         │  │                          │
│  - 휴리스틱 알고리즘      │  │  - 영역 선택 GUI         │
│  - 유효 이동 탐색        │  │  - 좌표 출력             │
│  - 점수 계산             │  │                          │
└─────────────────────────┘  └──────────────────────────┘
```

### 데이터 흐름
1. **캡처 단계**: `mss` 라이브러리를 사용하여 전체 화면 캡처
2. **인식 단계**: OpenCV 템플릿 매칭을 통해 게임 보드 위치 탐지
3. **파싱 단계**: 각 셀을 템플릿 매칭으로 숫자 인식
4. **계산 단계**: `apple_solver.py`의 알고리즘으로 최적 경로 계산
5. **실행 단계**: PyAutoGUI를 사용하여 마우스 드래그로 이동 수행
6. **평가 단계**: 점수 평가 및 리셋/계속 결정

---

## 핵심 컴포넌트

### 1. main.py

#### 주요 함수

##### `load_templates()`
- **목적**: 숫자 템플릿 이미지 로드
- **동작**: `templates/` 디렉토리에서 1~9 숫자 이미지를 읽어 딕셔너리로 저장
- **반환값**: `{숫자: 그레이스케일 이미지}` 딕셔너리

##### `calibrate_game_board_template()`
- **목적**: 게임 보드 템플릿 이미지 설정
- **동작**: 사용자로부터 템플릿 이미지 경로를 입력받아 저장
- **저장 위치**: `game_board_template.png`, `config.json`

##### `find_game_board_in_screenshot(full_screenshot_img_cv)`
- **목적**: 전체 스크린샷에서 게임 보드 영역 찾기
- **알고리즘**: OpenCV `cv2.matchTemplate` 사용 (TM_CCOEFF_NORMED)
- **임계값**: 매칭 점수 ≥ 0.8
- **반환값**: `{left, top, width, height}` 딕셔너리

##### `recognize_board_from_image(full_screenshot_img_cv, capture_area, templates)`
- **목적**: 게임 보드 이미지에서 각 셀의 숫자 인식
- **알고리즘**:
  1. 보드 영역을 10×17 그리드로 분할
  2. 각 셀에 대해 모든 템플릿(1~9)과 매칭
  3. 최고 매칭 점수를 가진 숫자 선택
- **반환값**: 2D 리스트 `[[숫자, ...], ...]`

##### `perform_mouse_drag(move, game_board_offset, cell_dims, mouse_offset)`
- **목적**: 계산된 이동을 마우스 드래그로 실행
- **파라미터**:
  - `move`: `(r1, c1, r2, c2)` 튜플 (시작/끝 행/열)
  - `game_board_offset`: 보드의 화면상 위치
  - `cell_dims`: 셀의 픽셀 크기
  - `mouse_offset`: 마우스 보정 오프셋
- **동작**:
  1. 시작 좌표로 마우스 이동
  2. 마우스 다운
  3. 끝 좌표로 드래그
  4. 미세 조정 (좌우 이동으로 드래그 확실히 인식)
  5. 마우스 업

##### `calibrate_mouse_offset(game_board_area)`
- **목적**: 마우스 클릭 위치 보정
- **동작**: 사용자가 보드 좌상단 모서리에 마우스를 위치시키면 오프셋 계산

##### `find_and_click_button(image_path, button_name)`
- **목적**: 화면에서 버튼 이미지를 찾아 클릭
- **알고리즘**: PyAutoGUI `locateOnScreen` 사용
- **신뢰도**: 0.9 (90%)

#### 메인 실행 루프
```python
while True:
    1. 스크린샷 캡처
    2. 게임 보드 영역 탐지
    3. 보드 상태 인식
    4. 최적 이동 경로 계산
    5. 점수 평가
    6. 최고 점수 갱신 시 자동 플레이
    7. 목표 점수 달성 시 종료
    8. 리셋 및 재시작
```

---

### 2. apple_solver.py

#### 핵심 데이터 구조

##### Summed Area Table (SAT)
- **목적**: O(1) 시간 복잡도로 직사각형 영역의 합 계산
- **구조**: `(ROWS+1) × (COLS+1)` 크기의 2D 배열
- **초기화**:
  ```python
  sat[r+1][c+1] = board[r][c] + sat[r][c+1] + sat[r+1][c] - sat[r][c]
  ```
- **합 계산**:
  ```python
  sum = sat[r2+1][c2+1] - sat[r1][c2+1] - sat[r2+1][c1] + sat[r1][c1]
  ```

#### 주요 함수

##### `calculate_summed_area_table(board)`
- **목적**: 보드의 누적 합 테이블 생성
- **시간 복잡도**: O(ROWS × COLS)
- **공간 복잡도**: O(ROWS × COLS)

##### `get_rect_sum(sat, r1, c1, r2, c2)`
- **목적**: 직사각형 영역의 합을 O(1)로 계산
- **파라미터**: SAT와 직사각형 좌표

##### `find_valid_moves(board, sat_values, sat_counts)`
- **목적**: 합이 10인 모든 유효한 직사각형 찾기
- **알고리즘**: 모든 가능한 직사각형 조합 탐색
- **시간 복잡도**: O(ROWS² × COLS²)
- **제약 조건**:
  - 합 = 10
  - 사과 개수 ≥ 2

##### `get_distance_from_center(move)`
- **목적**: 이동의 중심점과 보드 중심점 사이의 유클리드 거리 계산
- **용도**: 중심에 가까운 이동을 선호하는 휴리스틱

##### `get_emptiness_score(board, move, radius=3)`
- **목적**: 이동 주변의 빈 셀 개수 계산
- **용도**: 빈 공간 근처의 이동을 선호하는 휴리스틱

##### `get_max_number_in_move(board, move)`
- **목적**: 이동 영역 내 최대 숫자 찾기
- **용도**: 높은 숫자를 포함한 이동을 선호하는 휴리스틱

#### 알고리즘 구현

##### 1. `solve_min_apples_center_close(initial_board)`
- **전략**: 최소 사과 개수 → 중심에 가까운 이동
- **정렬 키**: `(사과 개수, 중심 거리)`
- **특징**: 보드 중심으로 이동을 집중시켜 보드 정리

##### 2. `solve_min_apples_empty_bias(initial_board)`
- **전략**: 최소 사과 개수 → 빈 공간 근처 이동
- **정렬 키**: `(사과 개수, -빈 공간 점수)`
- **특징**: 빈 공간을 활용하여 보드 분산

##### 3. `solve_min_apples_max_number_bias(initial_board)` ⭐ **현재 사용 중**
- **전략**: 최소 사과 개수 → 최대 숫자 포함
- **정렬 키**: `(사과 개수, -최대 숫자)`
- **특징**: 높은 숫자를 우선 제거하여 점수 최적화
- **성능**: 실험적으로 가장 높은 점수를 달성

##### 4. `solve_max_number_min_apples(initial_board)`
- **전략**: 최대 숫자 포함 → 최소 사과 개수
- **정렬 키**: `(-최대 숫자, 사과 개수)`
- **특징**: 높은 숫자를 우선적으로 제거

##### 5. `solve_full_simulation(initial_board)`
- **전략**: 각 이동에 대해 전체 게임 시뮬레이션 수행
- **알고리즘**:
  1. 모든 유효 이동에 대해:
     - 이동 적용
     - 나머지 게임을 `solve_min_apples_max_number_bias`로 시뮬레이션
     - 최종 점수 계산
  2. 최고 점수를 가진 이동 선택
- **시간 복잡도**: O(이동 수 × 게임 시뮬레이션 시간)
- **특징**: 가장 정확하지만 매우 느림

##### `evaluate_full_game_score(initial_board, initial_move)`
- **목적**: 특정 이동 후 전체 게임 점수 평가
- **동작**: 이동 적용 후 나머지 게임을 휴리스틱으로 시뮬레이션

---

### 3. capture_area_selector.py

#### 목적
게임 보드 캡처 영역을 시각적으로 선택하기 위한 GUI 도구

#### 구현
- **라이브러리**: Tkinter
- **기능**:
  - 전체 화면 오버레이 (반투명)
  - 마우스 드래그로 영역 선택
  - 실시간 사각형 표시
  - 좌표 출력

#### 사용법
```bash
python capture_area_selector.py
```
드래그하여 영역 선택 후 좌표가 콘솔에 출력됨

---

## 알고리즘 상세

### 최적화 전략

#### 1. Greedy 알고리즘
현재 구현은 **Greedy 알고리즘**을 사용합니다:
- 각 단계에서 가장 좋아 보이는 이동을 선택
- 미래 결과를 완전히 고려하지 않음
- 빠른 계산 속도

#### 2. 휴리스틱 함수
여러 휴리스틱을 조합하여 이동 선택:
- **사과 개수 최소화**: 더 많은 이동 기회 확보
- **숫자 최대화**: 높은 점수 달성
- **위치 최적화**: 보드 상태 개선

#### 3. Summed Area Table 최적화
- **문제**: 모든 직사각형의 합을 계산하는 것은 O(ROWS² × COLS² × ROWS × COLS) 시간 소요
- **해결**: SAT를 사용하여 O(1) 합 계산
- **개선**: 전체 시간 복잡도를 O(ROWS² × COLS²)로 감소

### 알고리즘 비교

| 알고리즘 | 평균 점수 | 계산 시간 | 특징 |
|---------|---------|----------|------|
| Min Apples (Center) | 중간 | 빠름 | 중심 집중 |
| Min Apples (Empty) | 중간 | 빠름 | 분산 전략 |
| **Min Apples (Max Number)** | **높음** | **빠름** | **점수 최적화** |
| Max Number (Min Apples) | 중간 | 빠름 | 높은 숫자 우선 |
| Full Simulation | 매우 높음 | 매우 느림 | 정확하지만 비현실적 |

---

## 설치 및 설정

### 필수 패키지
```bash
pip install mss pillow opencv-python numpy pyautogui tqdm joblib
```

### 디렉토리 구조
```
apple_game_MAIN/
├── main.py                    # 메인 실행 파일
├── apple_solver.py            # 알고리즘 엔진
├── capture_area_selector.py   # 영역 선택 도구
├── config.json                # 설정 파일
├── game_board_template.png    # 보드 템플릿 이미지
├── play_button.png            # 플레이 버튼 이미지
├── reset_button.png           # 리셋 버튼 이미지
├── templates/                 # 숫자 템플릿 (1.png ~ 9.png)
├── output_images/             # 출력 이미지 저장
├── capture.png                # 캡처된 보드 이미지
├── recognized_board.txt       # 인식된 보드 상태
└── score_log.txt              # 점수 로그
```

### 초기 설정

#### 1. 템플릿 이미지 준비
- `templates/` 폴더에 1.png ~ 9.png 숫자 이미지 준비
- 각 이미지는 게임 내 숫자와 동일한 스타일이어야 함

#### 2. 게임 보드 템플릿 설정
```bash
python main.py --calibrate
```
- 게임 보드의 스크린샷 이미지 경로 입력
- 템플릿이 `game_board_template.png`로 저장됨

#### 3. 마우스 오프셋 보정
```bash
python main.py --calibrate-mouse
```
- 게임 보드 좌상단 모서리에 마우스 이동 후 Enter
- 오프셋이 `config.json`에 저장됨

#### 4. 버튼 이미지 준비
- `play_button.png`: 게임 시작 버튼 이미지
- `reset_button.png`: 게임 리셋 버튼 이미지

---

## 사용 방법

### 기본 실행
```bash
python main.py
```

### 실행 시 입력
1. **최고 점수 (highest score)**: 현재 최고 점수 입력 (기본값: 150)
2. **목표 점수 (target score)**: 달성 시 종료할 점수 (기본값: 170)
3. **녹화 여부 (recording)**: 1 = 녹화 활성화, 0 = 비활성화

### 실행 흐름
1. 프로그램 시작 후 3초 대기
2. 플레이 버튼 자동 클릭
3. 게임 루프 시작:
   - 스크린샷 캡처
   - 보드 인식
   - 최적 경로 계산
   - 점수 평가
   - 최고 점수 갱신 시 자동 플레이
   - 목표 점수 달성 시 종료
   - 리셋 및 재시작

### 설정 파일 (config.json)
```json
{
  "game_board_template_width": 702,
  "game_board_template_height": 418,
  "mouse_offset_x": 3,
  "mouse_offset_y": 4
}
```

---

## 기술 스택

### 핵심 라이브러리

#### 1. mss
- **용도**: 고속 스크린샷 캡처
- **특징**: 하드웨어 가속 지원, 빠른 성능

#### 2. OpenCV (cv2)
- **용도**: 이미지 처리 및 템플릿 매칭
- **주요 기능**:
  - `cv2.matchTemplate`: 템플릿 매칭
  - `cv2.imread`: 이미지 읽기
  - `cv2.cvtColor`: 색상 변환

#### 3. PyAutoGUI
- **용도**: 마우스 자동 제어
- **주요 기능**:
  - `pyautogui.moveTo`: 마우스 이동
  - `pyautogui.click`: 클릭
  - `pyautogui.locateOnScreen`: 화면에서 이미지 찾기
  - `pyautogui.hotkey`: 단축키 입력

#### 4. PIL (Pillow)
- **용도**: 이미지 처리 및 변환
- **주요 기능**: RGB 변환, 이미지 저장

#### 5. NumPy
- **용도**: 배열 연산 및 이미지 데이터 처리

#### 6. tqdm
- **용도**: 진행 상황 표시 (알고리즘 테스트 시)

#### 7. joblib
- **용도**: 병렬 처리 (현재는 import만 되어 있음)

---

## 성능 최적화

### 1. Summed Area Table (SAT)
- **문제**: 직사각형 합 계산이 O(ROWS × COLS) 시간 소요
- **해결**: SAT로 O(1) 시간으로 감소
- **효과**: 알고리즘 전체 시간 복잡도 대폭 감소

### 2. 템플릿 매칭 최적화
- **그레이스케일 변환**: 컬러 이미지를 그레이스케일로 변환하여 처리 속도 향상
- **임계값 설정**: 매칭 점수 0.8 이상만 인식하여 오인식 방지

### 3. 이미지 캡처 최적화
- **mss 사용**: 하드웨어 가속 스크린샷으로 빠른 캡처
- **필요 영역만 처리**: 전체 화면 중 보드 영역만 처리

### 4. 알고리즘 선택
- **현재**: `solve_min_apples_max_number_bias` 사용
- **이유**: 빠른 계산 속도와 높은 점수 달성의 균형

---

## 제한사항 및 향후 개선

### 현재 제한사항

#### 1. 템플릿 매칭 의존성
- **문제**: 게임 UI 변경 시 템플릿 이미지 재생성 필요
- **해결 방안**: OCR (Optical Character Recognition) 도입 고려

#### 2. 고정된 보드 크기
- **문제**: 10×17 크기로 하드코딩됨
- **해결 방안**: 동적 보드 크기 감지

#### 3. 단일 휴리스틱
- **문제**: 하나의 휴리스틱만 사용
- **해결 방안**: 여러 휴리스틱 조합 또는 학습 기반 선택

#### 4. 완전 탐색 불가
- **문제**: 모든 가능한 경로 탐색은 시간이 너무 오래 걸림
- **현재**: Greedy 알고리즘으로 근사 해결

### 향후 개선 방향

#### 1. 머신러닝 통합
- **목적**: 보드 상태에 따라 최적 휴리스틱 자동 선택
- **방법**: 강화학습 또는 지도학습 모델 도입

#### 2. OCR 도입
- **목적**: 템플릿 이미지 의존성 제거
- **방법**: Tesseract OCR 또는 딥러닝 기반 숫자 인식

#### 3. 병렬 처리
- **목적**: 여러 알고리즘 동시 실행 및 최고 결과 선택
- **방법**: `joblib`을 사용한 멀티프로세싱

#### 4. 시뮬레이션 최적화
- **목적**: `solve_full_simulation`의 성능 개선
- **방법**: 가지치기(Pruning), 메모이제이션, 제한 깊이 탐색

#### 5. 동적 보드 감지
- **목적**: 다양한 보드 크기 지원
- **방법**: 그리드 라인 감지 알고리즘

#### 6. GUI 개선
- **목적**: 사용자 친화적인 인터페이스
- **방법**: 설정 GUI, 실시간 보드 표시, 점수 그래프

---

## 코드 예제

### 보드 인식 예제
```python
# 템플릿 로드
templates = load_templates()

# 스크린샷 캡처
with mss.mss() as sct:
    sct_img = sct.grab(sct.monitors[0])
    full_screenshot_pil = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    full_screenshot_cv = cv2.cvtColor(np.array(full_screenshot_pil), cv2.COLOR_RGB2GRAY)

# 보드 영역 찾기
game_board_area = find_game_board_in_screenshot(full_screenshot_cv)

# 보드 상태 인식
current_board = recognize_board_from_image(full_screenshot_cv, game_board_area, templates)
```

### 알고리즘 실행 예제
```python
# 최적 이동 경로 계산
optimal_moves, total_removed = solve_min_apples_max_number_bias(current_board)

# 이동 실행
for move in optimal_moves:
    perform_mouse_drag(move, game_board_area, (cell_width_px, cell_height_px), (mouse_offset_x, mouse_offset_y))
```

### SAT 사용 예제
```python
# SAT 생성
sat = calculate_summed_area_table(board)

# 직사각형 합 계산 (O(1))
rect_sum = get_rect_sum(sat, r1, c1, r2, c2)
```

---

## 문제 해결

### 일반적인 문제

#### 1. 보드를 찾을 수 없음
- **원인**: 템플릿 이미지 불일치 또는 화면 해상도 변경
- **해결**: `--calibrate` 옵션으로 템플릿 재설정

#### 2. 숫자 인식 오류
- **원인**: 템플릿 이미지 품질 문제
- **해결**: 더 명확한 템플릿 이미지 사용

#### 3. 마우스 클릭 위치 오류
- **원인**: 마우스 오프셋 미설정 또는 변경
- **해결**: `--calibrate-mouse` 옵션으로 오프셋 재설정

#### 4. 버튼을 찾을 수 없음
- **원인**: 버튼 이미지 변경 또는 화면 해상도 변경
- **해결**: 최신 버튼 이미지로 교체

---

## 참고 자료

### 관련 알고리즘
- **Greedy Algorithm**: 각 단계에서 최선의 선택
- **Summed Area Table**: 누적 합 테이블을 이용한 빠른 영역 합 계산
- **Template Matching**: 이미지에서 특정 패턴 찾기

### 유사 프로젝트
- 게임 자동화 매크로
- 컴퓨터 비전 기반 게임 봇
- 퍼즐 게임 솔버

---

## 라이선스 및 저작권

이 프로젝트는 교육 및 개인 사용 목적으로 제작되었습니다. 상업적 사용 시 해당 게임의 이용약관을 확인하시기 바랍니다.

---

## 버전 정보

- **현재 버전**: 1.0
- **최종 업데이트**: 2024
- **주요 알고리즘**: `solve_min_apples_max_number_bias`

---

## 작성자 정보

이 문서는 프로젝트 코드 분석을 통해 자동 생성되었습니다.

