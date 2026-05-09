"""
Run this once to regenerate colab.ipynb:
    python gen_notebook.py
"""
import json

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

cells = []

# ── 0. 타이틀 ─────────────────────────────────────────────────────────────────
cells.append(md(
    "# 🍎 Apple Game RL Training\n"
    "\n"
    "| 단계 | 위치 | 내용 |\n"
    "|------|------|------|\n"
    "| **Phase 1** | 로컬 PC (Go) | `sl_data.bin` 생성 |\n"
    "| **Phase 2** | 이 노트북 | SL 지도학습 → `model_sl.pt` / `model_sl.onnx` |\n"
    "| **Phase 3** | 이 노트북 | MaskablePPO (SB3) → `model_rl.zip` / `model_rl.onnx` |\n"
    "| **Phase 4** | 로컬 PC (Go) | ModelAnA vs AnA 비교 |\n"
    "\n"
    "### 사전 준비\n"
    "1. 로컬에서 데이터 생성: `go run ./cmd/apple-game -gen-sl 500 -sl-out data/generated/sl_data.bin`\n"
    "2. `sl_data.bin` (~228 MB) → Google Drive `내 드라이브/apple_game/` 에 업로드\n"
    "3. **런타임 → 런타임 유형 변경 → T4 GPU**\n"
    "4. 설정 셀 확인 후 **위에서부터 순서대로** 실행"
))

# ── 1. 설정 ───────────────────────────────────────────────────────────────────
cells.append(md("---\n## ⚙️ 설정"))

cells.append(code(
    'GITHUB_URL   = "https://github.com/scheveningen361/10_apple_game.git"\n'
    'DRIVE_FOLDER = "/content/drive/MyDrive/apple_game"\n'
    "\n"
    "# Phase 2 SL\n"
    "SL_EPOCHS = 50\n"
    "\n"
    "# Phase 3 RL\n"
    "RL_ITERS  = 200   # PPO 업데이트 횟수\n"
    "RL_ENVS   = 4     # 병렬 환경 수 (T4: 4~8 권장)\n"
    "RL_STEPS  = 2048  # 환경당 롤아웃 스텝"
))

# ── 2. 환경 셋업 ──────────────────────────────────────────────────────────────
cells.append(md("---\n## 🔧 환경 셋업"))

cells.append(code(
    "# 1. GPU 확인\n"
    "import torch, os, shutil\n"
    "if torch.cuda.is_available():\n"
    "    prop = torch.cuda.get_device_properties(0)\n"
    '    print(f"✅ GPU : {prop.name}  ({prop.total_memory/1e9:.1f} GB)")\n'
    "else:\n"
    '    print("⚠️  GPU 없음 — 런타임 유형을 T4로 변경하세요")'
))

cells.append(code(
    "# 2. Google Drive 마운트\n"
    "from google.colab import drive\n"
    'drive.mount("/content/drive")\n'
    "os.makedirs(DRIVE_FOLDER, exist_ok=True)\n"
    'print("Drive 마운트 완료:", DRIVE_FOLDER)'
))

cells.append(code(
    "# 3. repo 클론 / 최신화\n"
    'REPO_DIR = "/content/apple_game"\n'
    'RL_DIR   = REPO_DIR + "/rl"\n'
    "\n"
    'if os.path.exists(REPO_DIR + "/.git"):\n'
    "    !cd {REPO_DIR} && git pull\n"
    "else:\n"
    "    !git clone {GITHUB_URL} {REPO_DIR}\n"
    "\n"
    "os.chdir(RL_DIR)\n"
    'print("\\n--- 현재 커밋 ---")\n'
    "!cd {REPO_DIR} && git log --oneline -3\n"
    'print("\\n--- rl/ 파일 목록 ---")\n'
    "!ls -lh"
))

cells.append(code(
    "# 4. sl_data.bin 복사  (Drive → 작업 디렉토리)\n"
    'src = f"{DRIVE_FOLDER}/sl_data.bin"\n'
    'dst = f"{RL_DIR}/sl_data.bin"\n'
    "\n"
    "if os.path.exists(dst):\n"
    '    print(f"✅ sl_data.bin 이미 있음  ({os.path.getsize(dst)/1e6:.0f} MB)")\n'
    "elif os.path.exists(src):\n"
    "    shutil.copy(src, dst)\n"
    '    print(f"✅ Drive에서 복사 완료   ({os.path.getsize(dst)/1e6:.0f} MB)")\n'
    "else:\n"
    "    raise FileNotFoundError(\n"
    '        f"sl_data.bin 없음!\\n"\n'
    '        f"로컬에서 생성 후 {DRIVE_FOLDER}/ 에 업로드하세요.\\n"\n'
    '        f"명령: go run ./cmd/apple-game -gen-sl 500 -sl-out data/generated/sl_data.bin")'
))

# ── Phase 2 ───────────────────────────────────────────────────────────────────
cells.append(md("---\n## 🎓 Phase 2: Supervised Learning"))

cells.append(code(
    "# SL 학습 (50 epochs, T4 기준 AMP FP16 ~5 min)\n"
    "!python train_sl.py \\\n"
    "    --data   sl_data.bin \\\n"
    "    --out    model_sl.pt \\\n"
    "    --epochs {SL_EPOCHS}"
))

cells.append(code(
    "# SL 체크포인트 확인\n"
    "ckpt = torch.load('model_sl.pt', map_location='cpu')\n"
    "cfg  = ckpt['config']\n"
    "print(f\"Best val RMSE : {ckpt['best_val'] * 170:.4f}\")\n"
    "print(f\"Best epoch    : {ckpt['epoch'] + 1}\")\n"
    "print(f\"Config        : {cfg}\")"
))

cells.append(code(
    "# SL → ONNX  (Go 호환)\n"
    "import sys\n"
    "sys.path.insert(0, RL_DIR)\n"
    "from train_sl import AppleNetSL, export_onnx\n"
    "\n"
    "ckpt  = torch.load('model_sl.pt', map_location='cpu')\n"
    "cfg   = ckpt['config']\n"
    "model = AppleNetSL(channels=cfg['channels'], n_blocks=cfg['blocks'])\n"
    "model.load_state_dict(ckpt['model'])\n"
    "export_onnx(model, 'model_sl.onnx', device='cpu')\n"
    "print(f\"model_sl.onnx  {os.path.getsize('model_sl.onnx')/1e6:.1f} MB\")"
))

cells.append(code(
    "# SL 모델 Drive 백업\n"
    "for f in ['model_sl.pt', 'model_sl.onnx']:\n"
    "    if os.path.exists(f):\n"
    "        shutil.copy(f, f'{DRIVE_FOLDER}/{f}')\n"
    "        print(f'💾 {DRIVE_FOLDER}/{f}')"
))

# ── Phase 3 ───────────────────────────────────────────────────────────────────
cells.append(md("---\n## 🤖 Phase 3: PPO RL (MaskablePPO)"))

cells.append(code(
    "# SB3 설치 (최초 1회)\n"
    "!pip install -q stable-baselines3 sb3-contrib gymnasium"
))

cells.append(code(
    "# RL 학습\n"
    "!python train_rl.py \\\n"
    "    --sl      model_sl.pt \\\n"
    "    --out     model_rl.zip \\\n"
    "    --iters   {RL_ITERS}   \\\n"
    "    --n-envs  {RL_ENVS}    \\\n"
    "    --n-steps {RL_STEPS}"
))

cells.append(code(
    "# RL 체크포인트 확인\n"
    "from sb3_contrib import MaskablePPO\n"
    "\n"
    "rl_model = MaskablePPO.load('model_rl.zip')\n"
    "print(f'Policy device  : {rl_model.device}')\n"
    "print(f'Total timesteps: {rl_model.num_timesteps:,}')\n"
    "print('RL 모델 로드 OK ✅')"
))

cells.append(code(
    "# RL → ONNX  (값함수 헤드, Go nnCtxV2 호환)\n"
    "from train_rl import export_onnx_value\n"
    "\n"
    "export_onnx_value('model_rl.zip', 'model_rl.onnx')\n"
    "print(f\"model_rl.onnx  {os.path.getsize('model_rl.onnx')/1e6:.1f} MB\")"
))

cells.append(code(
    "# 최종 Drive 백업\n"
    "for f in ['model_rl.zip', 'model_rl.onnx']:\n"
    "    if os.path.exists(f):\n"
    "        shutil.copy(f, f'{DRIVE_FOLDER}/{f}')\n"
    "        print(f'💾 {DRIVE_FOLDER}/{f}')\n"
    "!ls -lh {DRIVE_FOLDER}"
))

# ── Phase 4 ───────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 🏆 Phase 4: 로컬 평가\n"
    "\n"
    "Drive에서 `model_rl.onnx` 다운로드 후 로컬 PC에서:\n"
    "```bash\n"
    "go run -tags nn ./cmd/apple-game -nn-ana -n 100 -model models/model_rl.onnx -ort-lib runtime/onnxruntime.dll\n"
    "```\n"
    "\n"
    "| 지표 | 설명 |\n"
    "|------|------|\n"
    "| AnA baseline | ~131.6 (greedy 탐색) |\n"
    "| ModelAnA | RL 값함수로 greedy 탐색 |\n"
    "| 목표 | ModelAnA mean score > AnA |\n"
))

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "accelerator": "GPU",
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

import os as _os
out = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"✅ {out}  ({len(cells)} cells)")
