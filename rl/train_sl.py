"""
Phase 2: Supervised Learning — AnA behaviour cloning
=====================================================
Trains AppleNetSL to predict playGreedy(b2) from board state b2.
Data produced by: go run . -gen-sl 500 -sl-out sl_data.bin

Usage (Colab T4 or local GPU):
    python train_sl.py --data sl_data.bin --out model_sl.pt
    python train_sl.py --data sl_data.bin --out model_sl.pt --epochs 50 --batch 2048

After training, export to ONNX:
    python train_sl.py --data sl_data.bin --out model_sl.pt --export model_sl.onnx
"""

import argparse
import math
import os
import struct
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ──────────────────────────────────────────────────────────────────
ROWS, COLS = 10, 17
NCELLS = ROWS * COLS           # 170
RECORD_BYTES = NCELLS + 1      # 171: 170 board bytes + 1 score byte
MAX_SCORE = 170.0              # for normalisation
MAX_RECTS = 8415.0             # for aux feature normalisation
MAX_SUM   = NCELLS * 9.0      # 1530

# ── Dataset ────────────────────────────────────────────────────────────────────

def _tick(label: str, t0: float) -> float:
    t1 = time.time()
    print(f"  [{t1-t0:6.2f}s] {label}")
    return t1


def load_sl_tensors(path: str, val_frac: float = 0.05, seed: int = 42):
    print(f"\n[데이터 로딩 시작] {path}")
    t = time.time()

    # 1. 파일 읽기
    raw = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
    t = _tick(f"파일 읽기  ({raw.nbytes/1e6:.0f} MB)", t)

    if raw.size % RECORD_BYTES != 0:
        raise ValueError(f"File size {raw.size} not divisible by {RECORD_BYTES}")
    n    = raw.size // RECORD_BYTES
    data = raw.reshape(n, RECORD_BYTES)

    boards_u8 = data[:, :NCELLS]
    scores_u8 = data[:, NCELLS]
    print(f"  records: {n:,}  score {int(scores_u8.min())}–{int(scores_u8.max())}"
          f"  mean {scores_u8.astype(np.float32).mean():.2f}")
    t = _tick("슬라이싱", t)

    # 2. aux 계산
    boards_f = boards_u8.astype(np.float32)
    t = _tick(f"uint8→float32 변환  ({boards_f.nbytes/1e6:.0f} MB)", t)

    nz  = (boards_u8 > 0).sum(axis=1).astype(np.float32) / NCELLS
    s   = boards_f.sum(axis=1) / MAX_SUM
    aux = np.stack([nz, nz * (nz - 1.0 / NCELLS), s], axis=1).astype(np.float32)
    del boards_f
    t = _tick(f"aux 계산  ({aux.nbytes/1e6:.0f} MB)", t)

    # 3. 셔플
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(n)
    t = _tick("permutation 생성", t)

    boards_u8 = np.ascontiguousarray(boards_u8[perm])
    t = _tick(f"boards 셔플  ({boards_u8.nbytes/1e6:.0f} MB)", t)

    scores_u8 = np.ascontiguousarray(scores_u8[perm])
    aux       = np.ascontiguousarray(aux[perm])
    t = _tick("scores/aux 셔플", t)

    # 4. 분할 + 텐서 변환
    trn_n = n - max(1, int(n * val_frac))

    def make(b, a, s):
        return (torch.from_numpy(b.copy()),
                torch.from_numpy(a.copy()),
                torch.from_numpy(s.copy()))

    trn = make(boards_u8[:trn_n], aux[:trn_n], scores_u8[:trn_n])
    val = make(boards_u8[trn_n:], aux[trn_n:], scores_u8[trn_n:])
    t = _tick(f"텐서 변환  train {trn_n:,}  val {n-trn_n:,}", t)

    print()
    return trn, val


# ── Model ──────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class AppleNetSL(nn.Module):
    """
    Input : board  (B, 1, 10, 17) float32  (already /9.0)
            aux    (B, 3)           float32
    Output: predicted Greedy score (B,) float32  in [0,1]  (× MAX_SCORE to get raw)
    ~1.8 M parameters with channels=128, n_blocks=6.
    """

    def __init__(self, channels: int = 128, n_blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
        self.gap    = nn.AdaptiveAvgPool2d(1)   # → (B, channels, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels + 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        feat     = self.gap(self.blocks(self.stem(x))).flatten(1)  # (B, ch)
        combined = torch.cat([feat, aux], dim=1)                   # (B, ch+3)
        return self.fc(combined).squeeze(-1)                       # (B,)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Training helpers (GPU-resident 텐서 직접 배치) ────────────────────────────
#
# DataLoader 없이 GPU 위의 텐서를 torch.randperm으로 셔플 후 슬라이싱.
# CPU↔GPU 전송이 epoch당 딱 1회(최초 .to(device))만 발생 → 수십 배 빠름.

def _prep(boards_u8, aux, labels_u8):
    """uint8 board/label → float32, reshape board. GPU 위에서 실행."""
    b = boards_u8.float().mul_(1.0 / 9.0).reshape(-1, 1, ROWS, COLS)
    l = labels_u8.float().mul_(1.0 / MAX_SCORE)
    return b, aux, l


def train_epoch(model, boards, aux, labels, batch_size, optimizer, scaler):
    """boards: (N,170) uint8 GPU  aux: (N,3) float32 GPU  labels: (N,) uint8 GPU"""
    model.train()
    n          = len(labels)
    perm       = torch.randperm(n, device=boards.device)
    total_loss = 0.0
    steps      = 0
    for start in range(0, n - batch_size + 1, batch_size):
        idx     = perm[start: start + batch_size]
        b, a, l = _prep(boards[idx], aux[idx], labels[idx])
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            pred = model(b, a)
            loss = F.mse_loss(pred, l)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        steps      += 1
    return math.sqrt(total_loss / steps) * MAX_SCORE


@torch.no_grad()
def eval_epoch(model, boards, aux, labels, batch_size):
    """boards: (N,170) uint8 GPU  aux: (N,3) float32 GPU  labels: (N,) uint8 GPU"""
    model.eval()
    n          = len(labels)
    total_loss = 0.0
    steps      = 0
    for start in range(0, n, batch_size):
        end     = min(start + batch_size, n)
        b, a, l = _prep(boards[start:end], aux[start:end], labels[start:end])
        with torch.amp.autocast("cuda"):
            pred = model(b, a)
            loss = F.mse_loss(pred, l)
        total_loss += loss.item()
        steps      += 1
    return math.sqrt(total_loss / steps) * MAX_SCORE


# ── ONNX export ────────────────────────────────────────────────────────────────

def export_onnx(model: AppleNetSL, path: str, device: str = "cpu"):
    model.eval().to(device)
    dummy_board = torch.zeros(1, 1, ROWS, COLS, device=device)
    dummy_aux   = torch.zeros(1, 3,             device=device)
    torch.onnx.export(
        model,
        (dummy_board, dummy_aux),
        path,
        input_names  = ["board", "aux"],
        output_names = ["score"],
        dynamic_axes = {
            "board": {0: "batch"},
            "aux"  : {0: "batch"},
            "score": {0: "batch"},
        },
        opset_version = 17,
    )
    print(f"ONNX model exported: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="sl_data.bin",  help="binary data file")
    parser.add_argument("--out",     default="model_sl.pt",  help="output .pt checkpoint")
    parser.add_argument("--export",  default="",             help="also export to ONNX at this path")
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--batch",   type=int,   default=4096)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--wd",      type=float, default=1e-4)
    parser.add_argument("--channels",type=int,   default=128)
    parser.add_argument("--blocks",  type=int,   default=6)
    parser.add_argument("--val",     type=float, default=0.05,
                        help="validation fraction")
    parser.add_argument("--resume",  default="", help="resume from checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 데이터 로드 (numpy에서 셔플+분할, uint8 유지)
    (trn_b, trn_a, trn_l), (val_b, val_a, val_l) = \
        load_sl_tensors(args.data, val_frac=args.val)

    # GPU 전송 (단계별 타이밍)
    print("[GPU 전송]")
    t = time.time()
    trn_board = trn_b.to(device); t = _tick(f"trn_board  {trn_b.nbytes/1e6:.0f} MB", t)
    trn_aux   = trn_a.to(device); t = _tick(f"trn_aux    {trn_a.nbytes/1e6:.0f} MB", t)
    trn_label = trn_l.to(device); t = _tick(f"trn_label  {trn_l.nbytes/1e6:.0f} MB", t)
    val_board = val_b.to(device); t = _tick(f"val_board  {val_b.nbytes/1e6:.0f} MB", t)
    val_aux   = val_a.to(device); t = _tick(f"val_aux    {val_a.nbytes/1e6:.0f} MB", t)
    val_label = val_l.to(device); t = _tick(f"val_label  {val_l.nbytes/1e6:.0f} MB", t)

    if device == "cuda":
        used_gb  = torch.cuda.memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM 사용: {used_gb:.2f} / {total_gb:.1f} GB")

    # Model
    model = AppleNetSL(channels=args.channels, n_blocks=args.blocks).to(device)
    print(f"Model: {count_params(model):,} parameters  "
          f"(channels={args.channels}, blocks={args.blocks})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    start_epoch = 0
    best_val    = float("inf")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", float("inf"))
        print(f"Resumed from epoch {start_epoch}  best_val_rmse={best_val:.4f}")

    amp_str = "AMP(FP16)" if device == "cuda" else "FP32"
    print(f"Mixed precision: {amp_str}")
    print(f"\n{'Ep':>4}  {'Trn RMSE':>10}  {'Val RMSE':>10}  {'LR':>8}  {'Time':>8}")
    print("-" * 50)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        trn_rmse = train_epoch(model, trn_board, trn_aux, trn_label,
                               args.batch, optimizer, scaler)
        t_trn = time.time()
        val_rmse = eval_epoch (model, val_board, val_aux, val_label,
                               args.batch * 2)
        t_val = time.time()
        scheduler.step()
        elapsed = t_val - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"{epoch+1:4d}  {trn_rmse:10.4f}  {val_rmse:10.4f}  "
              f"{lr_now:8.2e}  {elapsed:7.1f}s"
              f"  (trn {t_trn-t0:.1f}s  val {t_val-t_trn:.1f}s)")

        # Save best checkpoint
        if val_rmse < best_val:
            best_val = val_rmse
            torch.save({
                "epoch"     : epoch,
                "model"     : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "scheduler" : scheduler.state_dict(),
                "best_val"  : best_val,
                "config"    : {
                    "channels": args.channels,
                    "blocks"  : args.blocks,
                    "rows"    : ROWS, "cols": COLS,
                },
            }, args.out)
            print(f"      ↑ saved (val RMSE {best_val:.4f})")

    print(f"\nTraining complete. Best val RMSE: {best_val:.4f}")
    print(f"Checkpoint: {args.out}")

    # ONNX export
    if args.export:
        # Load best weights before export
        ckpt = torch.load(args.out, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        export_onnx(model, args.export, device="cpu")


if __name__ == "__main__":
    main()
