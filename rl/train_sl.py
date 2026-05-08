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

def load_sl_tensors(path: str):
    """
    Reads sl_data.bin and returns (board_t, aux_t, label_t) as CPU float32 tensors.
    aux 특징을 numpy 벡터 연산으로 한 번에 계산 → per-sample 오버헤드 없음.

    Returns:
        board_t : (N, 1, 10, 17) float32
        aux_t   : (N, 3)         float32
        label_t : (N,)           float32  [0, 1] 정규화
    """
    raw = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
    if raw.size % RECORD_BYTES != 0:
        raise ValueError(f"File size {raw.size} not divisible by {RECORD_BYTES}")

    n    = raw.size // RECORD_BYTES
    data = raw.reshape(n, RECORD_BYTES)

    boards_u8 = data[:, :NCELLS]                          # (N, 170) uint8
    scores_f  = data[:, NCELLS].astype(np.float32)        # (N,)

    print(f"Loaded {n:,} records from {path}")
    print(f"  Score range: {scores_f.min():.0f} – {scores_f.max():.0f}"
          f"  mean: {scores_f.mean():.2f}")

    # aux 사전 계산 (벡터 연산)
    boards_f = boards_u8.astype(np.float32)
    nz = (boards_u8 > 0).sum(axis=1).astype(np.float32) / NCELLS
    s  = boards_f.sum(axis=1) / MAX_SUM
    aux = np.stack([nz, nz * (nz - 1.0 / NCELLS), s], axis=1).astype(np.float32)

    board_t = torch.from_numpy(boards_f / 9.0).reshape(n, 1, ROWS, COLS)
    aux_t   = torch.from_numpy(aux)
    label_t = torch.from_numpy(scores_f / MAX_SCORE)

    return board_t, aux_t, label_t


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

def train_epoch(model, boards, aux, labels, batch_size, optimizer):
    """boards/aux/labels 는 이미 GPU 위에 있는 텐서."""
    model.train()
    n          = len(labels)
    perm       = torch.randperm(n, device=boards.device)
    total_loss = 0.0
    steps      = 0
    for start in range(0, n - batch_size + 1, batch_size):
        idx   = perm[start: start + batch_size]
        pred  = model(boards[idx], aux[idx])
        loss  = F.mse_loss(pred, labels[idx])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        steps      += 1
    return math.sqrt(total_loss / steps) * MAX_SCORE   # RMSE in raw units


@torch.no_grad()
def eval_epoch(model, boards, aux, labels, batch_size):
    """boards/aux/labels 는 이미 GPU 위에 있는 텐서."""
    model.eval()
    n          = len(labels)
    total_loss = 0.0
    steps      = 0
    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        pred = model(boards[start:end], aux[start:end])
        loss = F.mse_loss(pred, labels[start:end])
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
    parser.add_argument("--batch",   type=int,   default=2048)
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

    # 데이터 로드 + GPU 전송 (1회만 발생)
    board_t, aux_t, label_t = load_sl_tensors(args.data)
    n     = len(label_t)
    val_n = max(1, int(n * args.val))
    trn_n = n - val_n

    # 재현 가능한 분할
    perm_split = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    trn_idx = perm_split[:trn_n]
    val_idx = perm_split[trn_n:]

    print(f"  Train: {trn_n:,}  Val: {val_n:,}")
    print(f"  GPU로 전송 중...", end=" ", flush=True)
    t_load = time.time()
    trn_board  = board_t[trn_idx].to(device)
    trn_aux    = aux_t  [trn_idx].to(device)
    trn_label  = label_t[trn_idx].to(device)
    val_board  = board_t[val_idx].to(device)
    val_aux    = aux_t  [val_idx].to(device)
    val_label  = label_t[val_idx].to(device)
    print(f"완료 ({time.time()-t_load:.1f}s)")

    # 메모리 사용량 출력
    if device == "cuda":
        used_gb = torch.cuda.memory_allocated() / 1e9
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

    print(f"\n{'Ep':>4}  {'Trn RMSE':>10}  {'Val RMSE':>10}  {'LR':>8}  {'Time':>8}")
    print("-" * 50)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        trn_rmse = train_epoch(model, trn_board, trn_aux, trn_label,
                               args.batch, optimizer)
        val_rmse = eval_epoch (model, val_board, val_aux, val_label,
                               args.batch * 2)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"{epoch+1:4d}  {trn_rmse:10.4f}  {val_rmse:10.4f}  "
              f"{lr_now:8.2e}  {elapsed:7.1f}s")

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
