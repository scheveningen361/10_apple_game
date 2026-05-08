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
from torch.utils.data import DataLoader, TensorDataset, random_split

# ── Constants ──────────────────────────────────────────────────────────────────
ROWS, COLS = 10, 17
NCELLS = ROWS * COLS           # 170
RECORD_BYTES = NCELLS + 1      # 171: 170 board bytes + 1 score byte
MAX_SCORE = 170.0              # for normalisation
MAX_RECTS = 8415.0             # for aux feature normalisation
MAX_SUM   = NCELLS * 9.0      # 1530

# ── Dataset ────────────────────────────────────────────────────────────────────

def load_sl_dataset(path: str) -> TensorDataset:
    """
    Reads the flat binary file produced by `go run . -gen-sl`.
    각 레코드: 171 bytes = [170 uint8 board values] [1 uint8 greedy score].

    aux 특징을 로드 시점에 전부 벡터 연산으로 사전 계산 → __getitem__ 오버헤드 제거.
    모든 데이터를 Tensor로 변환해 TensorDataset 반환 (가장 빠른 DataLoader 경로).
    """
    raw = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
    if raw.size % RECORD_BYTES != 0:
        raise ValueError(f"File size {raw.size} not divisible by {RECORD_BYTES}")

    n    = raw.size // RECORD_BYTES
    data = raw.reshape(n, RECORD_BYTES)

    boards_u8 = data[:, :NCELLS]                         # (N, 170) uint8  0-9
    scores_f  = data[:, NCELLS].astype(np.float32)       # (N,)     0-170

    print(f"Loaded {n:,} records from {path}")
    print(f"  Score range: {scores_f.min():.0f} – {scores_f.max():.0f}"
          f"  mean: {scores_f.mean():.2f}")

    # ── Aux 특징 사전 계산 (벡터 연산, 한 번만) ─────────────────────────────────
    boards_f = boards_u8.astype(np.float32)              # 0-9
    nz = (boards_u8 > 0).sum(axis=1).astype(np.float32) / NCELLS   # (N,)
    s  = boards_f.sum(axis=1) / MAX_SUM                             # (N,)
    aux = np.stack([nz, nz * (nz - 1.0 / NCELLS), s], axis=1)      # (N, 3)

    # ── Tensor 변환 ────────────────────────────────────────────────────────────
    board_t = torch.from_numpy(boards_f / 9.0).reshape(n, 1, ROWS, COLS)  # (N,1,10,17)
    aux_t   = torch.from_numpy(aux.astype(np.float32))                     # (N, 3)
    label_t = torch.from_numpy(scores_f / MAX_SCORE)                       # (N,)

    return TensorDataset(board_t, aux_t, label_t)


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


# ── Training helpers ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for boards, aux, labels in loader:
        boards = boards.to(device)
        aux    = aux.to(device)
        labels = labels.to(device)
        pred   = model(boards, aux)
        loss   = F.mse_loss(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n += len(labels)
    return math.sqrt(total_loss / n) * MAX_SCORE   # RMSE in raw score units


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for boards, aux, labels in loader:
        boards = boards.to(device)
        aux    = aux.to(device)
        labels = labels.to(device)
        pred   = model(boards, aux)
        loss   = F.mse_loss(pred, labels)
        total_loss += loss.item() * len(labels)
        n += len(labels)
    return math.sqrt(total_loss / n) * MAX_SCORE   # RMSE in raw score units


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
    parser.add_argument("--workers", type=int,   default=4,
                        help="DataLoader num_workers")
    parser.add_argument("--resume",  default="", help="resume from checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Dataset — aux 사전 계산 + TensorDataset으로 __getitem__ 오버헤드 제거
    dataset = load_sl_dataset(args.data)
    val_n   = max(1, int(len(dataset) * args.val))
    trn_n   = len(dataset) - val_n
    trn_ds, val_ds = random_split(
        dataset, [trn_n, val_n],
        generator=torch.Generator().manual_seed(42)
    )
    # TensorDataset은 num_workers=0 이 가장 빠름 (이미 메모리에 올라있음)
    trn_loader = DataLoader(
        trn_ds, batch_size=args.batch, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch * 2, shuffle=False,
        num_workers=0, pin_memory=False
    )

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
        trn_rmse = train_epoch(model, trn_loader, optimizer, device)
        val_rmse = eval_epoch (model, val_loader,             device)
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
