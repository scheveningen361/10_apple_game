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
from torch.utils.data import DataLoader, Dataset, random_split

# ── Constants ──────────────────────────────────────────────────────────────────
ROWS, COLS = 10, 17
NCELLS = ROWS * COLS           # 170
RECORD_BYTES = NCELLS + 1      # 171: 170 board bytes + 1 score byte
MAX_SCORE = 170.0              # for normalisation
MAX_RECTS = 8415.0             # for aux feature normalisation
MAX_SUM   = NCELLS * 9.0      # 1530

# ── Dataset ────────────────────────────────────────────────────────────────────

class SLDataset(Dataset):
    """
    Reads the flat binary file produced by `go run . -gen-sl`.
    Each record: 171 bytes = [170 uint8 board values] [1 uint8 greedy score].
    Returns (board_tensor, aux_tensor, label).
    """

    def __init__(self, path: str):
        data = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
        if data.size % RECORD_BYTES != 0:
            raise ValueError(
                f"File size {data.size} not divisible by {RECORD_BYTES}"
            )
        n = data.size // RECORD_BYTES
        data = data.reshape(n, RECORD_BYTES)
        self.boards = data[:, :NCELLS].astype(np.float32) / 9.0  # (N, 170)
        self.scores = data[:, NCELLS].astype(np.float32)          # (N,)  raw 0-170
        self.n = n
        print(f"Loaded {n:,} records from {path}")
        print(f"  Score range: {self.scores.min():.0f} – {self.scores.max():.0f}"
              f"  mean: {self.scores.mean():.2f}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        board_flat = self.boards[idx]          # (170,) float32
        board_2d   = board_flat.reshape(1, ROWS, COLS)  # (1,10,17)

        # Auxiliary features (hand-crafted global statistics):
        #   ① remaining non-zero cells / 170
        #   ② valid-rect count proxy: nz*(nz-1)/2 / MAX_RECTS  (cheap approx)
        #   ③ remaining cell sum / MAX_SUM
        raw = self.boards[idx] * 9.0           # un-normalise to 0-9
        nz  = float((raw > 0).sum()) / NCELLS
        s   = float(raw.sum()) / MAX_SUM
        # Valid-rect count proxy: use nz count as simple feature
        # (exact count too expensive to compute in Python; model will learn)
        aux = np.array([nz, nz * (nz - 1 / NCELLS), s], dtype=np.float32)

        label = self.scores[idx] / MAX_SCORE   # normalise to [0,1]
        return board_2d, aux, label


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

    # Dataset
    dataset = SLDataset(args.data)
    val_n   = max(1, int(len(dataset) * args.val))
    trn_n   = len(dataset) - val_n
    trn_ds, val_ds = random_split(
        dataset, [trn_n, val_n],
        generator=torch.Generator().manual_seed(42)
    )
    trn_loader = DataLoader(
        trn_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch * 2, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda")
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
