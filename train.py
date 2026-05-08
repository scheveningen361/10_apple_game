"""
Apple Game – Value Network Training (Phase 2)
==============================================
Board : 10 × 17 = 170 cells, each value 0–9
Label : number of apples Greedy removes from this board state (0–170)

Binary data format (171 bytes / record):
  bytes  0–169 : board cell values uint8  (0=removed, 1-9=alive)
  byte   170   : label uint8

Usage (Colab):
  from google.colab import files
  files.upload()          # upload training_data.bin
  !python train.py --data training_data.bin --epochs 20 --out model.pt
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import time, os

# ── Constants ────────────────────────────────────────────────────────────────
ROWS, COLS = 10, 17
N_CELLS    = ROWS * COLS   # 170

# ── Dataset ──────────────────────────────────────────────────────────────────
class AppleDataset(Dataset):
    """
    Reads the binary file produced by `go run main.go -gen`.

    X : float32 tensor (N, 1, 10, 17)  — board as 2D image, normalised 0–1
    y : float32 tensor (N,)            — label / N_CELLS  (regression in [0,1])
    """
    def __init__(self, path: str):
        raw = np.fromfile(path, dtype=np.uint8)
        assert raw.size % 171 == 0, "File size is not a multiple of 171"
        data  = raw.reshape(-1, 171)
        board = data[:, :N_CELLS].astype(np.float32) / 9.0   # normalise to [0,1]
        label = data[:, N_CELLS].astype(np.float32) / N_CELLS # regression target in [0,1]

        # Reshape board → (N, 1, 10, 17) for CNN
        self.X = torch.from_numpy(board.reshape(-1, 1, ROWS, COLS))
        self.y = torch.from_numpy(label)
        print(f"Loaded {len(self.X):,} records from {path}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ── Model ─────────────────────────────────────────────────────────────────────
class ValueNet(nn.Module):
    """
    Lightweight CNN for predicting Greedy value from board state.

    Architecture:
      Conv block 1 : 1  → 32  channels, 3×3 padding=1 → BN → ReLU
      Conv block 2 : 32 → 64  channels, 3×3 padding=1 → BN → ReLU
      Conv block 3 : 64 → 128 channels, 3×3 padding=1 → BN → ReLU
      Global Average Pooling
      FC 128 → 64 → 1  (sigmoid output, predicts label/N_CELLS)
    """
    def __init__(self):
        super().__init__()
        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_block(1,   32),
            conv_block(32,  64),
            conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # → (N, 128, 1, 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),   # output in (0, 1)
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x))).squeeze(1)

# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    ds = AppleDataset(args.data)
    val_size  = int(len(ds) * 0.1)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(0))

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=(device.type=="cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=args.batch*2, shuffle=False,
                          num_workers=args.workers, pin_memory=(device.type=="cuda"))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ValueNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    history  = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= train_size

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= val_size

        scheduler.step()
        elapsed = time.time() - t0

        # Convert MSE on [0,1] scale → RMSE in apples (multiply back by N_CELLS)
        train_rmse = (train_loss ** 0.5) * N_CELLS
        val_rmse   = (val_loss   ** 0.5) * N_CELLS
        lr_now     = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_rmse={train_rmse:.3f}  val_rmse={val_rmse:.3f}  "
              f"lr={lr_now:.2e}  {elapsed:.1f}s")

        history.append({"epoch": epoch,
                         "train_rmse": train_rmse,
                         "val_rmse": val_rmse})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "val_rmse": val_rmse,
                        "args": vars(args)},
                       args.out)
            print(f"  → saved best model  (val_rmse={val_rmse:.3f})")

    print(f"\nBest val_rmse : {(best_val**0.5)*N_CELLS:.3f} apples")
    print(f"Model saved   : {args.out}")
    return history

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Apple Game Value Network")
    parser.add_argument("--data",    default="training_data.bin", help="binary training file")
    parser.add_argument("--out",     default="model.pt",          help="output checkpoint path")
    parser.add_argument("--epochs",  type=int,   default=20)
    parser.add_argument("--batch",   type=int,   default=512)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--workers", type=int,   default=2)
    args = parser.parse_args()

    train(args)
