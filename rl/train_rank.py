"""
Train a candidate reranker for Apple Game.

Data:
    go run ./cmd/apple-game -gen-rank 500 -rank-out data/generated/rank_data.bin

Training:
    python rl/train_rank.py --data data/generated/rank_data.bin --out models/model_rank.pt --export models/model_rank.onnx

The model predicts a raw within-group rank/advantage score for a candidate's
next board. Go uses it only to rerank the top heuristic candidates, so the
absolute scale is intentionally unimportant.
"""

import argparse
import importlib.util
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROWS, COLS = 10, 17
NCELLS = ROWS * COLS
MAX_SCORE = 170.0
MAX_SUM = NCELLS * 9.0
CAND_BYTES = NCELLS + 2  # removed, target, board[170]


@dataclass
class RankData:
    boards: np.ndarray
    labels: np.ndarray
    removed: np.ndarray
    offsets: np.ndarray
    group_max: np.ndarray


def load_rank_data(path: str, val_frac: float = 0.05, seed: int = 42):
    t0 = time.time()
    boards_chunks, labels_chunks, removed_chunks = [], [], []
    offsets = [0]

    with open(path, "rb") as f:
        while True:
            hdr = f.read(2)
            if not hdr:
                break
            if len(hdr) != 2:
                raise ValueError("truncated group header")
            n = int.from_bytes(hdr, "little")
            payload = f.read(n * CAND_BYTES)
            if len(payload) != n * CAND_BYTES:
                raise ValueError("truncated candidate payload")
            arr = np.frombuffer(payload, dtype=np.uint8).reshape(n, CAND_BYTES)
            removed_chunks.append(arr[:, 0].copy())
            labels_chunks.append(arr[:, 1].copy())
            boards_chunks.append(arr[:, 2:].copy())
            offsets.append(offsets[-1] + n)

    boards = np.ascontiguousarray(np.vstack(boards_chunks))
    labels = np.ascontiguousarray(np.concatenate(labels_chunks))
    removed = np.ascontiguousarray(np.concatenate(removed_chunks))
    offsets = np.asarray(offsets, dtype=np.int64)

    group_max = np.empty(len(offsets) - 1, dtype=np.uint8)
    for g in range(len(group_max)):
        group_max[g] = labels[offsets[g]:offsets[g + 1]].max()

    rng = np.random.default_rng(seed)
    groups = np.arange(len(group_max))
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * val_frac))
    val_groups = np.sort(groups[:n_val])
    trn_groups = np.sort(groups[n_val:])

    data = RankData(boards, labels, removed, offsets, group_max)
    print(
        f"Loaded {path}: groups={len(group_max):,} candidates={len(labels):,} "
        f"label={labels.min()}..{labels.max()} mean={labels.mean():.2f} "
        f"time={time.time() - t0:.1f}s"
    )
    return data, trn_groups, val_groups


def aux_from_boards_u8(boards_u8: np.ndarray) -> np.ndarray:
    boards_f = boards_u8.astype(np.float32)
    nz_count = (boards_u8 > 0).sum(axis=1).astype(np.float32)
    nz = nz_count / NCELLS
    cell_sum = boards_f.sum(axis=1) / MAX_SUM
    return np.stack(
        [nz, nz_count * (nz_count - 1.0) / (NCELLS * NCELLS), cell_sum],
        axis=1,
    ).astype(np.float32)


def group_buckets(group_max: np.ndarray, group_ids: np.ndarray, n_buckets: int):
    lo, hi = int(group_max[group_ids].min()), int(group_max[group_ids].max())
    edges = np.linspace(lo, hi + 1, n_buckets + 1)
    buckets = []
    for i in range(n_buckets):
        mask = (group_max[group_ids] >= edges[i]) & (group_max[group_ids] < edges[i + 1])
        b = group_ids[mask]
        if len(b):
            buckets.append(b)
    return buckets


def sample_group_batch(rng, buckets, batch_groups: int):
    chosen = []
    while len(chosen) < batch_groups:
        for bucket in buckets:
            if len(chosen) >= batch_groups:
                break
            chosen.append(int(bucket[rng.integers(len(bucket))]))
    rng.shuffle(chosen)
    return chosen


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


class AppleRankNet(nn.Module):
    def __init__(self, channels: int = 96, n_blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels + 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, board, aux):
        feat = self.gap(self.blocks(self.stem(board))).flatten(1)
        return self.fc(torch.cat([feat, aux], dim=1)).squeeze(-1)


def prep_batch(data: RankData, group_ids, device):
    idx_chunks, spans = [], []
    pos = 0
    for g in group_ids:
        s, e = int(data.offsets[g]), int(data.offsets[g + 1])
        idx = np.arange(s, e, dtype=np.int64)
        idx_chunks.append(idx)
        spans.append((pos, pos + len(idx)))
        pos += len(idx)

    idx = np.concatenate(idx_chunks)
    boards_u8 = data.boards[idx]
    aux = aux_from_boards_u8(boards_u8)
    board = torch.from_numpy(boards_u8).to(device).float().mul_(1.0 / 9.0).reshape(-1, 1, ROWS, COLS)
    aux_t = torch.from_numpy(aux).to(device)
    labels = torch.from_numpy(data.labels[idx].astype(np.float32)).to(device)
    return board, aux_t, labels, spans


def rank_loss(pred, labels, spans, temp: float, margin: float):
    losses = []
    top1_ok, regret_sum = 0, 0.0
    for s, e in spans:
        p = pred[s:e]
        y = labels[s:e]
        adv = (y - y.mean()) / MAX_SCORE
        mse = F.mse_loss(p, adv)

        target_prob = F.softmax((y - y.max()) / temp, dim=0)
        listwise = -(target_prob * F.log_softmax(p / temp, dim=0)).sum()

        best_y = y.max()
        best_idx = torch.argmax(y)
        pred_idx = torch.argmax(p)
        pair = F.relu(margin - (p[best_idx] - p)).mean()

        losses.append(mse + listwise + 0.25 * pair)
        top1_ok += int(y[pred_idx] == best_y)
        regret_sum += float(best_y - y[pred_idx])
    return torch.stack(losses).mean(), top1_ok, regret_sum


def run_epoch(model, data, group_ids, buckets, args, optimizer, scaler, device, train: bool):
    rng = np.random.default_rng(args.seed + int(time.time()) % 100000)
    n_steps = max(1, len(group_ids) // args.groups_per_batch)
    total_loss, total_top1, total_regret, total_groups = 0.0, 0, 0.0, 0

    model.train(train)
    for _ in range(n_steps):
        batch_groups = sample_group_batch(rng, buckets, args.groups_per_batch) if train else list(
            rng.choice(group_ids, size=min(args.groups_per_batch, len(group_ids)), replace=False)
        )
        board, aux, labels, spans = prep_batch(data, batch_groups, device)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                pred = model(board, aux)
                loss, top1, regret = rank_loss(pred, labels, spans, args.temp, args.margin)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.detach())
        total_top1 += top1
        total_regret += regret
        total_groups += len(spans)

    return {
        "loss": total_loss / n_steps,
        "top1": total_top1 / max(1, total_groups),
        "regret": total_regret / max(1, total_groups),
    }


def export_onnx(model, path, device):
    if importlib.util.find_spec("onnx") is None:
        raise RuntimeError("ONNX export requires the Python package 'onnx'. Install it with: pip install onnx")

    model.eval().to(device)
    dummy_board = torch.zeros(1, 1, ROWS, COLS, device=device)
    dummy_aux = torch.zeros(1, 3, device=device)
    torch.onnx.export(
        model,
        (dummy_board, dummy_aux),
        path,
        input_names=["board", "aux"],
        output_names=["score"],
        dynamic_axes={"board": {0: "batch"}, "aux": {0: "batch"}, "score": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX exported: {path}")


def load_rank_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config", {})
    model = AppleRankNet(cfg.get("channels", 96), cfg.get("blocks", 4)).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/generated/rank_data.bin")
    parser.add_argument("--out", default="models/model_rank.pt")
    parser.add_argument("--export", default="")
    parser.add_argument("--export-only", action="store_true", help="load --out and export ONNX without training")
    parser.add_argument("--resume", default="", help="checkpoint to resume from; --epochs is total target epochs")
    parser.add_argument("--resume-epoch", type=int, default=-1, help="last completed epoch for old checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--groups-per-batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--val", type=float, default=0.05)
    parser.add_argument("--temp", type=float, default=4.0)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.export_only:
        if not args.export:
            raise ValueError("--export-only requires --export")
        model, ckpt = load_rank_checkpoint(args.out, device)
        print(f"Loaded checkpoint: {args.out}  best_regret={ckpt.get('best_regret', 'n/a')}")
        os.makedirs(os.path.dirname(args.export) or ".", exist_ok=True)
        export_onnx(model, args.export, "cpu")
        return

    data, trn_groups, val_groups = load_rank_data(args.data, args.val, args.seed)
    trn_buckets = group_buckets(data.group_max, trn_groups, args.buckets)
    val_buckets = [val_groups]

    model = AppleRankNet(args.channels, args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    best_regret = float("inf")
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        cfg = ckpt.get("config", {})
        ckpt_channels = cfg.get("channels", args.channels)
        ckpt_blocks = cfg.get("blocks", args.blocks)
        if ckpt_channels != args.channels or ckpt_blocks != args.blocks:
            raise ValueError(
                f"resume checkpoint model is channels={ckpt_channels}, blocks={ckpt_blocks}; "
                f"current args are channels={args.channels}, blocks={args.blocks}"
            )
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        best_regret = ckpt.get("best_regret", best_regret)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        if start_epoch == 0:
            start_epoch = int(ckpt.get("epochs_done", 0))
        if args.resume_epoch >= 0:
            start_epoch = args.resume_epoch
        print(
            f"Resumed: {args.resume}  start_epoch={start_epoch + 1}  "
            f"best_regret={best_regret}"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"Device: {device}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'Ep':>3} {'trn_loss':>9} {'trn_top1':>9} {'trn_reg':>8} {'val_top1':>9} {'val_reg':>8} {'time':>7}")

    if start_epoch >= args.epochs:
        print(f"Nothing to train: checkpoint is already at epoch {start_epoch}, target epochs={args.epochs}")

    for ep in range(start_epoch, args.epochs):
        t0 = time.time()
        trn = run_epoch(model, data, trn_groups, trn_buckets, args, optimizer, scaler, device, True)
        with torch.no_grad():
            val = run_epoch(model, data, val_groups, val_buckets, args, optimizer, scaler, device, False)
        scheduler.step()
        print(
            f"{ep+1:3d} {trn['loss']:9.4f} {trn['top1']:9.3f} {trn['regret']:8.3f} "
            f"{val['top1']:9.3f} {val['regret']:8.3f} {time.time()-t0:6.1f}s"
        )
        if val["regret"] < best_regret:
            best_regret = val["regret"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": ep,
                    "epochs_done": ep + 1,
                    "best_regret": best_regret,
                    "config": {"channels": args.channels, "blocks": args.blocks},
                },
                args.out,
            )
            print(f"    saved: val regret {best_regret:.3f}")

    model, _ = load_rank_checkpoint(args.out, device)
    if args.export:
        os.makedirs(os.path.dirname(args.export) or ".", exist_ok=True)
        export_onnx(model, args.export, "cpu")


if __name__ == "__main__":
    main()
